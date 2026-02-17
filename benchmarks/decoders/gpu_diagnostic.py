#!/usr/bin/env python3
"""
GPU decode diagnostic - isolates each layer of the ROCm video decode stack.

Tests each component independently to identify exactly where the failure occurs:
  1. HIP runtime + device info
  2. rocDecode library availability
  3. torchcodec CPU-only decode (baseline)
  4. torchcodec core API GPU decode (low-level)
  5. torchcodec VideoDecoder GPU decode (high-level)

Run:
    python3 gpu_diagnostic.py [video_path]
"""

import ctypes
import os
import subprocess
import sys
import traceback
from pathlib import Path


def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def test_hip_runtime():
    """Test 1: HIP runtime and device enumeration."""
    section("1. HIP Runtime & Device Info")

    import torch
    print(f"  torch.version.hip = {torch.version.hip}")
    print(f"  torch.cuda.is_available() = {torch.cuda.is_available()}")
    print(f"  torch.cuda.device_count() = {torch.cuda.device_count()}")

    if not torch.cuda.is_available():
        print("  FAIL: No GPU available")
        return False

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        vram = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
        print(f"  Device {i}: {props.name} ({props.gcnArchName}) "
              f"VRAM={vram // (1024**2)}MB")

    # Test basic HIP allocation on device 0
    print("\n  Testing HIP memory ops on cuda:0...", end=" ", flush=True)
    try:
        t = torch.zeros(1024, device="cuda:0")
        t += 1
        torch.cuda.synchronize()
        del t
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}")
        return False

    return True


def test_rocdecode_library():
    """Test 2: Check if librocdecode.so is loadable and has expected symbols."""
    section("2. rocDecode Library")

    lib_names = ["librocdecode.so", "librocdecode.so.0", "librocdecode.so.1"]
    lib = None
    loaded_name = None

    for name in lib_names:
        try:
            lib = ctypes.CDLL(name)
            loaded_name = name
            break
        except OSError:
            continue

    if lib is None:
        print("  FAIL: Could not load librocdecode.so")
        print("  Checked:", ", ".join(lib_names))

        # Check if the file exists anywhere
        try:
            result = subprocess.run(
                ["find", "/", "-name", "librocdecode*", "-type", "f"],
                capture_output=True, text=True, timeout=10
            )
            if result.stdout.strip():
                print("  Found files:")
                for line in result.stdout.strip().split("\n"):
                    print(f"    {line}")
            else:
                print("  No librocdecode files found on system")
        except Exception:
            pass

        # Check LD_LIBRARY_PATH
        print(f"  LD_LIBRARY_PATH = {os.environ.get('LD_LIBRARY_PATH', '(not set)')}")
        return False

    print(f"  Loaded: {loaded_name}")

    # Check for expected symbols
    expected_symbols = [
        "rocDecCreateDecoder",
        "rocDecDestroyDecoder",
        "rocDecGetDecoderCaps",
        "rocDecDecodeFrame",
        "rocDecGetVideoFrame",
        "rocDecGetErrorName",
        "rocDecCreateVideoParser",
        "rocDecParseVideoData",
        "rocDecDestroyVideoParser",
    ]

    missing = []
    for sym in expected_symbols:
        try:
            getattr(lib, sym)
        except AttributeError:
            missing.append(sym)

    if missing:
        print(f"  FAIL: Missing symbols: {missing}")
        return False

    print(f"  All {len(expected_symbols)} expected symbols found")

    # Try to check version
    try:
        result = subprocess.run(
            ["dpkg", "-l", "rocdecode*"],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.strip().split("\n"):
            if "rocdecode" in line.lower():
                print(f"  Package: {line.strip()}")
    except Exception:
        pass

    return True


def test_cpu_decode(video_path):
    """Test 3: CPU-only decode to establish baseline."""
    section("3. CPU Decode (baseline)")

    from torchcodec.decoders import VideoDecoder

    print(f"  Video: {video_path}")
    print("  Creating CPU decoder...", end=" ", flush=True)
    try:
        decoder = VideoDecoder(str(video_path), device="cpu")
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}")
        return False

    meta = decoder.metadata
    print(f"  Codec: {meta.codec}  Resolution: {meta.width}x{meta.height}")
    print(f"  Frames: {meta.num_frames}  FPS: {meta.average_fps:.1f}")

    print("  Decoding first frame on CPU...", end=" ", flush=True)
    try:
        frame = next(iter(decoder))
        print(f"OK (shape={frame.data.shape}, dtype={frame.data.dtype})")
    except Exception as e:
        print(f"FAIL: {e}")
        return False

    print("  Decoding 10 frames on CPU...", end=" ", flush=True)
    try:
        decoder2 = VideoDecoder(str(video_path), device="cpu")
        count = 0
        for f in decoder2:
            count += 1
            if count >= 10:
                break
        print(f"OK ({count} frames)")
    except Exception as e:
        print(f"FAIL: {e}")
        return False

    return True


def test_core_api_gpu(video_path):
    """Test 4: Core API GPU decode (low-level, matches existing gpu_benchmark.py)."""
    section("4. Core API GPU Decode")

    import torch
    import torchcodec._core

    if not torch.cuda.is_available():
        print("  SKIP: No GPU")
        return False

    print("  Creating decoder from file...", end=" ", flush=True)
    try:
        decoder = torchcodec._core.create_from_file(str(video_path))
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}")
        traceback.print_exc()
        return False

    print("  Adding video stream on cuda:0...", end=" ", flush=True)
    try:
        torchcodec._core._add_video_stream(
            decoder,
            stream_index=-1,
            device="cuda:0",
            num_threads=1,
        )
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}")
        traceback.print_exc()
        return False

    print("  Decoding first frame via core API...", end=" ", flush=True)
    try:
        frame, *_ = torchcodec._core.get_next_frame(decoder)
        torch.cuda.synchronize()
        print(f"OK (shape={frame.shape}, device={frame.device})")
    except Exception as e:
        print(f"FAIL: {e}")
        traceback.print_exc()
        return False

    print("  Decoding 10 more frames...", end=" ", flush=True)
    try:
        count = 1
        for _ in range(10):
            frame, *_ = torchcodec._core.get_next_frame(decoder)
            count += 1
        torch.cuda.synchronize()
        print(f"OK ({count} total frames)")
    except Exception as e:
        print(f"FAIL after {count} frames: {e}")
        traceback.print_exc()
        return False

    return True


def test_videodecoder_gpu(video_path):
    """Test 5: High-level VideoDecoder GPU decode."""
    section("5. VideoDecoder GPU Decode")

    import torch
    from torchcodec.decoders import VideoDecoder

    if not torch.cuda.is_available():
        print("  SKIP: No GPU")
        return False

    print("  Creating VideoDecoder on cuda:0...", end=" ", flush=True)
    try:
        decoder = VideoDecoder(
            str(video_path),
            device="cuda:0",
            num_ffmpeg_threads=1,
        )
        print("OK")
    except Exception as e:
        print(f"FAIL: {e}")
        traceback.print_exc()
        return False

    print("  Decoding first frame...", end=" ", flush=True)
    try:
        frame = next(iter(decoder))
        torch.cuda.synchronize()
        print(f"OK (shape={frame.data.shape}, device={frame.data.device})")
    except Exception as e:
        print(f"FAIL: {e}")
        traceback.print_exc()
        return False

    print("  Decoding 10 frames...", end=" ", flush=True)
    try:
        decoder2 = VideoDecoder(
            str(video_path),
            device="cuda:0",
            num_ffmpeg_threads=1,
        )
        count = 0
        for f in decoder2:
            count += 1
            if count >= 10:
                break
        torch.cuda.synchronize()
        print(f"OK ({count} frames)")
    except Exception as e:
        print(f"FAIL after {count} frames: {e}")
        traceback.print_exc()
        return False

    return True


def test_drm_info():
    """Bonus: DRM render node info for multi-GPU debugging."""
    section("DRM / GPU Topology Info")

    # List DRI render nodes
    dri_path = Path("/dev/dri")
    if dri_path.exists():
        nodes = sorted(dri_path.glob("renderD*"))
        print(f"  DRI render nodes: {len(nodes)}")
        for n in nodes:
            print(f"    {n}")
    else:
        print("  /dev/dri not found")

    # Check KFD
    kfd = Path("/dev/kfd")
    print(f"  /dev/kfd exists: {kfd.exists()}")

    # rocminfo GPU topology
    try:
        result = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, timeout=10
        )
        agents = []
        current_agent = {}
        for line in result.stdout.splitlines():
            if "Agent " in line and "Agent " in line:
                if current_agent:
                    agents.append(current_agent)
                current_agent = {"header": line.strip()}
            if "Name:" in line:
                current_agent["name"] = line.split(":")[-1].strip()
            if "Device Type:" in line:
                current_agent["type"] = line.split(":")[-1].strip()
        if current_agent:
            agents.append(current_agent)

        gpu_agents = [a for a in agents if a.get("type", "").strip() == "GPU"]
        print(f"\n  rocminfo: {len(gpu_agents)} GPU agents")
        for i, a in enumerate(gpu_agents):
            print(f"    GPU {i}: {a.get('name', 'unknown')}")
    except Exception as e:
        print(f"  rocminfo: {e}")

    # HIP_VISIBLE_DEVICES
    print(f"\n  HIP_VISIBLE_DEVICES = {os.environ.get('HIP_VISIBLE_DEVICES', '(not set)')}")
    print(f"  ROCR_VISIBLE_DEVICES = {os.environ.get('ROCR_VISIBLE_DEVICES', '(not set)')}")
    print(f"  CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', '(not set)')}")


def main():
    # Find test video
    test_resources = Path("/workspace/test_resources")
    repo_resources = Path(__file__).resolve().parent.parent.parent / "test" / "resources"

    if len(sys.argv) > 1:
        video_path = Path(sys.argv[1])
    elif (test_resources / "nasa_13013.mp4").exists():
        video_path = test_resources / "nasa_13013.mp4"
    elif (repo_resources / "nasa_13013.mp4").exists():
        video_path = repo_resources / "nasa_13013.mp4"
    else:
        print("ERROR: No test video found. Pass a video path as argument.")
        sys.exit(1)

    print(f"Using video: {video_path}")

    results = {}

    # Run diagnostics in order
    test_drm_info()
    results["hip_runtime"] = test_hip_runtime()
    results["rocdecode_lib"] = test_rocdecode_library()
    results["cpu_decode"] = test_cpu_decode(video_path)
    results["core_api_gpu"] = test_core_api_gpu(video_path)
    results["videodecoder_gpu"] = test_videodecoder_gpu(video_path)

    # Summary
    section("SUMMARY")
    for test, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test:<25} {status}")

    all_passed = all(results.values())
    print(f"\n  Overall: {'ALL PASSED' if all_passed else 'FAILURES DETECTED'}")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
