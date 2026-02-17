#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Cross-platform GPU benchmark for TorchCodec: ROCm vs CUDA

Runs identical workloads on both AMD (rocDecode/VCN) and NVIDIA (NVDEC)
hardware decoders, producing comparable JSON output for head-to-head
analysis.

Usage:
    # Auto-detect GPU and run all benchmarks
    python rocm_vs_cuda_benchmark.py

    # Specify output file
    python rocm_vs_cuda_benchmark.py --output results.json

    # Quick mode (fewer iterations, faster turnaround)
    python rocm_vs_cuda_benchmark.py --quick

    # Test specific codecs only
    python rocm_vs_cuda_benchmark.py --codecs h264,h265

    # Include multi-threaded decode benchmark
    python rocm_vs_cuda_benchmark.py --threads 4
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.utils.benchmark as benchmark

import torchcodec
from torchcodec.decoders import VideoDecoder


# ---------------------------------------------------------------------------
# Test video management
# ---------------------------------------------------------------------------
TEST_RESOURCES = Path(__file__).resolve().parent.parent.parent / "test" / "resources"
# Docker containers copy test resources to /workspace/test_resources
DOCKER_TEST_RESOURCES = Path("/workspace/test_resources")


def _find_test_resources() -> Path:
    """Find test resources directory (works both in-repo and in Docker)."""
    if TEST_RESOURCES.exists():
        return TEST_RESOURCES
    if DOCKER_TEST_RESOURCES.exists():
        return DOCKER_TEST_RESOURCES
    return TEST_RESOURCES  # fallback, will fail gracefully later


# Videos bundled with the repo, keyed by codec
_res = _find_test_resources()
BUNDLED_VIDEOS = {
    "h264": _res / "nasa_13013.mp4",
    "h265": _res / "h265_video.mp4",
    "av1": _res / "av1_video.mkv",
}

# Synthetic video specs for generation when bundled videos are insufficient
SYNTHETIC_SPECS = [
    {
        "name": "synthetic_1080p_h264_30s",
        "resolution": "1920x1080",
        "codec": "libx264",
        "duration": 30,
        "fps": 30,
        "gop": 30,
        "pix_fmt": "yuv420p",
    },
    {
        "name": "synthetic_1080p_h264_120s",
        "resolution": "1920x1080",
        "codec": "libx264",
        "duration": 120,
        "fps": 60,
        "gop": 60,
        "pix_fmt": "yuv420p",
    },
    {
        "name": "synthetic_4k_h264_10s",
        "resolution": "3840x2160",
        "codec": "libx264",
        "duration": 10,
        "fps": 30,
        "gop": 30,
        "pix_fmt": "yuv420p",
    },
]


def generate_synthetic_video(spec: dict, output_dir: Path) -> Path:
    """Generate a synthetic test video using ffmpeg."""
    outfile = output_dir / f"{spec['name']}.mp4"
    if outfile.exists():
        return outfile

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"mandelbrot=s={spec['resolution']}",
        "-t", str(spec["duration"]),
        "-c:v", spec["codec"],
        "-r", str(spec["fps"]),
        "-g", str(spec["gop"]),
        "-pix_fmt", spec["pix_fmt"],
        str(outfile),
    ]
    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"Warning: could not generate {outfile.name} (ffmpeg not available?)")
        return None
    return outfile


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------
@dataclass
class GPUInfo:
    vendor: str  # "AMD" or "NVIDIA"
    name: str
    architecture: str
    vram_mb: int
    driver_version: str
    runtime_version: str
    device_count: int


def detect_gpu() -> GPUInfo:
    """Detect GPU vendor and capabilities."""
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available. torch.cuda.is_available() returned False.")

    props = torch.cuda.get_device_properties(0)
    device_count = torch.cuda.device_count()

    # Check if this is ROCm (HIP)
    if hasattr(torch.version, "hip") and torch.version.hip is not None:
        vendor = "AMD"
        runtime_version = torch.version.hip

        # Try to get architecture from rocminfo
        arch = "unknown"
        try:
            out = subprocess.check_output(
                ["rocminfo"], stderr=subprocess.DEVNULL, text=True
            )
            for line in out.splitlines():
                if "gfx" in line.lower() and "name:" in line.lower():
                    arch = line.split(":")[-1].strip()
                    break
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        # Try to get driver version
        driver = "unknown"
        try:
            out = subprocess.check_output(
                ["rocm-smi", "--showdriverversion"],
                stderr=subprocess.DEVNULL, text=True
            )
            for line in out.splitlines():
                if "driver" in line.lower() and "version" in line.lower():
                    driver = line.split(":")[-1].strip()
                    break
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    else:
        vendor = "NVIDIA"
        runtime_version = torch.version.cuda
        arch = f"sm_{props.major}{props.minor}"

        driver = "unknown"
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                stderr=subprocess.DEVNULL, text=True
            )
            driver = out.strip().split("\n")[0]
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    return GPUInfo(
        vendor=vendor,
        name=props.name,
        architecture=arch,
        vram_mb=props.total_mem // (1024 * 1024),
        driver_version=driver,
        runtime_version=runtime_version,
        device_count=device_count,
    )


# ---------------------------------------------------------------------------
# Benchmark data structures
# ---------------------------------------------------------------------------
@dataclass
class BenchmarkResult:
    test_name: str
    codec: str
    video_file: str
    video_resolution: str
    video_duration_s: float
    video_fps: float
    num_frames_decoded: int
    device: str  # "gpu" or "cpu"
    gpu_vendor: str
    gpu_name: str
    decode_fps_median: float
    decode_fps_p25: float
    decode_fps_p75: float
    time_median_s: float
    time_iqr_s: float
    seek_mode: str = "exact"
    resize: str = "none"
    num_threads: int = 1
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Benchmark workloads
# ---------------------------------------------------------------------------
def get_video_info(video_path: str) -> dict:
    """Extract metadata from a video file."""
    decoder = VideoDecoder(video_path)
    meta = decoder.metadata
    return {
        "codec": meta.codec,
        "width": meta.width,
        "height": meta.height,
        "duration_s": meta.duration_seconds,
        "fps": meta.average_fps,
        "num_frames": meta.num_frames,
    }


def bench_sequential_decode(
    video_path: str,
    device: str,
    num_frames: int,
    seek_mode: str = "approximate",
    min_run_time: float = 10.0,
) -> benchmark.Measurement:
    """Benchmark sequential frame decoding from the start."""
    def _decode():
        decoder = VideoDecoder(
            video_path,
            device=device,
            seek_mode=seek_mode,
            num_ffmpeg_threads=1 if device != "cpu" else 0,
        )
        count = 0
        for frame in decoder:
            count += 1
            if count >= num_frames:
                break
        return count

    t = benchmark.Timer(
        stmt="_decode()",
        globals={"_decode": _decode},
        label="sequential_decode",
        sub_label=f"device={device}",
        description=f"first {num_frames} frames",
    )
    return t.blocked_autorange(min_run_time=min_run_time)


def bench_full_video_decode(
    video_path: str,
    device: str,
    seek_mode: str = "approximate",
    min_run_time: float = 10.0,
) -> tuple[benchmark.Measurement, int]:
    """Benchmark decoding an entire video."""
    frame_count_holder = [0]

    def _decode():
        decoder = VideoDecoder(
            video_path,
            device=device,
            seek_mode=seek_mode,
            num_ffmpeg_threads=1 if device != "cpu" else 0,
        )
        count = 0
        for frame in decoder:
            count += 1
        frame_count_holder[0] = count
        return count

    # Warm up once to get frame count
    _decode()
    frame_count = frame_count_holder[0]

    t = benchmark.Timer(
        stmt="_decode()",
        globals={"_decode": _decode},
        label="full_video_decode",
        sub_label=f"device={device}",
        description=f"all {frame_count} frames",
    )
    return t.blocked_autorange(min_run_time=min_run_time), frame_count


def bench_random_seek_decode(
    video_path: str,
    device: str,
    num_seeks: int = 50,
    min_run_time: float = 10.0,
) -> benchmark.Measurement:
    """Benchmark random seek + decode operations."""
    meta = VideoDecoder(video_path).metadata
    duration = meta.duration_seconds

    # Generate deterministic random seek points
    torch.manual_seed(42)
    pts_list = (torch.rand(num_seeks) * duration).tolist()

    def _seek_decode():
        decoder = VideoDecoder(
            video_path,
            device=device,
            seek_mode="exact",
            num_ffmpeg_threads=1 if device != "cpu" else 0,
        )
        return decoder.get_frames_played_at(pts_list)

    t = benchmark.Timer(
        stmt="_seek_decode()",
        globals={"_seek_decode": _seek_decode},
        label="random_seek_decode",
        sub_label=f"device={device}",
        description=f"{num_seeks} random seeks",
    )
    return t.blocked_autorange(min_run_time=min_run_time)


def bench_decode_and_resize(
    video_path: str,
    device: str,
    num_frames: int = 100,
    resize_h: int = 256,
    resize_w: int = 256,
    min_run_time: float = 10.0,
) -> benchmark.Measurement:
    """Benchmark decode + resize pipeline (common in training)."""
    import torchvision.transforms.v2.functional as F

    def _decode_resize():
        decoder = VideoDecoder(
            video_path,
            device=device,
            seek_mode="approximate",
            num_ffmpeg_threads=1 if device != "cpu" else 0,
        )
        count = 0
        for frame in decoder:
            resized = F.resize(frame.data, (resize_h, resize_w))
            count += 1
            if count >= num_frames:
                break
        return count

    t = benchmark.Timer(
        stmt="_decode_resize()",
        globals={"_decode_resize": _decode_resize},
        label="decode_and_resize",
        sub_label=f"device={device}",
        description=f"{num_frames} frames -> {resize_h}x{resize_w}",
    )
    return t.blocked_autorange(min_run_time=min_run_time)


def bench_multithreaded_decode(
    video_path: str,
    device: str,
    num_videos: int = 10,
    num_threads: int = 4,
    min_run_time: float = 10.0,
) -> tuple[benchmark.Measurement, int]:
    """Benchmark concurrent video decoding across threads."""
    frame_count_holder = [0]

    def _decode_one(dev):
        decoder = VideoDecoder(
            video_path,
            device=dev,
            seek_mode="approximate",
            num_ffmpeg_threads=1 if dev != "cpu" else 0,
        )
        count = 0
        for frame in decoder:
            count += 1
        return count

    def _decode_all():
        total = 0
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            device_count = torch.cuda.device_count()
            futures = []
            for i in range(num_videos):
                dev = f"cuda:{i % device_count}" if device != "cpu" else "cpu"
                futures.append(executor.submit(_decode_one, dev))
            for f in futures:
                total += f.result()
        frame_count_holder[0] = total
        return total

    # Warm up
    _decode_all()
    total_frames = frame_count_holder[0]

    t = benchmark.Timer(
        stmt="_decode_all()",
        globals={"_decode_all": _decode_all},
        label="multithreaded_decode",
        sub_label=f"device={device}",
        description=f"{num_videos} videos x {num_threads} threads",
    )
    return t.blocked_autorange(min_run_time=min_run_time), total_frames


def measurement_to_result(
    measurement: benchmark.Measurement,
    test_name: str,
    codec: str,
    video_path: str,
    video_info: dict,
    num_frames: int,
    device: str,
    gpu_info: GPUInfo,
    **extra_fields,
) -> BenchmarkResult:
    """Convert a torch.benchmark.Measurement into our result format."""
    fps_median = num_frames / measurement.median
    fps_p25 = num_frames / measurement._p75  # inverted: slower time = lower fps
    fps_p75 = num_frames / measurement._p25

    return BenchmarkResult(
        test_name=test_name,
        codec=codec,
        video_file=Path(video_path).name,
        video_resolution=f"{video_info['width']}x{video_info['height']}",
        video_duration_s=video_info["duration_s"],
        video_fps=video_info["fps"],
        num_frames_decoded=num_frames,
        device="gpu" if device != "cpu" else "cpu",
        gpu_vendor=gpu_info.vendor,
        gpu_name=gpu_info.name,
        decode_fps_median=round(fps_median, 2),
        decode_fps_p25=round(fps_p25, 2),
        decode_fps_p75=round(fps_p75, 2),
        time_median_s=round(measurement.median, 4),
        time_iqr_s=round(measurement.iqr, 4),
        **extra_fields,
    )


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------
def run_all_benchmarks(args) -> dict:
    gpu_info = detect_gpu()
    print(f"\n{'='*70}")
    print(f"TorchCodec GPU Benchmark")
    print(f"{'='*70}")
    print(f"GPU Vendor:      {gpu_info.vendor}")
    print(f"GPU Name:        {gpu_info.name}")
    print(f"Architecture:    {gpu_info.architecture}")
    print(f"VRAM:            {gpu_info.vram_mb} MB")
    print(f"Driver:          {gpu_info.driver_version}")
    print(f"Runtime:         {gpu_info.runtime_version}")
    print(f"Device Count:    {gpu_info.device_count}")
    print(f"PyTorch:         {torch.__version__}")
    print(f"TorchCodec:      {torchcodec.__version__}")
    print(f"Python:          {platform.python_version()}")
    print(f"{'='*70}\n")

    min_run_time = 5.0 if args.quick else 15.0
    codecs_to_test = args.codecs.split(",") if args.codecs else ["h264"]

    # Collect videos to benchmark
    videos = {}
    for codec in codecs_to_test:
        if codec in BUNDLED_VIDEOS and BUNDLED_VIDEOS[codec].exists():
            videos[codec] = str(BUNDLED_VIDEOS[codec])
        else:
            print(f"Warning: no test video for codec '{codec}', skipping")

    # Generate synthetic videos if requested
    if args.synthetic:
        synth_dir = Path(args.synthetic_dir)
        synth_dir.mkdir(parents=True, exist_ok=True)
        for spec in SYNTHETIC_SPECS:
            path = generate_synthetic_video(spec, synth_dir)
            if path:
                videos[f"synthetic_{spec['name']}"] = str(path)

    if not videos:
        print("ERROR: No test videos found. Ensure test/resources/ contains videos.")
        sys.exit(1)

    devices = ["cuda:0", "cpu"]
    all_results = []
    torch_results = []  # for Compare table

    for codec_label, video_path in videos.items():
        video_info = get_video_info(video_path)
        print(f"\n--- Video: {Path(video_path).name} ---")
        print(f"    Codec: {video_info['codec']}  Resolution: {video_info['width']}x{video_info['height']}")
        print(f"    Duration: {video_info['duration_s']:.1f}s  FPS: {video_info['fps']:.1f}  Frames: {video_info['num_frames']}")

        for device in devices:
            if device != "cpu" and not torch.cuda.is_available():
                continue

            device_label = f"{gpu_info.vendor} GPU" if device != "cpu" else "CPU"
            print(f"\n  [{device_label}] ({device})")

            # 1. Sequential decode (first N frames)
            num_seq = min(200, video_info["num_frames"])
            print(f"    Sequential decode ({num_seq} frames)...", end=" ", flush=True)
            m = bench_sequential_decode(video_path, device, num_seq, min_run_time=min_run_time)
            torch_results.append(m)
            r = measurement_to_result(
                m, "sequential_decode", codec_label, video_path,
                video_info, num_seq, device, gpu_info,
                seek_mode="approximate",
            )
            all_results.append(r)
            print(f"{r.decode_fps_median:.1f} FPS (median)")

            # 2. Full video decode
            print(f"    Full video decode...", end=" ", flush=True)
            m, frame_count = bench_full_video_decode(
                video_path, device, min_run_time=min_run_time
            )
            torch_results.append(m)
            r = measurement_to_result(
                m, "full_video_decode", codec_label, video_path,
                video_info, frame_count, device, gpu_info,
                seek_mode="approximate",
            )
            all_results.append(r)
            print(f"{r.decode_fps_median:.1f} FPS (median), {frame_count} frames")

            # 3. Random seek + decode
            num_seeks = 20 if args.quick else 50
            print(f"    Random seek+decode ({num_seeks} seeks)...", end=" ", flush=True)
            m = bench_random_seek_decode(video_path, device, num_seeks, min_run_time=min_run_time)
            torch_results.append(m)
            r = measurement_to_result(
                m, "random_seek_decode", codec_label, video_path,
                video_info, num_seeks, device, gpu_info,
                seek_mode="exact",
            )
            all_results.append(r)
            print(f"{r.decode_fps_median:.1f} FPS (median)")

            # 4. Decode + resize (training-like pipeline)
            num_resize = min(100, video_info["num_frames"])
            print(f"    Decode+resize ({num_resize} frames -> 256x256)...", end=" ", flush=True)
            try:
                m = bench_decode_and_resize(
                    video_path, device, num_resize, min_run_time=min_run_time
                )
                torch_results.append(m)
                r = measurement_to_result(
                    m, "decode_and_resize", codec_label, video_path,
                    video_info, num_resize, device, gpu_info,
                    resize="256x256",
                )
                all_results.append(r)
                print(f"{r.decode_fps_median:.1f} FPS (median)")
            except Exception as e:
                print(f"skipped ({e})")

            # 5. Multi-threaded decode (GPU only)
            if device != "cpu" and args.threads > 1:
                num_vids = args.threads * 2
                print(f"    Multi-threaded decode ({num_vids} videos, {args.threads} threads)...", end=" ", flush=True)
                m, total_frames = bench_multithreaded_decode(
                    video_path, device, num_vids, args.threads,
                    min_run_time=min_run_time,
                )
                torch_results.append(m)
                r = measurement_to_result(
                    m, "multithreaded_decode", codec_label, video_path,
                    video_info, total_frames, device, gpu_info,
                    num_threads=args.threads,
                )
                all_results.append(r)
                print(f"{r.decode_fps_median:.1f} FPS (median)")

    # Print comparison table
    if torch_results:
        print(f"\n{'='*70}")
        print("PyTorch Benchmark Comparison Table")
        print(f"{'='*70}")
        compare = benchmark.Compare(torch_results)
        compare.print()

    # Build output
    output = {
        "gpu_info": asdict(gpu_info),
        "system_info": {
            "cpu_count": os.cpu_count(),
            "platform": platform.system(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "torchcodec_version": torchcodec.__version__,
        },
        "benchmark_config": {
            "min_run_time_s": min_run_time,
            "quick_mode": args.quick,
            "codecs_tested": codecs_to_test,
        },
        "results": [asdict(r) for r in all_results],
    }

    return output


def print_summary(output: dict):
    """Print a concise summary comparing GPU vs CPU performance."""
    results = output["results"]
    gpu_info = output["gpu_info"]

    print(f"\n{'='*70}")
    print(f"SUMMARY: {gpu_info['vendor']} {gpu_info['name']}")
    print(f"{'='*70}")
    print(f"{'Test':<30} {'GPU FPS':>10} {'CPU FPS':>10} {'Speedup':>10}")
    print(f"{'-'*60}")

    # Group by (test_name, codec, video_file)
    from collections import defaultdict
    groups = defaultdict(dict)
    for r in results:
        key = (r["test_name"], r["codec"], r["video_file"])
        groups[key][r["device"]] = r["decode_fps_median"]

    for (test_name, codec, video_file), devs in sorted(groups.items()):
        gpu_fps = devs.get("gpu", 0)
        cpu_fps = devs.get("cpu", 0)
        speedup = gpu_fps / cpu_fps if cpu_fps > 0 else float("inf")
        label = f"{test_name} ({codec})"
        print(f"{label:<30} {gpu_fps:>10.1f} {cpu_fps:>10.1f} {speedup:>9.1f}x")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="TorchCodec GPU Benchmark: ROCm vs CUDA comparison"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSON file path. Default: benchmark_results_{vendor}.json",
    )
    parser.add_argument(
        "--codecs",
        type=str,
        default="h264",
        help="Comma-separated codecs to test: h264,h265,av1 (default: h264)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: fewer iterations, faster turnaround",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of threads for multi-threaded decode benchmark (default: 1, disabled)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Also generate and benchmark synthetic videos (1080p, 4K)",
    )
    parser.add_argument(
        "--synthetic-dir",
        type=str,
        default="/tmp/torchcodec_benchmark_videos",
        help="Directory for synthetic test videos",
    )

    args = parser.parse_args()
    output = run_all_benchmarks(args)

    # Print summary
    print_summary(output)

    # Write JSON
    vendor = output["gpu_info"]["vendor"].lower()
    gpu_name = output["gpu_info"]["name"].replace(" ", "_")
    output_path = args.output or f"benchmark_results_{vendor}_{gpu_name}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results written to: {output_path}")


if __name__ == "__main__":
    main()
