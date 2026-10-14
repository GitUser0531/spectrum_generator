"""EDM focused black and white music spectrum generator.

The desktop UI is intentionally kept in one file so it can be copied to a
machine with Python and run directly. NumPy is used for the FFT; soundfile can
decode FLAC/OGG, while FFmpeg handles MP3/M4A and video encoding.

Examples:
    python spectrum_generator.py
    python spectrum_generator.py --input track.mp3 --export-mp4 track_spectrum.mp4
"""

from __future__ import annotations

import argparse
import ctypes
import importlib
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised by the friendly startup error
    np = None  # type: ignore[assignment]


TARGET_SAMPLE_RATE = 44_100
DEFAULT_FFT_SIZE = 4096
DEFAULT_HOP_SIZE = 512
DEFAULT_BAND_COUNT = 96
SPECTRUM_SPLIT_HZ = 1_600.0
SPECTRUM_MAX_HZ = 14_000.0
DEFAULT_VIDEO_FPS = 60
DEFAULT_VIDEO_WIDTH = 1280
DEFAULT_VIDEO_HEIGHT = 360
_FFMPEG_BOOTSTRAP_ATTEMPTED = False
_FFMPEG_BOOTSTRAP_ERROR = ""
_CUDA_BOOTSTRAP_ATTEMPTED = False
_CUDA_BOOTSTRAP_ERROR = ""
_CUDA_BACKEND_ERROR = ""
_CUPY_WINDOW_KERNEL = None
_CUDA_DLL_DIRECTORY = None


class AudioLoadError(RuntimeError):
    """Raised when an audio file cannot be decoded."""


@dataclass
class AudioData:
    samples: "np.ndarray"
    sample_rate: int
    title: str
    source_path: Optional[Path] = None


@dataclass
class AnalysisResult:
    sample_rate: int
    duration: float
    frame_hop: int
    band_frequencies: "np.ndarray"
    bands: "np.ndarray"
    pulse: "np.ndarray"
    times: "np.ndarray"
    beat_frames: list[int]


def _require_numpy() -> None:
    if np is None:
        raise AudioLoadError(
            "缺少 NumPy。请先在终端运行: python -m pip install -r requirements.txt"
        )


def _pcm_to_float(raw: bytes, sample_width: int) -> "np.ndarray":
    """Decode common integer PCM widths, including packed 24-bit audio."""
    _require_numpy()
    if sample_width == 1:
        return (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if sample_width == 3:
        values = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        ints = (
            values[:, 0].astype(np.int32)
            | (values[:, 1].astype(np.int32) << 8)
            | (values[:, 2].astype(np.int32) << 16)
        )
        ints = np.where(ints & 0x800000, ints - 0x1000000, ints)
        return ints.astype(np.float32) / 8388608.0
    if sample_width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    raise AudioLoadError(f"暂不支持 {sample_width * 8} 位 WAV PCM")


def _load_wav(path: Path) -> AudioData:
    _require_numpy()
    try:
        with wave.open(str(path), "rb") as wav:
            channels = wav.getnchannels()
            rate = wav.getframerate()
            width = wav.getsampwidth()
            count = wav.getnframes()
            raw = wav.readframes(count)
    except (wave.Error, OSError) as exc:
        raise AudioLoadError(f"无法读取 WAV 文件: {exc}") from exc
    values = _pcm_to_float(raw, width)
    if channels > 1:
        values = values[: len(values) - len(values) % channels]
        values = values.reshape(-1, channels).mean(axis=1)
    return AudioData(_resample(values, rate, TARGET_SAMPLE_RATE), TARGET_SAMPLE_RATE, path.stem, path)


def _imageio_ffmpeg_path() -> Optional[str]:
    local_python_root = Path(__file__).resolve().parent / "runtime" / "packages"
    if local_python_root.exists() and str(local_python_root) not in sys.path:
        sys.path.insert(0, str(local_python_root))
    try:
        import imageio_ffmpeg

        executable = imageio_ffmpeg.get_ffmpeg_exe()
        if executable and Path(executable).exists():
            return executable
    except Exception:
        return None
    return None


def _local_ffmpeg_path() -> Optional[str]:
    """Look for an FFmpeg binary shipped beside the application first."""
    root = Path(__file__).resolve().parent
    candidates = [
        root / "runtime" / "ffmpeg" / "ffmpeg.exe",
        root / "runtime" / "ffmpeg" / "bin" / "ffmpeg.exe",
        root / "ffmpeg.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def _runtime_package_root() -> Path:
    local_root = Path(__file__).resolve().parent / "runtime" / "installed"
    try:
        local_root.mkdir(parents=True, exist_ok=True)
        return local_root
    except OSError:
        preferred = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "EDMSpectrumGenerator" / "packages"
        preferred.mkdir(parents=True, exist_ok=True)
        return preferred


def _install_python_packages(packages: list[str], index_url: Optional[str] = None) -> tuple[bool, str]:
    root = _runtime_package_root()
    root.mkdir(parents=True, exist_ok=True)
    base_command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-input",
        "--retries",
        "1",
        "--timeout",
        "10",
        "--upgrade",
        "--target",
        str(root),
    ]
    if index_url:
        base_command.extend(["--index-url", index_url])
    base_command.extend(packages)
    local_wheelhouse = Path(__file__).resolve().parent / "runtime" / "wheels"
    attempts = []
    if local_wheelhouse.exists():
        attempts.append(base_command[:-len(packages)] + ["--no-index", "--find-links", str(local_wheelhouse)] + packages)
    attempts.append(base_command)
    result = None
    error = ""
    for command in attempts:
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
        except Exception as exc:
            error = str(exc)
            continue
        if result.returncode == 0:
            break
        error = (result.stderr or result.stdout or "安装失败").strip()[-600:]
    if result is None:
        return False, error
    if result.returncode != 0:
        return False, error
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    importlib.invalidate_caches()
    return True, ""


def _nvidia_smi_path() -> Optional[str]:
    """Resolve nvidia-smi even when the GUI was launched without PATH setup."""
    candidates: list[Path] = []
    discovered = shutil.which("nvidia-smi")
    if discovered:
        candidates.append(Path(discovered))
    windows_root = Path(os.environ.get("WINDIR", r"C:\Windows"))
    program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    candidates.extend(
        [
            program_files / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe",
            windows_root / "System32" / "nvidia-smi.exe",
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def _nvidia_gpu_present() -> bool:
    if os.environ.get("SPECTRUM_USE_CUDA", "1") == "0":
        return False
    command = _nvidia_smi_path()
    if not command:
        # Some packaged/managed installs do not expose nvidia-smi. The CUDA
        # driver library still gives us a reliable signal to trigger local
        # CuPy completion in that case.
        try:
            loader = getattr(ctypes, "WinDLL", ctypes.CDLL)
            loader("nvcuda.dll" if os.name == "nt" else "libcuda.so.1")
            return True
        except (AttributeError, OSError):
            return False
    try:
        check = subprocess.run(
            [command, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return check.returncode == 0 and bool((check.stdout or "").strip())


def _cuda_wheelhouse_available() -> bool:
    """Return whether the bundled CuPy and NVIDIA CUDA wheels are available."""
    wheelhouse = Path(__file__).resolve().parent / "runtime" / "wheels"
    return (
        any(wheelhouse.glob("cupy_cuda12x-*.whl"))
        and any(wheelhouse.glob("nvidia_*_cu12-*.whl"))
    )


def _prepare_cuda_runtime() -> tuple[bool, str]:
    """Install a matching CUDA CuPy runtime locally before the GUI opens."""
    global _CUDA_BOOTSTRAP_ATTEMPTED, _CUDA_BOOTSTRAP_ERROR
    if os.environ.get("SPECTRUM_USE_CUDA", "1") == "0" or cuda_backend_available():
        return True, ""
    # The wheelhouse is shipped with the app, so a detected GPU can be
    # completed without waiting for a network download.
    # A packaged app may not expose nvidia-smi to the launcher. If the local
    # wheelhouse is present, try the offline completion anyway; a CPU-only
    # machine will simply keep the CPU backend after CuPy reports zero devices.
    if not _nvidia_gpu_present() and not _cuda_wheelhouse_available():
        return True, ""
    if _CUDA_BOOTSTRAP_ATTEMPTED:
        return False, _CUDA_BOOTSTRAP_ERROR
    _CUDA_BOOTSTRAP_ATTEMPTED = True
    cuda_package = os.environ.get("SPECTRUM_CUDA_PACKAGE", "cupy-cuda12x[ctk]==14.2.0")
    cuda_index = os.environ.get("SPECTRUM_CUDA_INDEX_URL", "") or None
    ok, details = _install_python_packages([cuda_package], index_url=cuda_index)
    if not ok:
        _CUDA_BOOTSTRAP_ERROR = details
        return False, details
    if not cuda_backend_available():
        _CUDA_BOOTSTRAP_ERROR = _CUDA_BACKEND_ERROR or "CUDA CuPy 已安装，但设备 FFT 初始化仍未成功。"
        return False, _CUDA_BOOTSTRAP_ERROR
    return True, ""


def _prepare_python_runtime() -> tuple[bool, str]:
    """Ensure the modules needed before the GUI opens are importable."""
    local_root = Path(__file__).resolve().parent / "runtime" / "packages"
    if local_root.exists() and str(local_root) not in sys.path:
        sys.path.insert(0, str(local_root))
    root = _runtime_package_root()
    if root.exists() and str(root) not in sys.path:
        sys.path.insert(0, str(root))
    required = [("numpy", "numpy>=1.26"), ("PIL", "Pillow>=10.0"), ("tkinterdnd2", "tkinterdnd2>=0.4.2"), ("imageio_ffmpeg", "imageio-ffmpeg>=0.5.1")]
    missing = []
    for module, package in required:
        try:
            importlib.import_module(module)
        except Exception:
            missing.append(package)
    if missing:
        ok, details = _install_python_packages(missing)
        if not ok:
            return False, "Python 运行依赖安装失败：" + details
    global np
    if np is None:
        try:
            np = importlib.import_module("numpy")
        except Exception as exc:
            return False, f"NumPy 加载失败：{exc}"
    cuda_requested = _nvidia_gpu_present()
    cuda_ok, cuda_details = _prepare_cuda_runtime()
    if not cuda_ok:
        # A machine without an NVIDIA device may continue in CPU mode, but a
        # real installation/import failure must stop startup with the actual
        # reason instead of opening a GUI that can only fail later.
        no_device = "设备数量为 0" in (cuda_details or "")
        if cuda_requested or (_cuda_wheelhouse_available() and not no_device):
            return False, "CUDA 运行依赖补全失败：" + (cuda_details or "未知错误")
    return True, ""


def _auto_install_imageio_ffmpeg() -> Optional[str]:
    """Install a self-contained FFmpeg provider into the user's app cache."""
    global _FFMPEG_BOOTSTRAP_ATTEMPTED, _FFMPEG_BOOTSTRAP_ERROR
    if _FFMPEG_BOOTSTRAP_ATTEMPTED or os.environ.get("SPECTRUM_AUTO_FFMPEG", "1") == "0":
        return None
    _FFMPEG_BOOTSTRAP_ATTEMPTED = True
    runtime_root = _runtime_package_root()
    runtime_root.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-input",
        "--retries",
        "1",
        "--timeout",
        "10",
        "--target",
        str(runtime_root),
        "imageio-ffmpeg>=0.5.1",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=25, check=False)
    except Exception as exc:
        _FFMPEG_BOOTSTRAP_ERROR = str(exc)
        return None
    if result.returncode != 0:
        _FFMPEG_BOOTSTRAP_ERROR = (result.stderr or result.stdout or "自动安装失败").strip()[-500:]
        return None
    sys.path.insert(0, str(runtime_root))
    importlib.invalidate_caches()
    return _imageio_ffmpeg_path()


def _ffmpeg_path(auto_install: bool = True) -> Optional[str]:
    configured = os.environ.get("FFMPEG_PATH")
    if configured and Path(configured).exists():
        return configured
    local_ffmpeg = _local_ffmpeg_path()
    if local_ffmpeg:
        return local_ffmpeg
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    bundled_ffmpeg = _imageio_ffmpeg_path()
    if bundled_ffmpeg:
        return bundled_ffmpeg
    return _auto_install_imageio_ffmpeg() if auto_install else None


def _ffmpeg_install_message() -> str:
    if _FFMPEG_BOOTSTRAP_ERROR:
        return f"程序已尝试自动准备 FFmpeg，但失败原因是：{_FFMPEG_BOOTSTRAP_ERROR}"
    return "程序已尝试自动准备 FFmpeg，但当前网络或权限不允许安装"


def _load_with_soundfile(path: Path) -> Optional[AudioData]:
    """Decode formats supported by libsndfile without requiring FFmpeg."""
    _require_numpy()
    try:
        import soundfile as sf
    except Exception:
        return None
    try:
        values, rate = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as exc:
        raise AudioLoadError(f"soundfile 无法读取 {path.suffix.lower()} 文件: {exc}") from exc
    if values.size == 0:
        raise AudioLoadError("音频没有可分析的采样数据")
    mono = values.mean(axis=1)
    return AudioData(_resample(mono, int(rate), TARGET_SAMPLE_RATE), TARGET_SAMPLE_RATE, path.stem, path)


def _load_with_ffmpeg(path: Path, ffmpeg: Optional[str] = None) -> AudioData:
    _require_numpy()
    ffmpeg = ffmpeg or _ffmpeg_path()
    if not ffmpeg:
        raise AudioLoadError(
            "读取此格式需要 FFmpeg。" + _ffmpeg_install_message() +
            "也可以设置 FFMPEG_PATH 指向已有的 ffmpeg.exe。"
        )
    command = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "pipe:1",
    ]
    try:
        result = subprocess.run(command, capture_output=True, check=False)
    except OSError as exc:
        raise AudioLoadError(f"启动 FFmpeg 失败: {exc}") from exc
    if result.returncode != 0 or not result.stdout:
        details = result.stderr.decode("utf-8", errors="replace").strip()
        raise AudioLoadError(f"FFmpeg 无法解码音频。{details[-400:]}")
    samples = np.frombuffer(result.stdout, dtype="<f4").astype(np.float32)
    samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
    return AudioData(np.clip(samples, -1.0, 1.0), TARGET_SAMPLE_RATE, path.stem, path)


def load_audio(path: str | Path) -> AudioData:
    path = Path(path)
    if not path.exists():
        raise AudioLoadError(f"找不到文件: {path}")
    suffix = path.suffix.lower()
    # Use one FFmpeg pipeline whenever it is already available. WAV keeps a
    # standard-library fallback so it still works on a fresh offline install.
    existing_ffmpeg = _ffmpeg_path(auto_install=False)
    if existing_ffmpeg:
        try:
            return _load_with_ffmpeg(path, existing_ffmpeg)
        except AudioLoadError:
            pass
    if suffix == ".wav":
        try:
            return _load_wav(path)
        except AudioLoadError:
            pass
    # FLAC/OGG can still be decoded offline when soundfile is installed.
    try:
        decoded = _load_with_soundfile(path)
    except AudioLoadError:
        decoded = None
    if decoded is not None:
        return decoded
    if _ffmpeg_path():
        return _load_with_ffmpeg(path)
    suffix = suffix or "音频"
    if suffix in {".flac", ".ogg", ".oga"}:
        raise AudioLoadError(
            f"读取 {suffix} 需要 soundfile 或 FFmpeg。请运行 `python -m pip install soundfile`，"
            "或让程序自动准备 FFmpeg。" + _ffmpeg_install_message()
        )
    raise AudioLoadError(
        "读取此格式需要 FFmpeg。" + _ffmpeg_install_message() +
        "也可以设置 FFMPEG_PATH 指向已有的 ffmpeg.exe。"
    )


def _resample(samples: "np.ndarray", source_rate: int, target_rate: int) -> "np.ndarray":
    if source_rate == target_rate or len(samples) == 0:
        return samples.astype(np.float32, copy=False)
    target_len = max(1, round(len(samples) * target_rate / source_rate))
    old_x = np.linspace(0.0, 1.0, len(samples), endpoint=False)
    new_x = np.linspace(0.0, 1.0, target_len, endpoint=False)
    return np.interp(new_x, old_x, samples).astype(np.float32)


def _rolling_median(values: "np.ndarray", radius: int) -> "np.ndarray":
    """Small dependency-free rolling median for adaptive beat thresholds."""
    if len(values) < 3 or radius <= 0:
        return values.copy()
    padded = np.pad(values, (radius, radius), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, radius * 2 + 1)
    return np.median(windows, axis=1)


def _normalise(values: "np.ndarray", low: float = 5.0, high: float = 98.0) -> "np.ndarray":
    if len(values) == 0:
        return values.astype(np.float32)
    lo, hi = np.percentile(values, [low, high])
    if hi - lo < 1e-9:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _detect_beats_from_features(low_energy: "np.ndarray", spectral_flux: "np.ndarray", hop: int, rate: int) -> tuple["np.ndarray", list[int]]:
    """Detect kick/snare transients from compact per-frame features.

    Only two one-dimensional arrays are retained for beat detection. This keeps
    hour-long DJ sets practical instead of retaining every FFT bin in memory.
    """
    if len(spectral_flux) < 2:
        return np.zeros(len(spectral_flux), dtype=np.float32), []
    low = np.log1p(np.maximum(low_energy, 0.0) * 30_000.0)
    flux = np.maximum(0.0, spectral_flux) + np.maximum(0.0, np.diff(low, prepend=low[0])) * 0.8
    flux = _normalise(flux, 15, 99.5)
    baseline = _rolling_median(flux, max(2, round(0.35 * rate / hop)))
    novelty = np.maximum(0.0, flux - baseline)
    beat_threshold = max(0.10, float(np.percentile(novelty, 73)) * 0.72)
    frame_gap = max(1, round(0.105 * rate / hop))
    candidates = np.flatnonzero(
        (novelty >= beat_threshold)
        & (novelty >= np.r_[novelty[0], novelty[:-1]])
        & (novelty >= np.r_[novelty[1:], novelty[-1]])
    )
    beats: list[int] = []
    for index in candidates.tolist():
        if not beats or index - beats[-1] >= frame_gap:
            beats.append(index)
        elif novelty[index] > novelty[beats[-1]]:
            beats[-1] = index
    pulse = np.zeros(len(spectral_flux), dtype=np.float32)
    for index in beats:
        pulse[index] = min(1.0, 0.42 + novelty[index] * 2.9)
    # Fast attack and musical decay. This makes each kick visibly enlarge the whole strip.
    decay = math.exp(-1.0 / max(1.0, 0.23 * rate / hop))
    for index in range(1, len(pulse)):
        pulse[index] = max(pulse[index], pulse[index - 1] * decay)
    return pulse, beats


def _cuda_torch():
    """Return a CUDA-enabled torch module when the optional GPU backend exists."""
    if os.environ.get("SPECTRUM_USE_CUDA", "1") == "0":
        return None
    try:
        import torch

        return torch if torch.cuda.is_available() else None
    except Exception:
        return None


def _cuda_cupy():
    """Return a CUDA-enabled CuPy module when the local CUDA runtime exists."""
    global _CUDA_BACKEND_ERROR
    if os.environ.get("SPECTRUM_USE_CUDA", "1") == "0":
        return None
    try:
        _prepare_cupy_environment()
        cache_dir = Path(__file__).resolve().parent / "runtime" / "cache" / "cupy"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("CUPY_CACHE_DIR", str(cache_dir))
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() > 0:
            _CUDA_BACKEND_ERROR = ""
            return cp
        _CUDA_BACKEND_ERROR = "CuPy 已加载，但 CUDA 设备数量为 0。"
    except Exception as exc:
        _CUDA_BACKEND_ERROR = f"CuPy 初始化失败：{type(exc).__name__}: {exc}"
        return None
    return None


def _prepare_cupy_environment() -> None:
    """Make the pip-installed CUDA DLLs visible as one local toolkit."""
    package_root = _runtime_package_root()
    cuda_root = package_root / "nvidia" / "cuda_runtime"
    if not cuda_root.exists():
        return
    include_root = cuda_root / "include"
    include_root.mkdir(parents=True, exist_ok=True)
    nvrtc_header = package_root / "nvidia" / "cuda_nvrtc" / "include" / "nvrtc.h"
    if nvrtc_header.exists() and not (include_root / "nvrtc.h").exists():
        shutil.copy2(nvrtc_header, include_root / "nvrtc.h")
    target_bin = cuda_root / "bin"
    target_bin.mkdir(parents=True, exist_ok=True)
    for source_bin in (package_root / "nvidia").glob("*/bin"):
        for dll in source_bin.glob("*.dll"):
            target = target_bin / dll.name
            if not target.exists():
                try:
                    shutil.copy2(dll, target)
                except OSError:
                    pass
    global _CUDA_DLL_DIRECTORY
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        try:
            if _CUDA_DLL_DIRECTORY is None:
                _CUDA_DLL_DIRECTORY = os.add_dll_directory(str(target_bin))
        except OSError:
            pass
    os.environ.setdefault("CUDA_PATH", str(cuda_root))


def _cupy_windowed_frames(cupy, frames, window):
    """Apply a Hann window without invoking CuPy's header-dependent ufuncs."""
    global _CUPY_WINDOW_KERNEL
    if _CUPY_WINDOW_KERNEL is None:
        _CUPY_WINDOW_KERNEL = cupy.RawKernel(
            r'''
            extern "C" __global__ void apply_window(
                const float* frames, const float* window, float* output,
                int frame_count, int frame_size) {
                int index = blockDim.x * blockIdx.x + threadIdx.x;
                int total = frame_count * frame_size;
                if (index < total) {
                    output[index] = frames[index] * window[index % frame_size];
                }
            }
            ''',
            "apply_window",
        )
    contiguous = cupy.ascontiguousarray(frames)
    output = cupy.empty_like(contiguous)
    total = int(contiguous.size)
    threads = 256
    blocks = (total + threads - 1) // threads
    _CUPY_WINDOW_KERNEL(
        (blocks,),
        (threads,),
        (contiguous, window, output, np.int32(contiguous.shape[0]), np.int32(contiguous.shape[1])),
    )
    return output


def _cupy_fft_ready(cupy) -> bool:
    global _CUDA_BACKEND_ERROR
    try:
        # Exercise the same custom window kernel and batched cuFFT path used
        # by analyse_audio, without CuPy elementwise kernels that require
        # optional header search paths.
        probe = cupy.empty((1, 64), dtype=cupy.float32)
        window = cupy.empty((64,), dtype=cupy.float32)
        windowed = _cupy_windowed_frames(cupy, probe, window)
        cupy.fft.rfft(windowed, axis=1)
        cupy.cuda.runtime.deviceSynchronize()
        _CUDA_BACKEND_ERROR = ""
        return True
    except Exception as exc:
        _CUDA_BACKEND_ERROR = f"CuPy FFT 初始化失败：{type(exc).__name__}: {exc}"
        return False


def _cuda_backend():
    torch = _cuda_torch()
    if torch is not None:
        return "torch", torch
    cupy = _cuda_cupy()
    if cupy is not None and _cupy_fft_ready(cupy):
        return "cupy", cupy
    return "cpu", None


def _normalise_backend_preference(preference: str | None) -> str:
    value = str(preference or "auto").strip().lower()
    if value.startswith("cuda"):
        return "cuda"
    if value.startswith("cpu"):
        return "cpu"
    return "auto"


def cuda_backend_available() -> bool:
    return _cuda_backend()[0] != "cpu"


def default_backend_choice() -> str:
    """Return the UI label that matches the backend available at startup."""
    return "CUDA / GPU" if cuda_backend_available() else "CPU / NumPy"


def _resolve_analysis_backend(preference: str | None):
    selected = _normalise_backend_preference(preference)
    if selected == "cpu":
        return "cpu", None
    backend, module = _cuda_backend()
    if selected == "cuda" and backend == "cpu":
        detail = _CUDA_BOOTSTRAP_ERROR or _CUDA_BACKEND_ERROR or "启动预检未能初始化 CUDA 设备"
        raise AudioLoadError(
            "已选择 CUDA，但当前 Python 环境没有可用的 CUDA 后端。"
            f"{detail}，或切换为 CPU / NumPy。"
        )
    if backend != "cpu":
        return backend, module
    return "cpu", None


def analysis_backend(preference: str = "auto") -> str:
    selected = _normalise_backend_preference(preference)
    if selected == "cpu":
        return "CPU / NumPy"
    if selected == "cuda":
        return "CUDA / GPU" if cuda_backend_available() else "CUDA / GPU（不可用）"
    return "CUDA / GPU" if cuda_backend_available() else "CPU / NumPy"


def analyse_audio(
    audio: AudioData,
    fft_size: int = DEFAULT_FFT_SIZE,
    hop_size: int = DEFAULT_HOP_SIZE,
    band_count: int = DEFAULT_BAND_COUNT,
    progress: Optional[Callable[[float], None]] = None,
    backend: str = "auto",
) -> AnalysisResult:
    """Create log-spaced spectrum bands and an EDM-friendly transient envelope.

    FFTs are processed in small chunks. When a CUDA-enabled backend is
    available, each chunk is transformed on the GPU; only the compact band
    matrix and beat features are kept after a chunk is complete.
    """
    _require_numpy()
    samples = audio.samples.astype(np.float32, copy=False)
    if len(samples) == 0:
        raise AudioLoadError("音频没有可分析的采样数据")
    peak = float(np.max(np.abs(samples)))
    if peak > 1e-6:
        samples = samples / max(1.0, peak)
    if len(samples) < fft_size:
        samples = np.pad(samples, (0, fft_size - len(samples)))
    frame_count = 1 + max(0, (len(samples) - fft_size) // hop_size)
    window = np.hanning(fft_size).astype(np.float32)
    frequencies = np.fft.rfftfreq(fft_size, 1.0 / audio.sample_rate)
    max_frequency = min(audio.sample_rate / 2.0, SPECTRUM_MAX_HZ)
    split_frequency = min(max_frequency * 0.45, SPECTRUM_SPLIT_HZ)
    low_band_count = max(24, round(band_count * 0.40))
    high_band_count = band_count - low_band_count
    low_edges = np.geomspace(35.0, split_frequency, low_band_count + 1)
    high_edges = np.geomspace(split_frequency, max_frequency, high_band_count + 1)[1:]
    edges = np.concatenate((low_edges, high_edges))
    centers = np.sqrt(edges[:-1] * edges[1:])
    band_masks = [(frequencies >= edges[i]) & (frequencies < edges[i + 1]) for i in range(band_count)]
    onset_edges = np.geomspace(35.0, min(audio.sample_rate / 2.0, 6500.0), 28)
    onset_masks = [(frequencies >= onset_edges[i]) & (frequencies < onset_edges[i + 1]) for i in range(len(onset_edges) - 1)]
    low_mask = (frequencies >= 35.0) & (frequencies <= 260.0)
    bands = np.zeros((frame_count, band_count), dtype=np.float32)
    low_feature = np.zeros(frame_count, dtype=np.float32)
    flux_feature = np.zeros(frame_count, dtype=np.float32)
    previous_onset_profile = None
    chunk_frames = 2048
    padded_len = (frame_count - 1) * hop_size + fft_size
    if len(samples) < padded_len:
        samples = np.pad(samples, (0, padded_len - len(samples)))
    selected_backend, cuda_module = _resolve_analysis_backend(backend)
    device = cuda_module.device("cuda") if selected_backend == "torch" else None
    torch_window = cuda_module.from_numpy(window).to(device) if selected_backend == "torch" else None
    torch_samples = cuda_module.from_numpy(samples) if selected_backend == "torch" else None
    cupy_window = cuda_module.asarray(window) if selected_backend == "cupy" else None
    for chunk_start in range(0, frame_count, chunk_frames):
        chunk_count = min(chunk_frames, frame_count - chunk_start)
        sample_start = chunk_start * hop_size
        sample_end = sample_start + (chunk_count - 1) * hop_size + fft_size
        if selected_backend == "torch":
            chunk_audio = torch_samples[sample_start:sample_end].to(device, non_blocking=True)
            frames = chunk_audio.unfold(0, fft_size, hop_size)[:chunk_count]
            chunk_spectrum = cuda_module.abs(cuda_module.fft.rfft(frames * torch_window, dim=1)) / (fft_size / 2.0)
            spectrum = chunk_spectrum.detach().cpu().numpy().astype(np.float32, copy=False)
            del chunk_audio, frames, chunk_spectrum
        elif selected_backend == "cupy":
            starts = sample_start + np.arange(chunk_count) * hop_size
            offsets = np.arange(fft_size)
            frames = cuda_module.asarray(samples[starts[:, None] + offsets[None, :]])
            windowed = _cupy_windowed_frames(cuda_module, frames, cupy_window)
            chunk_spectrum = cuda_module.fft.rfft(windowed, axis=1)
            spectrum = np.abs(cuda_module.asnumpy(chunk_spectrum)).astype(np.float32, copy=False) / (fft_size / 2.0)
            del frames, windowed, chunk_spectrum
        else:
            starts = sample_start + np.arange(chunk_count) * hop_size
            offsets = np.arange(fft_size)
            frames = samples[starts[:, None] + offsets[None, :]]
            spectrum = np.abs(np.fft.rfft(frames * window[None, :], axis=1)) / (fft_size / 2.0)
        energy = spectrum * spectrum
        target = slice(chunk_start, chunk_start + chunk_count)
        for band, mask in enumerate(band_masks):
            if mask.any():
                bands[target, band] = np.sqrt(np.mean(energy[:, mask], axis=1))
        if low_mask.any():
            low_feature[target] = np.mean(energy[:, low_mask], axis=1)
        onset_profile = np.zeros((chunk_count, len(onset_masks)), dtype=np.float32)
        for band, mask in enumerate(onset_masks):
            if mask.any():
                onset_profile[:, band] = np.log1p(np.mean(energy[:, mask], axis=1) * 50_000.0)
        if previous_onset_profile is not None:
            flux_feature[chunk_start] = np.maximum(0.0, onset_profile[0] - previous_onset_profile).mean()
        if chunk_count > 1:
            flux_feature[chunk_start + 1 : chunk_start + chunk_count] = np.maximum(0.0, np.diff(onset_profile, axis=0)).mean(axis=1)
        previous_onset_profile = onset_profile[-1]
        if progress:
            progress(min(1.0, (chunk_start + chunk_count) / frame_count))
    # Boost the first octave and use log compression so kicks stay readable in dense EDM mixes.
    bass_weight = 1.0 + 2.6 * np.exp(-centers / 300.0)
    bands *= bass_weight[None, :]
    db = 20.0 * np.log10(np.maximum(bands, 1e-7))
    floor = np.percentile(db, 8, axis=0)
    # Leave headroom so the beat envelope, rather than the static spectrum, owns the peaks.
    ceiling = np.percentile(db, 99.7, axis=0)
    bands = np.clip((db - floor[None, :]) / np.maximum(ceiling - floor, 1.0), 0.0, 1.0)
    bands = np.power(bands, 1.12).astype(np.float32) * 0.78
    # Smooth neighbouring bands so the spectrum reads as a connected contour.
    if band_count > 2:
        padded = np.pad(bands, ((0, 0), (2, 2)), mode="edge")
        bands = (
            padded[:, :-4] * 0.08
            + padded[:, 1:-3] * 0.18
            + padded[:, 2:-2] * 0.48
            + padded[:, 3:-1] * 0.18
            + padded[:, 4:] * 0.08
        )
    # Attack quickly enough for a kick, release more slowly for a fluid tail.
    if len(bands) > 1:
        smoothed = np.empty_like(bands)
        smoothed[0] = bands[0]
        for index in range(1, len(bands)):
            delta = bands[index] - smoothed[index - 1]
            response = np.where(delta >= 0.0, 0.20, 0.09)
            smoothed[index] = smoothed[index - 1] + delta * response
        bands = smoothed
    pulse, beat_frames = _detect_beats_from_features(low_feature, flux_feature, hop_size, audio.sample_rate)
    return AnalysisResult(
        sample_rate=audio.sample_rate,
        duration=len(audio.samples) / audio.sample_rate,
        frame_hop=hop_size,
        band_frequencies=centers.astype(np.float32),
        bands=bands,
        pulse=pulse,
        times=(np.arange(frame_count) * hop_size / audio.sample_rate).astype(np.float32),
        beat_frames=beat_frames,
    )


def _instant_frame_values(result: AnalysisResult, frame: int, low_focus: float = 0.0) -> "np.ndarray":
    frame = max(0, min(frame, len(result.bands) - 1))
    values = result.bands[frame].copy()
    # One control owns both the static bass focus and the beat response. Bass
    # gets the transient lift while the rest of the spectrum makes room for it.
    focus_value = np.clip(low_focus, 0.0, 4.0)
    focus_level = focus_value / 4.0
    frequencies = result.band_frequencies
    # Never boost sub-bass below 65 Hz. It fades down as focus increases, while
    # the useful kick and low-mid range fades in smoothly above the cutoff.
    focus_gate = 1.0 - np.exp(-np.maximum(frequencies - 65.0, 0.0) / 55.0)
    bass_profile = focus_gate * np.exp(-frequencies / 1050.0)
    bass_profile /= max(float(np.max(bass_profile)), 1e-6)
    kick_gate = 1.0 - np.exp(-np.maximum(frequencies - 80.0, 0.0) / 80.0)
    kick_profile = kick_gate * np.exp(-frequencies / 1650.0)
    kick_profile /= max(float(np.max(kick_profile)), 1e-6)
    sub_profile = np.exp(-frequencies / 78.0)
    pulse = np.clip(result.pulse[frame], 0.0, 1.0)
    values *= 0.68
    static_bass_boost = 1.0 + focus_level * (1.28 * bass_profile + 0.48 * kick_profile)
    static_high_attenuation = 1.0 - focus_level * 0.58 * focus_gate * (1.0 - bass_profile)
    beat_focus = pulse * (0.12 + 1.18 * focus_level)
    beat_bass_boost = 1.0 + beat_focus * (0.88 * bass_profile + 0.62 * kick_profile)
    beat_high_attenuation = 1.0 - pulse * focus_level * 0.42 * focus_gate * (1.0 - bass_profile)
    sub_attenuation = 1.0 - focus_level * (0.40 + pulse * 0.16) * sub_profile
    values *= static_bass_boost * static_high_attenuation * beat_bass_boost * beat_high_attenuation * sub_attenuation
    # Preserve a small border above and below the strip instead of hard clipping.
    # Keep the limiter from adding a second whole-band change to the sub-bass;
    # its intentional relative reduction is applied by sub_attenuation above.
    limiter_mask = frequencies >= 65.0 if focus_value > 0.0 else np.ones(len(values), dtype=bool)
    peak = float(np.max(values[limiter_mask])) if len(values) else 0.0
    if peak > 0.94:
        values[limiter_mask] *= 0.94 / peak
    return np.clip(values, 0.0, 0.94)


def frame_values(
    result: AnalysisResult,
    frame: int,
    low_focus: float = 0.0,
    delay_strength: float = 0.0,
) -> "np.ndarray":
    """Return the final bar heights with optional post-effect temporal lag."""
    frame = max(0, min(frame, len(result.bands) - 1))
    delay = float(np.clip(delay_strength, 0.0, 1.0))
    if delay <= 0.0 or frame <= 0:
        return _instant_frame_values(result, frame, low_focus)
    # Blend already-composed frames so bass focus and peak protection do not
    # erase the visible difference between an immediate hit and its tail.
    span = min(frame + 1, 2 + round(delay * 12.0))
    start = frame - span + 1
    history = np.stack([_instant_frame_values(result, index, low_focus) for index in range(start, frame + 1)])
    distances = np.arange(span - 1, -1, -1, dtype=np.float32)
    weights = np.exp(-distances / (0.8 + delay * 3.6))
    return np.clip(np.average(history, axis=0, weights=weights), 0.0, 0.94).astype(np.float32)


def _delay_alpha(delay_strength: float) -> float:
    delay = float(np.clip(delay_strength, 0.0, 1.0))
    return 1.0 / (1.0 + delay * 3.6)


def _draw_pillow_frame(
    result: AnalysisResult,
    frame: int,
    width: int,
    height: int,
    low_focus: float,
    delay_strength: float = 0.0,
    values: Optional["np.ndarray"] = None,
):
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("导出图片需要 Pillow，请运行: python -m pip install -r requirements.txt") from exc
    image = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    if values is None:
        values = frame_values(result, frame, low_focus, delay_strength)
    center_y = height // 2
    gap = max(1, round(width / len(values) * 0.18))
    bar_width = max(1, math.floor(width / len(values)) - gap)
    for index, value in enumerate(values):
        x = round(index * width / len(values) + gap / 2)
        half = max(1, round(value * (height * 0.46)))
        radius = min(bar_width // 2, half)
        draw.rounded_rectangle((x, center_y - half, x + bar_width, center_y + half), radius=radius, fill=(255, 255, 255))
    return image


def _video_encoder(ffmpeg: str) -> str:
    requested = os.environ.get("SPECTRUM_VIDEO_ENCODER", "").strip()
    if requested:
        return requested
    if os.environ.get("SPECTRUM_USE_NVENC", "0") == "1":
        try:
            check = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True, check=False)
            if "h264_nvenc" in check.stdout:
                return "h264_nvenc"
        except OSError:
            pass
    return "libx264"


def export_mp4(
    result: AnalysisResult,
    audio: AudioData,
    output: str | Path,
    fps: int = DEFAULT_VIDEO_FPS,
    width: int = DEFAULT_VIDEO_WIDTH,
    height: int = DEFAULT_VIDEO_HEIGHT,
    low_focus: float = 0.0,
    progress: Optional[Callable[[float], None]] = None,
    delay_strength: float = 0.0,
) -> None:
    """Stream a compact H.264 MP4 directly to FFmpeg without storing video frames."""
    _require_numpy()
    ffmpeg = _ffmpeg_path()
    if not ffmpeg:
        raise RuntimeError("生成 MP4 需要 FFmpeg。" + _ffmpeg_install_message())
    if not audio.source_path:
        raise RuntimeError("当前音频没有可用于合成视频的源文件路径")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoder = _video_encoder(ffmpeg)
    command = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-i",
        str(audio.source_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        encoder,
        "-preset",
        "p4" if encoder.endswith("_nvenc") else "veryfast",
        "-crf",
        "28",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "96k",
        "-shortest",
        "-movflags",
        "+faststart",
        str(output),
    ]
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except OSError as exc:
        raise RuntimeError(f"启动 FFmpeg 失败: {exc}") from exc
    total_frames = max(1, math.ceil(result.duration * fps))
    delayed_values = None
    delay_alpha = _delay_alpha(delay_strength) if delay_strength > 0.0 else 1.0
    try:
        for index in range(total_frames):
            source_frame = min(len(result.bands) - 1, round(index * result.sample_rate / fps / result.frame_hop))
            if delay_strength > 0.0:
                instant = _instant_frame_values(result, source_frame, low_focus)
                if delayed_values is None:
                    delayed_values = instant
                else:
                    delayed_values += (instant - delayed_values) * delay_alpha
                image = _draw_pillow_frame(result, source_frame, width, height, low_focus, values=delayed_values)
            else:
                image = _draw_pillow_frame(result, source_frame, width, height, low_focus)
            process.stdin.write(image.tobytes())
            if progress and (index % max(1, fps) == 0 or index == total_frames - 1):
                progress((index + 1) / total_frames)
        process.stdin.close()
    except (BrokenPipeError, OSError) as exc:
        process.kill()
        details = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        raise RuntimeError(f"FFmpeg 写入视频失败: {details[-500:]}") from exc
    return_code = process.wait()
    details = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    if return_code != 0:
        raise RuntimeError(f"FFmpeg 导出 MP4 失败: {details[-500:]}")


def export_gif(result: AnalysisResult, output: str | Path, fps: int = 24, width: int = 1280, height: int = 480, low_focus: float = 0.0, progress: Optional[Callable[[float], None]] = None, delay_strength: float = 0.0) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("导出 GIF 需要 Pillow，请运行: python -m pip install -r requirements.txt") from exc
    output = Path(output)
    step = max(1, round(result.sample_rate / result.frame_hop / fps))
    frame_ids = list(range(0, len(result.bands), step))
    frames = []
    delayed_values = None
    delay_alpha = _delay_alpha(delay_strength) if delay_strength > 0.0 else 1.0
    for index, frame in enumerate(frame_ids):
        if delay_strength > 0.0:
            instant = _instant_frame_values(result, frame, low_focus)
            if delayed_values is None:
                delayed_values = instant
            else:
                delayed_values += (instant - delayed_values) * delay_alpha
            frames.append(_draw_pillow_frame(result, frame, width, height, low_focus, values=delayed_values))
        else:
            frames.append(_draw_pillow_frame(result, frame, width, height, low_focus))
        if progress:
            progress((index + 1) / len(frame_ids))
    if not frames:
        raise RuntimeError("没有可导出的分析帧")
    duration_ms = round(1000 / fps)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0, disposal=2, optimize=False)


def export_png(result: AnalysisResult, output: str | Path, frame: int, width: int = 1280, height: int = 480, low_focus: float = 0.0, delay_strength: float = 0.0) -> None:
    image = _draw_pillow_frame(result, frame, width, height, low_focus, delay_strength)
    image.save(output, format="PNG")


def export_json(result: AnalysisResult, output: str | Path) -> None:
    payload = {
        "sample_rate": result.sample_rate,
        "duration": result.duration,
        "hop": result.frame_hop,
        "bands_hz": [round(float(x), 3) for x in result.band_frequencies],
        "times": [round(float(x), 4) for x in result.times],
        "pulse": [round(float(x), 4) for x in result.pulse],
        "beat_times": [round(float(result.times[i]), 4) for i in result.beat_frames],
        "bands": result.bands.round(4).tolist(),
    }
    Path(output).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def preflight_runtime() -> tuple[bool, str]:
    """Prepare every required runtime component before opening the GUI."""
    ok, details = _prepare_python_runtime()
    if not ok:
        return False, details
    ffmpeg = _ffmpeg_path(auto_install=True)
    if not ffmpeg:
        return False, "未找到可用的 FFmpeg。" + _ffmpeg_install_message()
    try:
        check = subprocess.run([ffmpeg, "-version"], capture_output=True, timeout=10, check=False)
    except Exception as exc:
        return False, f"FFmpeg 启动失败：{exc}"
    if check.returncode != 0:
        return False, "找到 FFmpeg，但无法启动。请检查应用目录 runtime/ffmpeg/ffmpeg.exe。"
    return True, ""


class SpectrumApp:
    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = root
        self.root.title("自动频谱生成器")
        self._set_window_icon()
        self.root.geometry("1180x760")
        self.root.minsize(860, 600)
        self.root.configure(bg="#111111")
        self.audio: Optional[AudioData] = None
        self.result: Optional[AnalysisResult] = None
        self.current_frame = 0
        self.playing = False
        self.analysis_thread: Optional[threading.Thread] = None
        self.export_thread: Optional[threading.Thread] = None
        self.selected_path: Optional[Path] = None
        self.auto_export = False
        self.pending_output: Optional[Path] = None
        self.drag_drop_enabled = False
        self.low_focus = tk.DoubleVar(value=0.0)
        self.delay_enabled = tk.BooleanVar(value=True)
        self.delay_strength = tk.DoubleVar(value=0.55)
        self.delay_check_text = tk.StringVar(value="☑ 丝滑延迟")
        self.time_value = tk.DoubleVar(value=0.0)
        self.backend_choice = tk.StringVar(value=default_backend_choice())
        self.active_backend = _normalise_backend_preference(self.backend_choice.get())
        self.status = tk.StringVar(value="拖入或选择音频，然后点击开始生成 MP4")
        self.file_label = tk.StringVar(value="尚未选择音频")
        self._build_ui()

    def _set_window_icon(self):
        """Use the bundled music-note icon for the window and taskbar entry."""
        icon_path = Path(__file__).resolve().parent / "spectrum_icon.ico"
        if not icon_path.exists():
            return
        try:
            self.root.iconbitmap(default=str(icon_path))
        except Exception:
            pass
        try:
            from PIL import Image, ImageTk

            icon = Image.open(icon_path).convert("RGBA").resize((32, 32), Image.Resampling.LANCZOS)
            self._window_icon = ImageTk.PhotoImage(icon)
            self.root.iconphoto(True, self._window_icon)
        except Exception:
            pass

    def _build_ui(self):
        tk, ttk = self.tk, self.ttk
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background="#111111")
        style.configure("TLabel", background="#111111", foreground="#dddddd")
        style.configure("Muted.TLabel", background="#111111", foreground="#818181")
        style.configure("TButton", padding=(10, 6))
        style.configure("Horizontal.TScale", background="#111111", troughcolor="#333333")

        header = ttk.Frame(self.root, padding=(18, 16, 18, 8))
        header.pack(fill="x")
        ttk.Label(header, text="自动频谱生成器", font=("Microsoft YaHei UI", 18, "bold"), foreground="#ffffff").pack(side="left")
        ttk.Label(header, textvariable=self.file_label, style="Muted.TLabel").pack(side="left", padx=(18, 0))
        self.start_button = ttk.Button(header, text="开始生成 MP4", command=self.start_generation, state="disabled")
        self.start_button.pack(side="right")
        ttk.Button(header, text="选择音频", command=self.choose_file).pack(side="right", padx=(0, 8))
        if hasattr(self.root, "drop_target_register"):
            try:
                dnd_type = getattr(self.root, "_spectrum_dnd_files", "DND_Files")
                self.root.drop_target_register(dnd_type)
                self.root.dnd_bind("<<Drop>>", self.handle_drop)
                self.drag_drop_enabled = True
            except Exception:
                pass

        preview_frame = ttk.Frame(self.root, padding=(18, 4, 18, 8))
        preview_frame.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(preview_frame, background="#000000", highlightthickness=1, highlightbackground="#242424")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _event: self.draw_frame())
        if self.drag_drop_enabled:
            try:
                self.canvas.drop_target_register(dnd_type)
                self.canvas.dnd_bind("<<Drop>>", self.handle_drop)
            except Exception:
                self.drag_drop_enabled = False

        transport = ttk.Frame(self.root, padding=(18, 2, 18, 6))
        transport.pack(fill="x")
        self.play_button = ttk.Button(transport, text="播放预览", command=self.toggle_play, state="disabled")
        self.play_button.pack(side="left")
        ttk.Button(transport, text="导出 GIF", command=self.save_gif, state="normal").pack(side="left", padx=(8, 0))
        self.time_scale = ttk.Scale(transport, from_=0, to=1, variable=self.time_value, command=self.seek, state="disabled")
        self.time_scale.pack(side="left", fill="x", expand=True, padx=(18, 8))
        self.time_label = ttk.Label(transport, text="00:00 / 00:00", style="Muted.TLabel", width=16)
        self.time_label.pack(side="right")

        controls = ttk.Frame(self.root, padding=(18, 2, 18, 8))
        controls.pack(fill="x")
        ttk.Label(controls, text="低音聚焦").pack(side="left")
        ttk.Label(controls, text="弱").pack(side="left", padx=(8, 4))
        ttk.Scale(
            controls,
            from_=0.0,
            to=4.0,
            length=520,
            variable=self.low_focus,
            command=lambda _v: self.draw_frame(),
        ).pack(side="left", fill="x", expand=True, padx=(0, 4))
        ttk.Label(controls, text="强").pack(side="left")

        delay_controls = ttk.Frame(self.root, padding=(18, 0, 18, 8))
        delay_controls.pack(fill="x")
        self.delay_toggle = tk.Checkbutton(
            delay_controls,
            textvariable=self.delay_check_text,
            variable=self.delay_enabled,
            command=self._delay_changed,
            indicatoron=False,
            anchor="w",
            padx=2,
            pady=2,
            borderwidth=0,
            highlightthickness=0,
            relief="flat",
            background="#111111",
            activebackground="#111111",
            foreground="#dddddd",
            activeforeground="#ffffff",
            selectcolor="#111111",
            font=("Segoe UI Symbol", 10),
        )
        self.delay_toggle.pack(side="left")
        ttk.Label(delay_controls, text="强度").pack(side="left", padx=(14, 6))
        ttk.Scale(
            delay_controls,
            from_=0.0,
            to=1.0,
            length=420,
            variable=self.delay_strength,
            command=lambda _v: self.draw_frame(),
        ).pack(side="left", fill="x", expand=True)

        footer = ttk.Frame(self.root, padding=(18, 0, 18, 12))
        footer.pack(fill="x")
        self.progress = ttk.Progressbar(footer, mode="determinate", maximum=100)
        self.progress.pack(fill="x", side="left", expand=True)
        backend_panel = ttk.Frame(footer)
        backend_panel.pack(side="right")
        ttk.Label(backend_panel, text="渲染方式", style="Muted.TLabel").pack(side="left", padx=(12, 6))
        self.backend_menu = ttk.Combobox(
            backend_panel,
            textvariable=self.backend_choice,
            values=("CUDA / GPU", "CPU / NumPy"),
            state="readonly",
            width=17,
        )
        self.backend_menu.pack(side="left")
        self.backend_menu.bind("<<ComboboxSelected>>", self._backend_changed)
        ttk.Label(footer, textvariable=self.status, style="Muted.TLabel", width=40).pack(side="right", padx=(12, 0))
        if not self.drag_drop_enabled:
            self.status.set("拖拽未启用，请安装 tkinterdnd2；也可以点击选择音频")

    def _delay_changed(self):
        self.delay_check_text.set("☑ 丝滑延迟" if self.delay_enabled.get() else "☐ 丝滑延迟")
        self.draw_frame()

    def _backend_preference(self) -> str:
        return _normalise_backend_preference(self.backend_choice.get())

    def _backend_changed(self, _event=None):
        selected = self._backend_preference()
        if selected == "cuda" and not cuda_backend_available():
            details = _CUDA_BOOTSTRAP_ERROR or _CUDA_BACKEND_ERROR or "启动预检未能初始化 CUDA 设备"
            self.status.set(f"CUDA 不可用：{details}")
        else:
            self.status.set(f"已选择 {analysis_backend(selected)}")

    def choose_file(self):
        from tkinter import filedialog

        path = filedialog.askopenfilename(
            title="选择音频",
            filetypes=[("音频文件", "*.wav *.mp3 *.flac *.ogg *.m4a *.aac"), ("所有文件", "*.*")],
        )
        if path:
            self.set_selected_file(path)

    def handle_drop(self, event):
        """Accept the first dropped audio path when tkinterdnd2 is installed."""
        try:
            paths = self.root.tk.splitlist(event.data)
        except Exception:
            paths = [event.data]
        if paths:
            self.set_selected_file(paths[0])

    def set_selected_file(self, path: str):
        candidate = Path(path)
        if not candidate.exists() or candidate.is_dir():
            return
        self.selected_path = candidate
        program_folder = Path(__file__).resolve().parent
        self.pending_output = program_folder / f"{candidate.stem}_spectrum.mp4"
        self.file_label.set(candidate.name)
        self.status.set("已选择音频，点击“开始生成 MP4”")
        self.start_button.configure(state="normal")

    def start_generation(self):
        if not self.selected_path:
            return
        self.auto_export = True
        self.load_and_analyse(str(self.selected_path))

    def load_and_analyse(self, path: str):
        from tkinter import messagebox

        if self.analysis_thread and self.analysis_thread.is_alive():
            return
        backend = self._backend_preference()
        self.status.set("正在读取音频…")
        self.progress["value"] = 0
        self.play_button["state"] = "disabled"
        self.start_button["state"] = "disabled"
        self.backend_menu.configure(state="disabled")
        self.file_label.set(Path(path).name)

        def worker():
            try:
                if Path(path).suffix.lower() != ".wav" and not _ffmpeg_path(auto_install=False):
                    self.root.after(0, lambda: self.status.set("正在准备通用 FFmpeg 解码器…"))
                audio = load_audio(path)
                self.root.after(0, lambda: self.status.set("正在分析频谱…"))
                result = analyse_audio(
                    audio,
                    progress=lambda value: self.root.after(0, lambda: self.progress.configure(value=value * 100)),
                    backend=backend,
                )
            except Exception as exc:  # UI boundary: show a useful message and keep the app alive.
                self.root.after(0, lambda: messagebox.showerror("分析失败", str(exc)))
                self.root.after(0, lambda: self.status.set("分析失败，请检查文件和依赖"))
                self.root.after(0, lambda: self.start_button.configure(state="normal"))
                self.root.after(0, lambda: self.backend_menu.configure(state="readonly"))
                return
            self.root.after(0, lambda: self._analysis_ready(audio, result, backend))

        self.analysis_thread = threading.Thread(target=worker, daemon=True)
        self.analysis_thread.start()

    def _analysis_ready(self, audio: AudioData, result: AnalysisResult, backend: str):
        self.audio, self.result = audio, result
        self.active_backend = backend
        self.current_frame = 0
        self.time_value.set(0)
        self.time_scale.configure(to=max(0.001, result.duration), state="normal")
        self.play_button.configure(state="normal")
        minutes, seconds = divmod(int(result.duration), 60)
        self.status.set(f"{analysis_backend(backend)} · {len(result.beat_frames)} 个鼓点 · {len(result.bands)} 帧 · {minutes:02d}:{seconds:02d}")
        self.progress["value"] = 100
        self.backend_menu.configure(state="readonly")
        self.draw_frame()
        if self.auto_export and self.pending_output:
            self.export_mp4_async(self.pending_output)
        else:
            self.start_button.configure(state="normal")

    def export_mp4_async(self, output: Path):
        from tkinter import messagebox

        if not self.result or not self.audio:
            return
        self.auto_export = False
        self.status.set(f"正在生成 MP4（{analysis_backend(self.active_backend)}）…")
        self.progress["value"] = 0
        low_focus = self.low_focus.get()
        delay_strength = self.delay_strength.get() if self.delay_enabled.get() else 0.0

        def worker():
            try:
                export_mp4(
                    self.result,
                    self.audio,
                    output,
                    fps=DEFAULT_VIDEO_FPS,
                    width=DEFAULT_VIDEO_WIDTH,
                    height=DEFAULT_VIDEO_HEIGHT,
                    low_focus=low_focus,
                    delay_strength=delay_strength,
                    progress=lambda value: self.root.after(0, lambda: self.progress.configure(value=value * 100)),
                )
            except Exception as exc:
                self.root.after(0, lambda: messagebox.showerror("视频生成失败", str(exc)))
                self.root.after(0, lambda: self.status.set("视频生成失败，请检查 FFmpeg"))
                self.root.after(0, lambda: self.start_button.configure(state="normal"))
                return
            self.root.after(0, lambda: self.status.set(f"已生成 {output.name}（程序目录 · {DEFAULT_VIDEO_WIDTH}×{DEFAULT_VIDEO_HEIGHT}）"))
            self.root.after(0, lambda: self.start_button.configure(state="normal"))

        self.export_thread = threading.Thread(target=worker, daemon=True)
        self.export_thread.start()

    def seek(self, value):
        if not self.result:
            return
        self.current_frame = int(float(value) * self.result.sample_rate / self.result.frame_hop)
        self.draw_frame()

    def draw_frame(self):
        if not self.result:
            return
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        self.canvas.delete("all")
        delay_strength = self.delay_strength.get() if self.delay_enabled.get() else 0.0
        values = frame_values(self.result, self.current_frame, self.low_focus.get(), delay_strength)
        center = height // 2
        gap = max(1, round(width / len(values) * 0.18))
        bar_width = max(1, math.floor(width / len(values)) - gap)
        for index, value in enumerate(values):
            x = index * width / len(values) + gap / 2
            half = max(1.0, value * height * 0.46)
            radius = min(bar_width / 2.0, half)
            self.canvas.create_rectangle(x, center - half + radius, x + bar_width, center + half - radius, fill="#ffffff", outline="")
            self.canvas.create_oval(x, center - half, x + bar_width, center - half + radius * 2, fill="#ffffff", outline="")
            self.canvas.create_oval(x, center + half - radius * 2, x + bar_width, center + half, fill="#ffffff", outline="")
        current = self.current_frame * self.result.frame_hop / self.result.sample_rate
        self.time_value.set(current)
        total_minutes, total_seconds = divmod(int(self.result.duration), 60)
        cur_minutes, cur_seconds = divmod(int(current), 60)
        self.time_label.configure(text=f"{cur_minutes:02d}:{cur_seconds:02d} / {total_minutes:02d}:{total_seconds:02d}")

    def toggle_play(self):
        if not self.result:
            return
        self.playing = not self.playing
        self.play_button.configure(text="暂停预览" if self.playing else "播放预览")
        if self.playing:
            self.tick()

    def tick(self):
        if not self.playing or not self.result:
            return
        self.current_frame += 1
        if self.current_frame >= len(self.result.bands):
            self.current_frame = 0
        self.draw_frame()
        self.root.after(max(10, round(self.result.frame_hop / self.result.sample_rate * 1000)), self.tick)

    def save_png(self):
        from tkinter import filedialog, messagebox

        if not self.result:
            messagebox.showinfo("提示", "请先导入并分析音频")
            return
        output = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG 图片", "*.png")])
        if output:
            try:
                delay_strength = self.delay_strength.get() if self.delay_enabled.get() else 0.0
                export_png(self.result, output, self.current_frame, low_focus=self.low_focus.get(), delay_strength=delay_strength)
                self.status.set(f"已导出 {Path(output).name}")
            except Exception as exc:
                messagebox.showerror("导出失败", str(exc))

    def save_gif(self):
        from tkinter import filedialog, messagebox

        if not self.result:
            messagebox.showinfo("提示", "请先导入并分析音频")
            return
        output = filedialog.asksaveasfilename(defaultextension=".gif", filetypes=[("GIF 动画", "*.gif")])
        if not output:
            return
        self.status.set("正在导出 GIF…")
        try:
            delay_strength = self.delay_strength.get() if self.delay_enabled.get() else 0.0
            export_gif(self.result, output, low_focus=self.low_focus.get(), delay_strength=delay_strength, progress=lambda value: self.root.after(0, lambda: self.progress.configure(value=value * 100)))
            self.status.set(f"已导出 {Path(output).name}")
        except Exception as exc:
            messagebox.showerror("导出失败", str(exc))


def run_cli(args) -> int:
    audio = load_audio(args.input)
    result = analyse_audio(audio)
    if args.export_mp4:
        export_mp4(result, audio, args.export_mp4, fps=args.fps, width=args.width, height=args.height)
    if args.export_gif:
        export_gif(result, args.export_gif, fps=args.fps, width=args.width, height=args.height)
    if args.export_png:
        export_png(result, args.export_png, frame=0, width=args.width, height=args.height)
    if args.export_json:
        export_json(result, args.export_json)
    if not any((args.export_mp4, args.export_gif, args.export_png, args.export_json)):
        print(f"分析完成: {audio.title} | {result.duration:.2f}s | {len(result.beat_frames)} beats | {analysis_backend()}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="生成黑底白色 EDM 横向频谱")
    parser.add_argument("--input", help="音频文件路径；不指定则启动桌面界面")
    parser.add_argument("--export-mp4", help="导出带原音频的 MP4 路径")
    parser.add_argument("--export-gif", help="导出 GIF 动画路径")
    parser.add_argument("--export-png", help="导出第一帧 PNG 路径")
    parser.add_argument("--export-json", help="导出完整分析数据 JSON 路径")
    parser.add_argument("--fps", type=int, default=DEFAULT_VIDEO_FPS)
    parser.add_argument("--width", type=int, default=DEFAULT_VIDEO_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_VIDEO_HEIGHT)
    args = parser.parse_args()
    ready, details = preflight_runtime()
    if not ready:
        print("启动预检失败，未进入主界面。", file=sys.stderr)
        print(details, file=sys.stderr)
        return 1
    if args.input:
        try:
            return run_cli(args)
        except Exception as exc:
            print(f"错误: {exc}", file=sys.stderr)
            return 1
    try:
        from tkinterdnd2 import DND_FILES, TkinterDnD

        root = TkinterDnD.Tk()
        root._spectrum_dnd_files = DND_FILES
    except Exception:
        import tkinter as tk

        root = tk.Tk()
    SpectrumApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
