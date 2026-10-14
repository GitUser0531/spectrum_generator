#注意

为了方便直接开包即用，这里也提供程序运行依赖，在Release中找到tag为Runtime的文件根据指示进行放置

# 自动频谱生成器

一个面向 EDM、DJ SET 和长音频的黑底白色横向频谱生成器。它会把频率分成对数频段，强化低频区域，并用频谱通量检测 kick、snare 等瞬态。鼓点出现时只推动低频并压低其他频段，适合 drop 的律动效果。

## 安装与启动

```bash
python -m pip install -r requirements.txt
python spectrum_generator.py
```

程序进入主界面前会做一次运行环境预检：NumPy、Pillow、拖拽组件和 FFmpeg 都可用后才会打开窗口。项目的 `runtime/packages` 已包含离线 Python wheel 和 FFmpeg 二进制，预检会优先使用这些本地文件，不会等到导入音频时才补依赖。发布时也可以把 FFmpeg 放在 `runtime/ffmpeg/ffmpeg.exe`。

有 NVIDIA 显卡时，启动预检会先检测 `nvidia-smi`，再从项目 `runtime/wheels` 离线补齐 CUDA 版 CuPy；本地没有对应 wheel 时才访问 PyPI。当前窗口右下角的“渲染方式”菜单可以手动选择 CUDA 或 CPU，检测到可用 CUDA 时默认选中 CUDA。

```bash
python -m pip install -r requirements-gpu.txt
```

程序启动时会自动检测 CUDA。可用时，分析状态会显示 `CUDA / GPU`，分块 FFT 在 GPU 上执行；没有 CUDA 后端时菜单会保留 CUDA 选项并提示补齐方式，也可以切换为 CPU / NumPy。

FLAC、OGG 等格式优先使用 `soundfile` 解码；MP3、M4A、AAC 等格式和 MP4 输出需要 FFmpeg。程序会优先使用系统中的 FFmpeg，否则自动使用 `imageio-ffmpeg` 自带的 FFmpeg；首次自动补齐时需要网络。离线时也可以设置环境变量指定已有的程序：

```powershell
$env:FFMPEG_PATH = "C:\\ffmpeg\\bin\\ffmpeg.exe"
```

如果当前 Python 环境没有这些依赖，启动预检会先从 `runtime/wheels` 离线安装；本地包不匹配时才尝试联网安装到用户缓存目录。安装失败会直接停止启动并显示原因，不会让程序进入不可用的主界面。

WAV 文件可以由程序直接读取，支持常见的 8/16/24/32 位 PCM。拖拽功能由 `tkinterdnd2` 提供；如果窗口底部提示拖拽未启用，请重新运行 `python -m pip install -r requirements.txt`，也可以继续使用“选择音频”。

## 使用方式

1. 点击“选择音频”，或把音频文件拖进窗口。
2. 点击“开始生成 MP4”。程序会在程序所在文件夹写入 `原文件名_spectrum.mp4`，视频默认是 1280×360、60 FPS、H.264，音频使用 AAC 96 kbps，适合 DJ SET 预览和发布。
3. 分析完成后可以使用合并后的“低音聚焦”滑块控制低频增强和鼓点响应，也可以打开“丝滑延迟”并调节强度，让频谱回落和连续帧过渡更平滑。低音聚焦不会增强超低频，而会相对压低超低频和高中频，把空间留给更高的鼓点和低中频区域。也可以拖动时间轴定位。GIF 是无声音的频谱动画预览；PNG 是单帧截图，主界面已隐藏，只保留命令行导出。

视频使用流式管道交给 FFmpeg 编码，不会把整段视频帧保存在内存里。频谱分析也以 2048 帧为一块，只保留 96 个视觉频段和鼓点特征，长音频的内存占用主要来自解码后的单声道采样。

也支持无界面批量处理：

```bash
python spectrum_generator.py --input track.mp3 --export-mp4 track_spectrum.mp4 --export-json track.json
```

如果 FFmpeg 编译了 NVENC，也可以设置 `SPECTRUM_USE_NVENC=1` 让视频编码尝试使用 `h264_nvenc`；FFT 的 CUDA 加速由 CuPy（或已有的 CUDA PyTorch）控制。JSON 中包含每个时间帧的频谱值、鼓点时间和中心频率，可用于后续网页动画或在视频剪辑软件中对齐节拍。
