# ═══════════════════════════════════════════════════════════
# sherpa-post-asr — Windows PowerShell 快速部署
# ═══════════════════════════════════════════════════════════
# 使用方法：在 PowerShell 中逐行粘贴执行

# ── 0. 准备工作目录 ──
cd ~\Projects
mkdir sherpa-post-asr -Force
cd sherpa-post-asr

# ── 1. 克隆仓库（或用 ZIP 下载）──
# 方式 A：git clone（需要先安装 git）
git clone https://github.com/Hi-Barry/sherpa-post-asr.git .
# 如果没装 git，去 https://github.com/Hi-Barry/sherpa-post-asr 点 Code → Download ZIP

# ── 2. 创建 Python 虚拟环境 ──
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # 激活虚拟环境
# 如果提示"无法加载文件...禁止执行脚本"，先执行：
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# ── 3. 安装依赖 ──
pip install sherpa-onnx sounddevice numpy soundfile

# ── 4. 下载模型 ──
# Zipformer-CTC 模型（中文 ASR）
mkdir models\zipformer-ctc-zh-int8 -Force
cd models\zipformer-ctc-zh-int8
# 如果装了 curl（Windows 10/11 自带）：
curl.exe -L -o model.tar.bz2 ^
  https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03.tar.bz2
# 或用 PowerShell 的 Invoke-WebRequest：
# Invoke-WebRequest -Uri "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03.tar.bz2" -OutFile model.tar.bz2

# 解压（需要 7-Zip 或 tar，Windows 10 1803+ 自带 tar）
tar xf model.tar.bz2
cd ..\..

# Silero VAD 模型（语音活动检测）
mkdir models\vad -Force
cd models\vad
curl.exe -L -o silero_vad.onnx ^
  https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
cd ..\..

# ── 5. 验证模型文件 ──
dir models\zipformer-ctc-zh-int8\sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03\model.int8.onnx
dir models\vad\silero_vad.onnx
# 应该看到两个文件存在

# ── 6. 启动麦克风实时转写 ──
# 先查看可用麦克风：
python scripts\mic_transcribe.py --list-devices

# 实时模式（持续监听，说话自动转写）：
python scripts\mic_transcribe.py

# 指定设备（如果默认设备不对）：
python scripts\mic_transcribe.py --device 0

# 单次模式（回车开始/停止）：
python scripts\mic_transcribe.py --one-shot

# ── 7. （可选）跑 benchmark ──
# 先下载测试音频的噪声文件（太大没提交到 git）
# 或者在干净音频上直接测试：
python scripts\benchmark_zipformer_ctc.py

# ═══════════════════════════════════════════════════════════
# 常见问题
# ═══════════════════════════════════════════════════════════

# Q: pip install sherpa-onnx 失败？
# A: Windows 需要 VC++ 运行时，装一下：
#    https://aka.ms/vs/17/release/vc_redist.x64.exe

# Q: sounddevice 找不到麦克风？
# A: 检查 Windows 录音权限：
#    设置 → 隐私和安全性 → 麦克风 → 允许应用访问
#    或者用 --list-devices 查看可用设备编号

# Q: 模型下载太慢？
# A: 可以用浏览器直接下载，然后把文件放到对应目录：
#    - Zipformer-CTC: https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03.tar.bz2
#    - Silero VAD:   https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

# Q: 转写结果全是乱码/英文？
# A: 确保麦克风采集到的是中文语音。
#    Zipformer-CTC 只支持中文，不支持英文混合识别。

# Q: "Overflow!" 警告？
# A: 说话时间太长，VAD缓冲区满了。不影响结果，已修复。
