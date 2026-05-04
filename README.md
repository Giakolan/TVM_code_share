🧪 一次性完整流程（從 0 到可跑）

0️⃣ 系統與工具（含 git-lfs）
sudo apt update
sudo apt install -y git git-lfs build-essential cmake llvm llvm-dev clang \
  python3 python3-pip ffmpeg pkg-config

git lfs install

1️⃣ Python 套件（把你們踩過的全補）
pip3 install -U pip
pip3 install -U onnx transformers sentencepiece soundfile librosa yt-dlp \
  ninja psutil setuptools wheel cython pillow scikit-build-core pybind11
  
2️⃣ 下載專案 + 取回 LFS 模型
cd ~
git clone https://github.com/lewis21456/tvm-byoc-kiwipedia.git
cd tvm-byoc-kiwipedia

git lfs pull
✔ 檢查 ONNX 不是 pointer（避免 DecodeError）
head -n 3 whisper-tiny/onnx/encoder_model.onnx
若看到 git-lfs.github.com/spec/v1 → 再跑一次 git lfs pull

3️⃣ 修正硬編碼路徑（避免 /home/xxx 問題）
cd ~/tvm-byoc-kiwipedia
sed -i "s|/home/lewis56|$HOME|g" $(grep -rl "/home/lewis56" .)

4️⃣ 清除可能殘留的舊 tvm_ffi（避免 import core 問題）
pip3 uninstall -y tvm-ffi tvm_ffi || true
find ~/.local/lib/python3.10/site-packages -maxdepth 1 \
  \( -name "*.pth" -o -name "*.egg-link" \) | \
  xargs grep -l "/home/" 2>/dev/null | xargs -r rm -f
  
5️⃣ 編譯 TVM（開 LLVM + kiwipedia）
cd ~/tvm-byoc-kiwipedia/tvm
rm -rf build
mkdir build
cd build

cmake .. -DUSE_LLVM=ON -DUSE_KIWIPEDIA_CODEGEN=ON -DUSE_KIWIPEDIA_RUNTIME=ON
cmake --build . --parallel $(nproc)

6️⃣ 編譯 kiwipedia 的 matmul（必要）
cd ~/tvm-byoc-kiwipedia/tvm/src/runtime/contrib/kiwipedia

g++ -O3 -shared -fPIC libmatmul.cpp \
  -I$HOME/tvm-byoc-kiwipedia/tvm/3rdparty/tvm-ffi/3rdparty/dlpack/include \
  -o libmatmul.so
7️⃣ 安裝 tvm_ffi（解決 cannot import core）

cd ~/tvm-byoc-kiwipedia/tvm/3rdparty/tvm-ffi
pip3 install -e .

8️⃣ 設定環境變數（本次 terminal 生效）
export PYTHONPATH=$HOME/tvm-byoc-kiwipedia/tvm/python:$HOME/tvm-byoc-kiwipedia/tvm/3rdparty/tvm-ffi/python
export LD_LIBRARY_PATH=$HOME/tvm-byoc-kiwipedia/tvm/build:$HOME/tvm-byoc-kiwipedia/tvm/build/lib:$LD_LIBRARY_PATH
export KIWIPEDIA_MATMUL_SO=$HOME/tvm-byoc-kiwipedia/tvm/src/runtime/contrib/kiwipedia/libmatmul.so
export PATH=$HOME/.local/bin:$PATH

9️⃣ 驗證關鍵功能（確保不會在 compile 爆）
python3 -c "import tvm; print(tvm.get_global_func('target.build.llvm', True))"
python3 -c "import tvm; print(tvm.get_global_func('runtime.kiwipedia_runtime_create', True))"
兩行都不應該是 None

🔟 Compile ONNX
cd ~/tvm-byoc-kiwipedia/whisper-tiny/onnx
python3 compile_encoder.py
python3 compile_decoder.py
python3 compile_decoder_with_past.py

（只有 warning 屬正常）

1️⃣1️⃣ 準備音檔（避免 soundfile 錯）
cd ~/tvm-byoc-kiwipedia/whisper-tiny

python3 -m yt_dlp --extract-audio --audio-format wav \
  --download-sections "*00:00:00-00:00:20" \
  -o "audio.wav" \
  https://youtu.be/7NdSnwDkPYs
  
1️⃣2️⃣ 執行推論
python3 inference.py
🧯 若仍出錯，對應快速修復

Pillow / WhisperProcessor 錯

pip3 install -U pillow transformers sentencepiece

缺 PyTorch（若之後需要）

pip3 install torch

audio.wav 錯或不存在

file audio.wav
ffprobe audio.wav

不正常就重抓

matmul 載入失敗

ls $KIWIPEDIA_MATMUL_SO

沒有就回到第 6 步重編

🟢 日常使用（之後每次只做）
cd ~/tvm-byoc-kiwipedia
#（建議你把下面存成 env_tvm.sh 再 source）
export PYTHONPATH=$HOME/tvm-byoc-kiwipedia/tvm/python:$HOME/tvm-byoc-kiwipedia/tvm/3rdparty/tvm-ffi/python
export LD_LIBRARY_PATH=$HOME/tvm-byoc-kiwipedia/tvm/build:$HOME/tvm-byoc-kiwipedia/tvm/build/lib:$LD_LIBRARY_PATH
export KIWIPEDIA_MATMUL_SO=$HOME/tvm-byoc-kiwipedia/tvm/src/runtime/contrib/kiwipedia/libmatmul.so
export PATH=$HOME/.local/bin:$PATH

cd whisper-tiny
python3 inference.py
