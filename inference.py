import numpy as np
import tvm
from tvm import relax, runtime
from tvm.relax import VirtualMachine
from transformers import WhisperProcessor, WhisperTokenizer

from datetime import datetime
import csv
from collections import defaultdict
import numpy as np
import soundfile as sf
from scipy import signal
#import os

# Global aggregation dict: {Name: [total_duration_us, total_count]}
profile_agg = defaultdict(lambda: [0, 0])
#os.makedirs("./profile_data", exist_ok=True)
start_time_all = datetime.now()
print("Start of all:", start_time_all)

# === 初始化空的 16 個 KV（給 prefill 和 step-by-step 共用）===
def init_zero_past_kv(num_layers=4, num_heads=6, head_dim=64,
                      decoder_seq_len=0, encoder_seq_len=1500, dtype="float32"):
    shape_decoder = (1, num_heads, decoder_seq_len, head_dim)
    shape_encoder = (1, num_heads, encoder_seq_len, head_dim)
    kvs = []
    for _ in range(num_layers):
        kvs += [
            tvm.runtime.tensor(np.zeros(shape_decoder, dtype=dtype)),  # self.key
            tvm.runtime.tensor(np.zeros(shape_decoder, dtype=dtype)),  # self.value
            tvm.runtime.tensor(np.zeros(shape_encoder, dtype=dtype)),  # cross.key
            tvm.runtime.tensor(np.zeros(shape_encoder, dtype=dtype))   # cross.value
        ]
    return kvs

def insert_profile_report(csv_str):
    """Insert one profile report (CSV string format) into the global aggregator."""
    reader = csv.DictReader(csv_str.strip().splitlines())
    for row in reader:
        name = row["Name"]
        duration = float(row["Duration (us)"])
        count = int(row["Count"])
        profile_agg[name][0] += duration
        profile_agg[name][1] += count

# === 載入模型與 tokenizer ===
processor = WhisperProcessor.from_pretrained("./")
tokenizer = WhisperTokenizer.from_pretrained("./")

# === 音訊轉 mel spectrogram ===


# === 1. Load audio ===
waveform, sr = sf.read("audio.wav")

# === 2. Resample to 16kHz if needed ===
target_sr = 16000
if sr != target_sr:
    num_samples = int(len(waveform) * target_sr / sr)
    waveform = signal.resample(waveform, num_samples)
    sr = target_sr

# === 3. Convert to mono ===
if waveform.ndim > 1:
    waveform = waveform.mean(axis=1)

# === 4. Pass into Hugging Face processor (same as torchaudio flow) ===
inputs = processor(waveform, sampling_rate=16000, return_tensors="np")

# === 5. Get float32 mel features ===
mel = inputs.input_features.astype("float32")

print("Mel shape:", mel.shape)


# === Encoder ===
encoder_vm = VirtualMachine(runtime.load_module("./onnx/encoder_model.so"), tvm.cpu(), profile=True)

# === Profile code block ===

#Profile the encoder execution
#profile_report = encoder_vm.profile("main", tvm.runtime.tensor(mel)).csv()
#insert_profile_report(profile_report)
#Convert to CSV and save to file
#with open("./profile_data/encoder.csv", "w") as f:
#    f.write(profile_report)

# === End of Profile code block ===

start_time = datetime.now()
print("Start of encoder:", start_time)
encoder_out = encoder_vm["main"](tvm.runtime.tensor(mel))  # shape: (1, 1500, 384)
end_time = datetime.now()
print("End of encoder:", end_time)
print("Encoder takes: ", (end_time-start_time).total_seconds())

# === Decoder Step 0: Prefill ===
# Initialize decoder VM with profiling enabled


start_token = 50258
eos_token = tokenizer.eos_token_id
tokens = [start_token]
input_ids = np.array([[start_token]], dtype="int64")
past_kvs = init_zero_past_kv()
inputs = [tvm.runtime.tensor(input_ids), encoder_out]

decoder_prefill_vm = VirtualMachine(
    runtime.load_module("./onnx/decoder_model.so"), 
    tvm.cpu(), 
    profile=True
)

# Initialize empty KV (self + cross) for prefill decoder

# === Decoder profiling ===


#print("\n=== Step 0 (Prefill) - Profiling ===")
#profile_report = decoder_prefill_vm.profile("main", *inputs).csv()
#insert_profile_report(profile_report)
# # Save CSV-formatted profiling report
#with open("./profile_data/decoder_prefill.csv", "w") as f:
#    f.write(profile_report)
# === End of Decoder profiling ===

# Get the actual output (without profiling)
start_time = datetime.now()
print("Start of decoder prefill:", start_time)
out = decoder_prefill_vm["main"](*inputs)
end_time = datetime.now()
print("End of decoder prefill:", end_time)
print("Decoder prefill takes: ", (end_time-start_time).total_seconds())


logits = out[0].numpy()
next_token = int(np.argmax(logits[0, -1]))

top5 = np.argsort(logits[0, -1])[-5:][::-1]
#print(f"\n[DEBUG] prefill top5 tokens:")
#for tid in top5:
#    tid = int(tid)
#    print(tid, repr(tokenizer.decode([tid])), float(logits[0, -1, tid]))

#print("[DEBUG] prefill next_token:", next_token, repr(tokenizer.decode([next_token])))

assert logits.ndim == 2 or logits.ndim == 3, "logits 維度不符"

tokens.append(next_token)
#print("[DEBUG] current text:", repr(tokenizer.decode(tokens, skip_special_tokens=True)))
# print(f"⬆️ Next token: {next_token} ({tokenizer.decode([next_token])})")

# 將 decoder 回傳的 16 個 KV 擷取出來
decoder_kvs = list(out[1:])  # out[1]~out[16]

if next_token == eos_token:
    print("🛑 遇到 <eos>，結束解碼")
    transcript = tokenizer.decode(tokens, skip_special_tokens=True)
    print("\n📝 Transcription:\n", transcript)
    exit()

# === Decoder Step 1~N: step-by-step 解碼 ===

# === Decoder profiling ===
decoder_vm = VirtualMachine(
    runtime.load_module("./onnx/decoder_with_past_model.so"), 
    tvm.cpu(),
    profile=True  # Enable profiling
)
# === Decoder profiling ===



max_length = 64
all_reports = []  # Store all profiling reports

start_time = datetime.now()
print(f"Start of decoder token generation: {start_time}")

for step in range(1, max_length):
    # print(f"\n=== Step {step} ===")
    input_ids = np.array([[tokens[-1]]], dtype="int64")
    inputs = [tvm.runtime.tensor(input_ids)] + decoder_kvs

    # Profile every step (optional: skip warm-up steps)
    # === Decoder profiling ===

    #profile_report = decoder_vm.profile("main", *inputs).csv()
    #insert_profile_report(profile_report)

    # Save with step number in filename

    #with open(f"./profile_data/decoder_step_{step}.csv", "w") as f:
    #    f.write(profile_report)
    # === Decoder profiling ===



    
    # Normal execution

    out = decoder_vm["main"](*inputs)
    #print(f"\n[DEBUG] step {step} output nan check:")
    #for oi, item in enumerate(out):
    #    arr = item.numpy()
    #    print(
    #        "out", oi,
    #        "shape", arr.shape,
    #        "has_nan", np.isnan(arr).any(),
    #        "nan_count", np.isnan(arr).sum(),
    #        "min", np.nanmin(arr),
    #        "max", np.nanmax(arr),
    #    )
    end_time = datetime.now()


    logits = out[0].numpy()

    #print(f"\n[DEBUG] step {step} logits shape:", logits.shape)
    #print("[DEBUG] logits min/max:", np.nanmin(logits), np.nanmax(logits))
    #print("[DEBUG] logits has nan:", np.isnan(logits).any())
    #print("[DEBUG] logits nan count:", np.isnan(logits).sum())

    if np.isnan(logits).any():
        nan_ids = np.where(np.isnan(logits[0, -1]))[0]
        #print("[DEBUG] first nan token ids:", nan_ids[:20])
        #print("[DEBUG] first nan tokens:", [repr(tokenizer.decode([int(i)])) for i in nan_ids[:10]])

    # 暫時用 nan_to_num 避免 np.argmax 被 NaN 干擾
    safe_logits = np.nan_to_num(logits[0, -1], nan=-1e30)
    next_token = int(np.argmax(safe_logits))
    #top5 = np.argsort(safe_logits)[-5:][::-1]
    #print(f"[DEBUG] step {step} top5 tokens:")
    #for tid in top5:
    #    tid = int(tid)
    #    print(tid, repr(tokenizer.decode([tid])), float(safe_logits[tid]))
    #print("[DEBUG] step next_token:", next_token, repr(tokenizer.decode([next_token])))

    tokens.append(next_token)
    #print("[DEBUG] current text:", repr(tokenizer.decode(tokens, skip_special_tokens=True)))    
    # print(f"⬆️ Next token: {next_token} ({tokenizer.decode([next_token])})")

    if next_token == eos_token:
        print("🛑 遇到 <eos>，結束解碼")
        break

    # Update self-attention positions (index 0,1,4,5,8,9,12,13)
    for i, dst_idx in enumerate([0,1,4,5,8,9,12,13]):
        decoder_kvs[dst_idx] = out[i + 1]
    
    #print(f"[DEBUG] step {step} KV shapes after update:")
    #for idx in [0, 1, 4, 5, 8, 9, 12, 13]:
    #    print(idx, decoder_kvs[idx].shape)

print(f"End of decoder token generation: {end_time}")
print(f"Decoder token generation takes: {(end_time-start_time).total_seconds()}")

# === 最後輸出結果 ===
transcript = tokenizer.decode(tokens, skip_special_tokens=True)
print("\n📝 Transcription:\n", transcript)

"""Write aggregated results to a CSV file."""
# === profiling data aggregation === 
#with open("./profile_data/aggregation.csv", "w") as f:
#    f.write("Name,Total Duration (us),Total Count\n")
#    for name, (duration, count) in sorted(profile_agg.items(), key=lambda x: -x[1][0]):
#        f.write(f"{name},{duration},{count}\n")
# === profiling data aggregation === 
end_time_all = datetime.now()
print("End of all:", end_time_all)
print("All takes: ", (end_time_all-start_time_all).total_seconds())
