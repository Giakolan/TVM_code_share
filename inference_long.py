"""Transcribe long audio with reusable 15s/30s Whisper-tiny TVM VMs.

This first version uses independent, non-overlapping 30-second chunks.
Run on the Banana Pi from ~/whisper-tiny:

    python -u inference_long.py jfk.flac
    python -u inference_long.py a_long_recording.wav

The decoder VM and tokenizer stay loaded for all chunks. Decoder tokens and
KV cache are reset for each chunk because its Encoder output is different.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal
import tvm
from tvm import runtime
from tvm.relax import VirtualMachine
from transformers import WhisperProcessor


ROOT = Path(__file__).resolve().parent
SAMPLE_RATE = 16000
CHUNK_SECONDS = 30
START_TOKEN = 50258  # Same start token as the existing inference_auto.py.
SELF_KV_INDICES = (0, 1, 4, 5, 8, 9, 12, 13)
ENCODER_MODELS = {
    15: ROOT / "onnx/encoder_model_15s_conv_riscv.so",
    30: ROOT / "onnx/encoder_model_30s_conv_riscv.so",
}
PREFILL_MODEL = ROOT / "onnx/decoder_model_dynamic_riscv.so"
GENERATION_MODEL = ROOT / "onnx/decoder_with_past_model_dynamic_riscv.so"


def load_vm(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing TVM model: {path}")
    return VirtualMachine(runtime.load_module(str(path)), tvm.cpu(), profile=False)


def choose_bucket(audio_seconds):
    return 15 if audio_seconds <= 15 else 30


def read_audio(path):
    waveform, sample_rate = sf.read(path)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    elif waveform.ndim != 1:
        raise ValueError(f"Unsupported audio shape: {waveform.shape}")
    if not len(waveform):
        raise ValueError("Empty audio")
    if sample_rate != SAMPLE_RATE:
        count = int(len(waveform) * SAMPLE_RATE / sample_rate)
        waveform = signal.resample(waveform, count)
    return waveform


def decode_chunk(encoder_out, prefill_vm, generation_vm, tokenizer, max_tokens):
    # These variables must be new for every chunk. Keep the VM objects loaded.
    tokens = [START_TOKEN]
    input_ids = np.array([[START_TOKEN]], dtype="int64")
    t0 = time.perf_counter()
    result = prefill_vm["main"](tvm.runtime.tensor(input_ids), encoder_out)
    prefill_seconds = time.perf_counter() - t0

    next_token = int(np.argmax(result[0].numpy()[0, -1]))
    tokens.append(next_token)
    kv = list(result[1:])
    if len(kv) != 16:
        raise RuntimeError(f"Expected 16 decoder KV tensors; got {len(kv)}")

    generation_seconds = 0.0
    for _ in range(1, max_tokens):
        if next_token == tokenizer.eos_token_id:
            break
        input_ids = np.array([[tokens[-1]]], dtype="int64")
        inputs = [tvm.runtime.tensor(input_ids)] + kv
        t0 = time.perf_counter()
        result = generation_vm["main"](*inputs)
        generation_seconds += time.perf_counter() - t0

        # Match the existing inference script's NaN treatment.
        logits = np.nan_to_num(result[0].numpy()[0, -1], nan=-1e30)
        next_token = int(np.argmax(logits))
        tokens.append(next_token)
        if next_token == tokenizer.eos_token_id:
            break

        for i, dst_index in enumerate(SELF_KV_INDICES):
            kv[dst_index] = result[i + 1]

    if next_token != tokenizer.eos_token_id:
        raise RuntimeError(
            f"Reached --max-tokens={max_tokens} without <eos>. "
            "Text may be incomplete; increase the limit or use shorter chunks."
        )
    return tokenizer.decode(tokens, skip_special_tokens=True), prefill_seconds, generation_seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()
    if not 2 <= args.max_tokens <= 448:
        parser.error("--max-tokens must be between 2 and 448")

    all_start = time.perf_counter()
    print("Loading Processor and Decoder VMs ...", flush=True)
    processor = WhisperProcessor.from_pretrained(str(ROOT))
    tokenizer = processor.tokenizer
    prefill_vm = load_vm(PREFILL_MODEL)
    generation_vm = load_vm(GENERATION_MODEL)
    encoder_vms = {}  # Load a 15s or 30s Encoder only when first needed.
    print(f"Initialization: {time.perf_counter() - all_start:.3f} s", flush=True)

    waveform = read_audio(args.audio)
    total_samples = len(waveform)
    stride = CHUNK_SECONDS * SAMPLE_RATE
    print(f"Audio duration: {total_samples / SAMPLE_RATE:.3f} s", flush=True)

    transcriptions = []
    for index, start in enumerate(range(0, total_samples, stride), start=1):
        end = min(start + stride, total_samples)
        chunk = waveform[start:end]
        bucket = choose_bucket(len(chunk) / SAMPLE_RATE)
        chunk_start = time.perf_counter()
        print(
            f"\nChunk {index}: {start / SAMPLE_RATE:.2f}–{end / SAMPLE_RATE:.2f} s "
            f"(Encoder bucket: {bucket} s)",
            flush=True,
        )
        if bucket not in encoder_vms:
            encoder_vms[bucket] = load_vm(ENCODER_MODELS[bucket])
        vm = encoder_vms[bucket]

        inputs = processor(
            chunk,
            sampling_rate=SAMPLE_RATE,
            return_tensors="np",
            padding="max_length",
            max_length=bucket * SAMPLE_RATE,
            truncation=True,
        )
        mel = inputs.input_features.astype("float32")
        assert mel.shape == (1, 80, bucket * 100), mel.shape

        t0 = time.perf_counter()
        encoder_out = vm["main"](tvm.runtime.tensor(mel))
        encoder_seconds = time.perf_counter() - t0
        expected_shape = (1, bucket * 50, 384)
        if tuple(encoder_out.shape) != expected_shape:
            raise RuntimeError(
                f"Unexpected Encoder shape {tuple(encoder_out.shape)}; "
                f"expected {expected_shape}"
            )

        text, prefill_seconds, generation_seconds = decode_chunk(
            encoder_out, prefill_vm, generation_vm, tokenizer, args.max_tokens
        )
        transcriptions.append(text.strip())
        print(f"Chunk {index} text: {text}", flush=True)
        print(
            f"Chunk {index} times: encoder={encoder_seconds:.3f} s, "
            f"prefill={prefill_seconds:.3f} s, "
            f"generation VM={generation_seconds:.3f} s, "
            f"chunk total={time.perf_counter() - chunk_start:.3f} s",
            flush=True,
        )

    print("\nTranscription:\n" + " ".join(transcriptions), flush=True)
    print(f"Total elapsed: {time.perf_counter() - all_start:.3f} s", flush=True)


if __name__ == "__main__":
    main()
