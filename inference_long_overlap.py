"""Transcribe long audio with reusable 15s/30s Whisper-tiny TVM VMs.

Audio is processed in overlapping chunks.  The overlap helps Whisper retain
words that cross a 30-second boundary, while exact word-level de-duplication
removes repeated text when chunk transcriptions are joined.

If a chunk returns very few tokens or low confidence, it is automatically
retried with a window shifted one second earlier.  The accepted result is
chosen by validity and boundary continuity first, then by confidence.

Run on the Banana Pi from ~/whisper-tiny:

    python -u inference_long_overlap.py tedlium_65s.wav
    python -u inference_long_overlap.py tedlium_65s.wav --overlap-seconds 2

The decoder VMs and tokenizer stay loaded for all chunks.  Decoder tokens and
KV cache are reset for every chunk because each chunk has different Encoder
output.
"""

import argparse
import re
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
DEFAULT_OVERLAP_SECONDS = 2.0
DECODER_PROMPT_TOKENS = (
    "<|startoftranscript|>",
    "<|en|>",
    "<|transcribe|>",
    "<|notimestamps|>",
)
SELF_KV_INDICES = (0, 1, 4, 5, 8, 9, 12, 13)
ENCODER_MODELS = {
    15: ROOT / "onnx/encoder_model_15s_conv_riscv.so",
    30: ROOT / "onnx/encoder_model_30s_conv_riscv.so",
}
PREFILL_MODEL = ROOT / "onnx/decoder_model_dynamic_riscv.so"
GENERATION_MODEL = ROOT / "onnx/decoder_with_past_model_dynamic_riscv.so"
WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def load_vm(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing TVM model: {path}")
    return VirtualMachine(runtime.load_module(str(path)), tvm.cpu(), profile=False)


def choose_bucket(audio_seconds):
    return 15 if audio_seconds <= 15 else 30


def get_token_logprob(logits, token_id):
    """Return the selected token's numerically stable log-softmax value."""
    logits = np.asarray(logits, dtype=np.float64)
    logits = np.nan_to_num(logits, nan=-1e30, posinf=1e30, neginf=-1e30)
    maximum = np.max(logits)
    log_sum_exp = maximum + np.log(np.exp(logits - maximum).sum())
    return float(logits[token_id] - log_sum_exp)


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


def merge_transcription(
    previous,
    current,
    max_overlap_words=40,
    max_leading_noise_words=6,
):
    """Merge chunks by removing an exact shared word sequence at the boundary.

    Punctuation and letter case are ignored during comparison.  At least two
    matching words are required, which avoids deleting an ordinary repeated
    word such as "the" or "and".

    A few unmatched words are allowed at the beginning of ``current``.  This
    handles boundary errors such as "the carpet fast forward ..." versus the
    previous chunk's "rapid fast forward ..." without using unsafe fuzzy
    matching.  A skipped prefix requires at least four exactly matching words.

    Returns
    -------
    merged_text : str
        Combined transcription.
    removed_words : int
        Total number of words removed from the beginning of ``current``.
    matched_words : int
        Number of exact repeated words used to align the boundary.
    skipped_words : int
        Number of unmatched leading words discarded before the exact match.
    """
    previous = previous.strip()
    current = current.strip()
    if not previous:
        return current, 0, 0, 0
    if not current:
        return previous, 0, 0, 0

    previous_words = [m.group(0).lower() for m in WORD_PATTERN.finditer(previous)]
    current_matches = list(WORD_PATTERN.finditer(current))
    current_words = [m.group(0).lower() for m in current_matches]
    best_match = 0
    best_skip = 0
    max_skip = min(max_leading_noise_words, max(0, len(current_words) - 2))
    for skip in range(max_skip + 1):
        limit = min(
            max_overlap_words,
            len(previous_words),
            len(current_words) - skip,
        )
        minimum_match = 2 if skip == 0 else 4
        for count in range(limit, minimum_match - 1, -1):
            if previous_words[-count:] == current_words[skip : skip + count]:
                if count > best_match or (count == best_match and skip < best_skip):
                    best_match = count
                    best_skip = skip
                break

    removed_words = best_skip + best_match if best_match else 0

    if removed_words:
        if removed_words < len(current_matches):
            remainder = current[current_matches[removed_words].start():].lstrip()
        else:
            remainder = ""
    else:
        remainder = current

    if not remainder:
        return previous, removed_words, best_match, best_skip
    return (
        previous.rstrip() + " " + remainder,
        removed_words,
        best_match,
        best_skip,
    )


def decode_chunk(encoder_out, prefill_vm, generation_vm, tokenizer, max_tokens):
    # These variables must be new for every chunk. Keep only the VM objects loaded.
    prompt_ids = tokenizer.convert_tokens_to_ids(list(DECODER_PROMPT_TOKENS))
    if len(prompt_ids) != len(DECODER_PROMPT_TOKENS) or len(set(prompt_ids)) != len(
        DECODER_PROMPT_TOKENS
    ):
        raise RuntimeError(
            f"Failed to create Whisper decoder prompt: {prompt_ids}"
        )
    tokens = list(prompt_ids)
    input_ids = np.array([tokens], dtype="int64")
    t0 = time.perf_counter()
    result = prefill_vm["main"](tvm.runtime.tensor(input_ids), encoder_out)
    prefill_seconds = time.perf_counter() - t0

    first_logits = np.nan_to_num(result[0].numpy()[0, -1], nan=-1e30)
    next_token = int(np.argmax(first_logits))
    generated_token_ids = [next_token]
    generated_logprobs = [get_token_logprob(first_logits, next_token)]
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
        generated_token_ids.append(next_token)
        generated_logprobs.append(get_token_logprob(logits, next_token))
        tokens.append(next_token)
        if next_token == tokenizer.eos_token_id:
            break

        for i, dst_index in enumerate(SELF_KV_INDICES):
            kv[dst_index] = result[i + 1]

    if next_token != tokenizer.eos_token_id:
        partial_text = tokenizer.decode(tokens, skip_special_tokens=True)
        raise RuntimeError(
            f"Reached --max-tokens={max_tokens} without <eos>. "
            "The decoder may be repeating or hallucinating. "
            f"Last generated text: {partial_text[-500:]!r}"
        )
    scored_logprobs = [
        logprob
        for token_id, logprob in zip(generated_token_ids, generated_logprobs)
        if token_id != tokenizer.eos_token_id
    ]
    if scored_logprobs:
        average_logprob = float(np.mean(scored_logprobs))
        confidence = float(np.exp(average_logprob))
    else:
        average_logprob = float("-inf")
        confidence = 0.0

    return {
        "text": tokenizer.decode(tokens, skip_special_tokens=True),
        "prefill_seconds": prefill_seconds,
        "generation_seconds": generation_seconds,
        "token_ids": generated_token_ids,
        "token_logprobs": generated_logprobs,
        "scored_token_count": len(scored_logprobs),
        "average_logprob": average_logprob,
        "confidence": confidence,
    }


def transcribe_window(
    waveform,
    start,
    end,
    processor,
    encoder_vms,
    prefill_vm,
    generation_vm,
    tokenizer,
    max_tokens,
):
    """Run one audio window and return text, confidence, and timing data."""
    attempt_start = time.perf_counter()
    chunk = waveform[start:end]
    bucket = choose_bucket(len(chunk) / SAMPLE_RATE)
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

    result = decode_chunk(
        encoder_out, prefill_vm, generation_vm, tokenizer, max_tokens
    )
    result.update(
        {
            "start": start,
            "end": end,
            "bucket": bucket,
            "encoder_seconds": encoder_seconds,
            "attempt_seconds": time.perf_counter() - attempt_start,
        }
    )
    return result


def is_low_quality(result, confidence_threshold, min_tokens):
    return (
        result["confidence"] < confidence_threshold
        or result["scored_token_count"] < min_tokens
    )


def boundary_match(previous, current):
    _, removed, matched, skipped = merge_transcription(previous, current)
    return {
        "removed": removed,
        "matched": matched,
        "skipped": skipped,
    }


def candidate_rank(result, previous, confidence_threshold, min_tokens):
    boundary = boundary_match(previous, result["text"])
    valid = not is_low_quality(result, confidence_threshold, min_tokens)
    # Correct boundary continuity is preferred over confidence after validity.
    rank = (
        int(valid),
        boundary["matched"],
        -boundary["skipped"],
        result["confidence"],
        result["scored_token_count"],
    )
    return rank, boundary, valid


def print_attempt(label, result):
    print(
        f"{label}: {result['start'] / SAMPLE_RATE:.2f}–"
        f"{result['end'] / SAMPLE_RATE:.2f} s "
        f"(Encoder bucket: {result['bucket']} s)",
        flush=True,
    )
    print(f"{label} text: {result['text']}", flush=True)
    print(
        f"{label} confidence: average_logprob="
        f"{result['average_logprob']:.6f}, "
        f"geometric_mean_probability={result['confidence']:.6f}, "
        f"scored_tokens={result['scored_token_count']}",
        flush=True,
    )
    print(
        f"{label} times: encoder={result['encoder_seconds']:.3f} s, "
        f"prefill={result['prefill_seconds']:.3f} s, "
        f"generation VM={result['generation_seconds']:.3f} s, "
        f"attempt total={result['attempt_seconds']:.3f} s",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--overlap-seconds",
        type=float,
        default=DEFAULT_OVERLAP_SECONDS,
        help="Audio shared by adjacent chunks (default: 2.0 seconds)",
    )
    parser.add_argument(
        "--retry-confidence-threshold",
        type=float,
        default=0.30,
        help="Retry a chunk below this confidence (default: 0.30)",
    )
    parser.add_argument(
        "--retry-min-tokens",
        type=int,
        default=10,
        help="Retry a chunk with fewer scored tokens (default: 10)",
    )
    parser.add_argument(
        "--retry-shift-seconds",
        type=float,
        default=1.0,
        help="Shift a failed chunk earlier by this amount (default: 1.0)",
    )
    parser.add_argument(
        "--disable-auto-retry",
        action="store_true",
        help="Disable confidence-based shifted-window retry",
    )
    args = parser.parse_args()
    if not 2 <= args.max_tokens <= 448:
        parser.error("--max-tokens must be between 2 and 448")
    if not 0 <= args.overlap_seconds < CHUNK_SECONDS:
        parser.error(f"--overlap-seconds must be in [0, {CHUNK_SECONDS})")
    if not 0 <= args.retry_confidence_threshold <= 1:
        parser.error("--retry-confidence-threshold must be in [0, 1]")
    if args.retry_min_tokens < 0:
        parser.error("--retry-min-tokens must be non-negative")
    if not 0 < args.retry_shift_seconds < CHUNK_SECONDS:
        parser.error(f"--retry-shift-seconds must be in (0, {CHUNK_SECONDS})")

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
    chunk_samples = CHUNK_SECONDS * SAMPLE_RATE
    overlap_samples = int(round(args.overlap_seconds * SAMPLE_RATE))
    print(f"Audio duration: {total_samples / SAMPLE_RATE:.3f} s", flush=True)
    print(f"Chunk overlap: {args.overlap_seconds:.3f} s", flush=True)

    transcription = ""
    retry_count = 0
    retry_selected_count = 0
    index = 1
    start = 0
    while start < total_samples:
        end = min(start + chunk_samples, total_samples)
        chunk_start = time.perf_counter()
        print(f"\n===== Chunk {index} =====", flush=True)
        original = transcribe_window(
            waveform,
            start,
            end,
            processor,
            encoder_vms,
            prefill_vm,
            generation_vm,
            tokenizer,
            args.max_tokens,
        )
        print_attempt(f"Chunk {index} original", original)
        selected = original

        original_rank, original_boundary, original_valid = candidate_rank(
            original,
            transcription,
            args.retry_confidence_threshold,
            args.retry_min_tokens,
        )
        should_retry = (
            not args.disable_auto_retry
            and not original_valid
            and start > 0
        )
        if should_retry:
            shift_samples = int(round(args.retry_shift_seconds * SAMPLE_RATE))
            retry_start = max(0, start - shift_samples)
            if end == total_samples:
                retry_end = total_samples
                if retry_end - retry_start > chunk_samples:
                    retry_start = retry_end - chunk_samples
            else:
                retry_end = end - (start - retry_start)

            if retry_start != start and retry_end > retry_start:
                retry_count += 1
                print(
                    f"[AUTO-RETRY] Low-quality Chunk {index}: "
                    f"confidence={original['confidence']:.6f}, "
                    f"tokens={original['scored_token_count']}. "
                    f"Retrying {args.retry_shift_seconds:.2f} s earlier.",
                    flush=True,
                )
                retry = transcribe_window(
                    waveform,
                    retry_start,
                    retry_end,
                    processor,
                    encoder_vms,
                    prefill_vm,
                    generation_vm,
                    tokenizer,
                    args.max_tokens,
                )
                print_attempt(f"Chunk {index} retry", retry)
                retry_rank, retry_boundary, retry_valid = candidate_rank(
                    retry,
                    transcription,
                    args.retry_confidence_threshold,
                    args.retry_min_tokens,
                )
                if retry_rank > original_rank:
                    selected = retry
                    retry_selected_count += 1
                    print(
                        f"[AUTO-RETRY] Selected retry: valid={retry_valid}, "
                        f"boundary_matches={retry_boundary['matched']}, "
                        f"confidence={retry['confidence']:.6f}",
                        flush=True,
                    )
                else:
                    print(
                        f"[AUTO-RETRY] Kept original: valid={original_valid}, "
                        f"boundary_matches={original_boundary['matched']}, "
                        f"confidence={original['confidence']:.6f}",
                        flush=True,
                    )

        transcription, removed_words, matched_words, skipped_words = (
            merge_transcription(transcription, selected["text"])
        )
        print(f"Chunk {index} accepted text: {selected['text']}", flush=True)
        if index > 1:
            if matched_words:
                print(
                    f"Boundary merge: matched {matched_words} repeated word(s), "
                    f"skipped {skipped_words} noisy leading word(s), "
                    f"removed {removed_words} word(s) total",
                    flush=True,
                )
            else:
                print(
                    "Boundary merge: no exact repeated phrase found; kept all text",
                    flush=True,
                )
        print(
            f"Chunk {index} total including retries="
            f"{time.perf_counter() - chunk_start:.3f} s",
            flush=True,
        )

        accepted_end = selected["end"]
        if accepted_end == total_samples:
            break
        start = accepted_end - overlap_samples
        index += 1

    print("\nTranscription:\n" + transcription, flush=True)
    print(
        f"Auto-retry summary: attempted={retry_count}, "
        f"selected={retry_selected_count}",
        flush=True,
    )
    print(f"Total elapsed: {time.perf_counter() - all_start:.3f} s", flush=True)


if __name__ == "__main__":
    main()
