"""
Semi-Cascaded SLU on SLURP -- performance-optimized version
==============================================================

This is a drop-in replacement for the original training script with the
single-CPU-core bottleneck removed. Summary of what changed and why (see
inline comments tagged "# PERF:" for exact locations):

  1. ASR transcription (`ASRTranscriber.build_cache`) now decodes/resamples
     audio in a multi-process pool instead of a serial list comprehension,
     so CPU-side audio I/O overlaps with GPU-side Whisper decoding instead
     of blocking it.

  2. The frozen Whisper acoustic encoder is only ever run ONCE per
     recording now (`build_acoustic_feature_cache`), not once per training
     step per epoch. Since `freeze_whisper=True` means its output for a
     given waveform never changes, recomputing it every forward pass (the
     original behavior) wasted a full Whisper-encoder forward pass on
     every single step of every epoch. Features are cached to disk as
     fp16 tensors and just loaded (not recomputed) at train time.

  3. The word-level Levenshtein tag alignment (`_align_and_project_tags`)
     is a deterministic, pure-Python, CPU-bound function of
     (gt_words, gt_tags, asr_text). The original code re-ran it inside
     `collate_fn` on every batch of every epoch. It's now computed once
     per recording, in parallel across CPU cores, and cached to disk.

  4. DataLoaders use `persistent_workers=True` + `pin_memory=True` so
     worker processes aren't torn down/rebuilt every epoch and host->GPU
     copies are faster. Worker cost is now trivial (just loading a small
     cached tensor + tokenizing text) since (1)-(3) moved the expensive
     work out of the training loop entirely.

  5. `WhisperForConditionalGeneration.generate(...)` now explicitly pins
     `num_beams=1` (greedy) for the ASR pass. Some Whisper checkpoints'
     `generation_config` defaults to beam search, which multiplies ASR
     decode time by the beam width for no accuracy benefit in this
     use case (we only need a plausible noisy hypothesis, not the best
     possible transcript).

Net effect: audio decode + Whisper-encoder + alignment work that used to
happen O(epochs) times now happens O(1) times, in parallel, before
training starts. Training itself becomes GPU-bound instead of
single-core-CPU-bound.

Everything else (model architecture, losses, metrics) is unchanged from
the original script.
"""

import hashlib
import json
import os
import random
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Must run before any `transformers`/`huggingface_hub` import.
from hf_offline_utils import enable_offline_mode
enable_offline_mode()

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    WhisperModel,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    AutoModel,
    AutoTokenizer,
    get_scheduler,
)


# =========================================================================== #
# 0. Config
# =========================================================================== #
@dataclass
class SLURPConfig:
    root_dir: str = "slurp_dataset"
    train_jsonl: str = "dataset/slurp/train.jsonl"
    valid_jsonl: str = "dataset/slurp/devel.jsonl"
    test_jsonl: str = "dataset/slurp/test.jsonl"
    train_synthetic_jsonl: str = "dataset/slurp/train_synthetic.jsonl"
    include_synthetic: bool = False

    audio_real_subdir: str = "slurp_real"
    audio_synth_subdir: str = "slurp_synth"

    recording_filter: str = "correct_only"   # "correct_only" | "all"
    recording_type_filter: Optional[str] = None

    sample_rate: int = 16000
    max_audio_seconds: float = 11.0

    whisper_model_name: str = "openai/whisper-small"
    bert_model_name: str = "bert-base-uncased"
    whisper_layer: int = 6
    # PERF/robustness: how much of the pretrained Whisper encoder to fine-tune. See
    # WhisperAcousticEncoder docstring. "frozen" keeps the original fast-caching behavior;
    # "last_n_layers"/"full"/"adapters" let the encoder adapt to domains (e.g. severe
    # noise/low SNR) its pretraining features may not represent well, at the cost of
    # losing acoustic feature caching (the encoder's output is no longer fixed once part
    # of it trains). "adapters" is usually the safest first thing to try: it never touches
    # the pretrained weights (so it can't forget the clean-speech representation) and has
    # far fewer trainable params than "last_n_layers".
    whisper_finetune_mode: str = "frozen"    # "frozen" | "last_n_layers" | "full" | "adapters" | "joint"
    whisper_unfreeze_last_n_layers: int = 2  # only used when whisper_finetune_mode == "last_n_layers"
    adapter_bottleneck_dim: int = 64         # only used when whisper_finetune_mode == "adapters"
    adapter_dropout: float = 0.1             # only used when whisper_finetune_mode == "adapters"

    # JOINT-ASR fine-tuning ("joint" mode): the encoder is trained with THREE simultaneous
    # gradient signals (intent loss, slot loss, and a real ASR seq2seq loss against the
    # ground-truth SLURP sentence), instead of only the two SLU losses. The idea: as the
    # encoder adapts to noisy audio to reduce ASR loss, the SAME adapted representation
    # feeds the SLU heads, so intent/slot should benefit too -- not just ASR transcription
    # quality. See JointWhisperASR for the implementation.
    #
    # `joint_encoder_finetune_mode` reuses the same sub-policy vocabulary as
    # whisper_finetune_mode ("frozen"|"last_n_layers"|"full"|"adapters") but applies it to
    # the encoder INSIDE the joint model -- "adapters" is the recommended safe default.
    joint_encoder_finetune_mode: str = "adapters"
    # Whether the Whisper DECODER's own weights are trainable. Left False by default: fine-
    # tuning the decoder on a comparatively small, narrow-domain dataset risks catastrophically
    # forgetting Whisper's general language modeling ability. With this False, the ASR loss
    # still backprops into the ENCODER (that's the whole point -- see JointWhisperASR.forward),
    # it just doesn't update decoder weights. Try True only if encoder-only adaptation plateaus.
    asr_decoder_trainable: bool = False
    # Weight of the ASR seq2seq loss in the joint objective:
    #   total_loss = intent_loss + slot_loss_weight * slot_loss + asr_loss_weight * asr_loss
    asr_loss_weight: float = 1.0
    # ASR hypotheses (feeding the BERT text branch) and the slot-tag alignment cache are
    # necessarily STALE relative to the live, adapting encoder/decoder weights -- refreshing
    # them requires an autoregressive generate() pass, which is far more expensive than a
    # teacher-forced forward pass, so it's done periodically rather than every step.
    # Refreshing every epoch is the default; raise this to refresh less often and save time,
    # at the cost of the text branch / slot labels lagging further behind the adapting model.
    asr_refresh_every_n_epochs: int = 1
    # Batch size used only during the periodic generate() refresh pass (independent of the
    # training batch_size below, since generate() is far more memory-hungry per-sample than
    # a teacher-forced forward pass).
    asr_refresh_batch_size: int = 16

    # Which cross-attention architecture to use. See CrossAttentionBlock / SemiCascadedSLU_SLURP:
    #   "text_query"   -- baseline (original script): text tokens attend into acoustic frames.
    #                     Fused sequence is word-level; feeds both the slot head and (pooled) intent head.
    #   "audio_query"  -- full swap: acoustic frames attend into text. Fused sequence is frame-level;
    #                     feeds the (pooled) intent head directly. A secondary text-query block then
    #                     attends into that frame-level representation to recover word-level features
    #                     for the slot head (frame-level output alone has no notion of "one tag per word").
    #   "dual_branch"  -- both blocks in parallel: text-query branch feeds the slot head (same as
    #                     "text_query"), audio-query branch feeds the intent head (same pooling as
    #                     "audio_query"). No secondary projection needed since each branch only
    #                     serves the head suited to its native granularity.
    fusion_mode: str = "text_query"
    comparison_modes: Tuple[str, ...] = ("text_query", "audio_query", "dual_branch")

    asr_cache_path: str = "slurp_asr_cache.json"
    use_ground_truth_transcript: bool = False

    # Which model produces the CACHED ASR TEXT HYPOTHESES that feed the BERT text branch
    # (and, downstream, the slot-tag alignment cache). This is INDEPENDENT of
    # whisper_finetune_mode, which controls the separate, TRAINABLE acoustic-embedding path
    # (WhisperAcousticEncoder / JointWhisperASR) -- the two roles don't have to use the same
    # model. "funasr_nano" uses Fun-ASR-Nano-2512 (FunAudioLLM/Fun-ASR-Nano-2512 via the
    # `funasr` package), which is INFERENCE-ONLY: it has no documented training/fine-tuning
    # API (its own model card TODO list still has "Support model training" unchecked), so it
    # can only ever serve as a fixed, frozen ASR engine here -- never as a trainable encoder.
    # Its published benchmarks show it substantially more noise-robust than Whisper (e.g.
    # far-field/complex-background WER roughly 3-4x lower), which is why it's offered as a
    # stronger source of TEXT-BRANCH hypotheses specifically for noisy/low-SNR data, while
    # Whisper (still trainable) continues to supply the acoustic embeddings.
    asr_engine: str = "whisper"   # "whisper" | "funasr_nano"
    funasr_model_name: str = "FunAudioLLM/Fun-ASR-Nano-2512"
    funasr_hub: str = "hf"        # "hf" (Hugging Face) | "ms" (ModelScope)
    # NOTE: Fun-ASR-Nano-2512 expects one of its documented language NAMES, not an ISO code --
    # "英文" (English), "中文" (Chinese), "日文" (Japanese). Passing "en" is NOT valid for this
    # model. See FunASRTranscriber docstring.
    funasr_language: str = "英文"
    # Inverse text normalisation (digit/punctuation formatting, e.g. "five" -> "5"). SLURP's
    # ground-truth sentences are NOT ITN-normalised, so leaving this True can inflate WER /
    # slot-alignment mismatches purely from formatting rather than genuine transcription
    # errors -- see FunASRTranscriber docstring for guidance on when to set this False instead.
    funasr_itn: bool = True
    funasr_device: str = "cuda:0"

    # PERF: caching knobs
    use_precomputed_acoustic_features: bool = True
    acoustic_feature_cache_dir: str = "slurp_acoustic_feature_cache"
    slot_tag_cache_path: str = "slurp_slot_tag_cache.json"
    prep_num_workers: int = max(os.cpu_count() or 4, 1)
    prep_batch_size: int = 32                # batch size used only during the one-time caching passes

    max_text_len: int = 64
    batch_size: int = 8
    num_workers: int = 4                     # DataLoader workers during actual training
    lr: float = 3e-5
    epochs: int = 10
    slot_loss_weight: float = 1.0

    # LR schedule + gradient accumulation. A flat LR with no warmup tends to underfit a
    # freshly-initialized cross-attention fusion module stacked on pretrained BERT/Whisper --
    # warmup lets the new params find a reasonable region before the full LR kicks in, and
    # decay avoids late-training instability. grad_accum_steps lets you reach a larger
    # effective batch size (batch_size * grad_accum_steps) without more GPU memory, which
    # tends to stabilize joint fine-tuning of a BERT-scale model.
    warmup_ratio: float = 0.06               # fraction of total optimizer steps spent warming up
    lr_scheduler_type: str = "linear"        # any type accepted by transformers.get_scheduler ("linear","cosine",...)
    grad_accum_steps: int = 4                # effective batch size = batch_size * grad_accum_steps

    # min_total_optimizer_steps guards against silently undertraining when grad_accum_steps
    # is increased: since the optimizer only steps once every grad_accum_steps batches, raising
    # it (for a larger effective batch) reduces the number of weight UPDATES per epoch by the
    # same factor unless epochs is also scaled up. If epochs * steps_per_epoch(after accum)
    # would fall below this floor, train_model automatically increases the number of epochs
    # actually run (and logs a clear message explaining why) rather than quietly training with
    # fewer updates than intended. Set to None to disable and use cfg.epochs exactly as given.
    min_total_optimizer_steps: Optional[int] = 2000

    # Class-weighted losses: SLURP has ~60 intents and ~55 slot types with a long tail: a few
    # very common intents/tags and many rare ones. Unweighted cross-entropy lets the model get
    # away with mostly predicting frequent classes, which suppresses macro-averaged metrics
    # (and, for slots, the many-O-vs-rare-entity imbalance specifically). Inverse-frequency
    # weights (computed from the training split, capped to avoid a handful of ultra-rare
    # classes dominating the loss) counteract this.
    use_class_weights: bool = True
    class_weight_cap: float = 10.0

    # Focal loss (Lin et al., 2017): complements/replaces class weights. Static weights fix
    # how much each class COUNTS in the loss; focal loss additionally down-weights examples
    # the model is already confident about (whatever their class), keeping gradient signal
    # focused on hard/misclassified cases throughout training -- often more effective than
    # class weights alone for long-tailed label sets with many near-zero-F1 rare classes.
    loss_type: str = "cross_entropy"         # "cross_entropy" | "focal"
    focal_gamma: float = 2.0                 # higher = more aggressive down-weighting of easy examples

    # Diagnostics: print the worst-performing intents/slot-types by F1 after the final test
    # evaluation, to see directly whether a low macro-F1/slot_f1 is driven by a handful of
    # near-zero-F1 rare classes rather than broadly poor performance.
    print_per_class_report: bool = True
    per_class_report_top_n: int = 15

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================================== #
# 1. jsonl reading + sentence_annotation parsing
# =========================================================================== #
def _read_jsonl(path: str) -> List[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


_ENTITY_PATTERN = re.compile(r"\[\s*([a-zA-Z_]+)\s*:\s*([^\]]+?)\s*\]")


def parse_sentence_annotation(sentence_annotation: str) -> Tuple[List[str], List[str]]:
    words, tags = [], []
    pos = 0
    for m in _ENTITY_PATTERN.finditer(sentence_annotation):
        before = sentence_annotation[pos:m.start()]
        for w in before.split():
            words.append(w)
            tags.append("O")

        ent_type = m.group(1).strip()
        filler_words = m.group(2).strip().split()
        for i, w in enumerate(filler_words):
            tags.append(f"B-{ent_type}" if i == 0 else f"I-{ent_type}")
            words.append(w)

        pos = m.end()

    after = sentence_annotation[pos:]
    for w in after.split():
        words.append(w)
        tags.append("O")

    return words, tags


# =========================================================================== #
# 2. Word-level Levenshtein alignment
# =========================================================================== #
def _align_and_project_tags(gt_words: List[str], gt_tags: List[str], hyp_words: List[str]) -> List[str]:
    n, m = len(gt_words), len(hyp_words)
    if m == 0:
        return []

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if gt_words[i - 1].lower() == hyp_words[j - 1].lower() else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    i, j = n, m
    hyp_tags = ["O"] * m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (
            0 if gt_words[i - 1].lower() == hyp_words[j - 1].lower() else 1
        ):
            hyp_tags[j - 1] = gt_tags[i - 1]
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return hyp_tags


# PERF: top-level, picklable worker for ProcessPoolExecutor -- computes one
# example's alignment. Kept separate from `_align_and_project_tags` so it
# can be mapped in parallel over the whole dataset once, instead of being
# called serially inside collate_fn on every batch of every epoch.
def _align_worker(args: Tuple[List[str], List[str], List[str]]) -> List[str]:
    gt_words, gt_tags, hyp_words = args
    return _align_and_project_tags(gt_words, gt_tags, hyp_words)


# =========================================================================== #
# 3. Label vocab
# =========================================================================== #
class SLURPLabelVocab:
    def __init__(self, jsonl_paths: List[str]):
        intents = set()
        slot_types = set()
        for p in jsonl_paths:
            for entry in _read_jsonl(p):
                intents.add(entry["intent"])
                _, tags = parse_sentence_annotation(entry.get("sentence_annotation", entry["sentence"]))
                for t in tags:
                    if t != "O":
                        slot_types.add(t[2:])

        self.intent2id = {v: i for i, v in enumerate(sorted(intents))}
        self.id2intent = {i: v for v, i in self.intent2id.items()}

        slot_labels = ["O"] + sorted(f"{prefix}-{t}" for t in sorted(slot_types) for prefix in ("B", "I"))
        self.slot2id = {v: i for i, v in enumerate(slot_labels)}
        self.id2slot = {i: v for v, i in self.slot2id.items()}

    @property
    def num_intents(self) -> int:
        return len(self.intent2id)

    @property
    def num_slot_labels(self) -> int:
        return len(self.slot2id)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({"intent2id": self.intent2id, "slot2id": self.slot2id}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SLURPLabelVocab":
        with open(path) as f:
            d = json.load(f)
        obj = cls.__new__(cls)
        obj.intent2id = d["intent2id"]
        obj.slot2id = d["slot2id"]
        obj.id2intent = {v: k for k, v in obj.intent2id.items()}
        obj.id2slot = {v: k for k, v in obj.slot2id.items()}
        return obj


# =========================================================================== #
# 4. Audio I/O
# =========================================================================== #
def load_and_resample(wav_path: str, target_sr: int):
    import numpy as np
    waveform, sr = torchaudio.load(wav_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform.squeeze(0).numpy()


def _load_and_resample_safe(wav_path: str, target_sr: int):
    """Same as load_and_resample but never raises -- returns None on any decode
    failure (truncated/corrupt file, unsupported codec, etc.) instead of killing
    the calling process. Some SLURP recordings are known to be malformed; a single
    bad file must not be able to take down an entire ProcessPoolExecutor batch or
    a DataLoader worker."""
    try:
        return load_and_resample(wav_path, target_sr)
    except Exception as e:
        print(f"[warn] failed to decode audio, will be excluded: {wav_path} ({e})")
        return None


def _cache_key(rel_path: str) -> str:
    """Filesystem-safe cache filename for a wav_rel_path."""
    return hashlib.md5(rel_path.encode("utf-8")).hexdigest() + ".pt"


def validate_audio_files(
    wav_rel_paths: List[str],
    root_dir: str,
    sample_rate: int,
    cache_path: str,
    num_workers: int = 4,
) -> set:
    """Tries decoding every recording once, in parallel, and caches the
    good/bad verdict to disk (`cache_path`) so future runs don't have to
    re-decode everything just to re-check validity. Returns the set of
    wav_rel_paths that FAILED to decode -- callers should exclude these
    from every split, every cache-building pass, and the Dataset itself,
    so a corrupt file is only ever discovered once instead of crashing
    training (or a caching pass) every time it's encountered."""
    from itertools import repeat

    known: Dict[str, bool] = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            known = json.load(f)

    todo = [p for p in wav_rel_paths if p not in known]
    print(f"[audio-validate] {len(todo)} / {len(wav_rel_paths)} recordings not yet validated")
    if todo:
        full_paths = [os.path.join(root_dir, p) for p in todo]
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(_load_and_resample_safe, full_paths, repeat(sample_rate), chunksize=32))
        for p, wav in zip(todo, results):
            known[p] = wav is not None
        with open(cache_path, "w") as f:
            json.dump(known, f)

    bad = {p for p in wav_rel_paths if not known.get(p, True)}
    if bad:
        print(f"[audio-validate] {len(bad)} recordings failed to decode and will be EXCLUDED "
              f"from all splits (see warnings above for which files / errors).")
    return bad


# =========================================================================== #
# 5. ASR transcript caching (parallel audio decode + GPU batched decode)
# =========================================================================== #
class ASRTranscriber:
    def __init__(self, model_name: str, device: str, language: str = "en"):
        self.processor = WhisperProcessor.from_pretrained(model_name, local_files_only=True)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name, local_files_only=True).to(device)
        self.model.eval()
        self.model.generation_config.forced_decoder_ids = None
        self.device = device
        self.language = language

    @torch.no_grad()
    def transcribe_batch(self, waveforms: List, sample_rate: int) -> List[str]:
        inputs = self.processor(
            waveforms, sampling_rate=sample_rate, return_tensors="pt", return_attention_mask=True
        )
        input_features = inputs.input_features.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        generated_ids = self.model.generate(
            input_features,
            attention_mask=attention_mask,
            language=self.language,
            task="transcribe",
            num_beams=1,   # PERF: greedy decode -- some checkpoints default to beam search in
                            # generation_config, which multiplies ASR decode time for no benefit
                            # here (we only need a plausible noisy hypothesis, not the best one).
        )
        return [t.strip() for t in self.processor.batch_decode(generated_ids, skip_special_tokens=True)]

    def build_cache(self, wav_paths: List[str], root_dir: str, sample_rate: int, cache_path: str,
                     batch_size: int = 16, num_workers: int = 4) -> Dict[str, str]:
        cache = {}
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                cache = json.load(f)

        todo = [p for p in wav_paths if p not in cache]
        print(f"[ASRTranscriber] {len(todo)} / {len(wav_paths)} utterances need decoding "
              f"(using {num_workers} parallel decode workers)")
        if not todo:
            return cache

        # PERF: submit all audio-decode jobs to a process pool up front. Because futures
        # are submitted before we start consuming them, the pool keeps decoding batch i+1,
        # i+2, ... on other cores while the GPU is busy running Whisper on batch i -- audio
        # I/O and GPU decode overlap instead of serializing on one core.
        # Uses the "safe" loader (returns None instead of raising) as defense-in-depth --
        # bad files should already be filtered out by validate_audio_files() before this
        # is called, but a single corrupt file must never be able to kill the whole pass.
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            full_paths = [os.path.join(root_dir, p) for p in todo]
            futures = [executor.submit(_load_and_resample_safe, fp, sample_rate) for fp in full_paths]

            for i in range(0, len(todo), batch_size):
                batch_paths = todo[i:i + batch_size]
                batch_futures = futures[i:i + batch_size]
                results = [f.result() for f in batch_futures]
                ok_paths = [p for p, w in zip(batch_paths, results) if w is not None]
                waveforms = [w for w in results if w is not None]
                if not waveforms:
                    continue
                hyps = self.transcribe_batch(waveforms, sample_rate)
                for p, h in zip(ok_paths, hyps):
                    cache[p] = h
                if (i // batch_size) % 20 == 0:
                    with open(cache_path, "w") as f:
                        json.dump(cache, f)
                    print(f"  ...{i + len(batch_paths)}/{len(todo)} decoded")

        with open(cache_path, "w") as f:
            json.dump(cache, f)
        return cache


class FunASRTranscriber:
    """Inference-only ASR hypothesis generator using Fun-ASR-Nano-2512 (FunAudioLLM/Fun-ASR-
    Nano-2512), an LLM-based ASR model (SenseVoice encoder + Qwen3-0.6B decoder) accessed via
    the `funasr` package's AutoModel.generate() -- NOT `transformers`. Its API and internals
    are unrelated to Whisper/WhisperForConditionalGeneration.

    IMPORTANT LIMITATIONS (please read before relying on this class):

    1. NO TRAINING SUPPORT. As of this writing, Fun-ASR-Nano-2512's own model card lists
       "Support model training" as an open TODO item -- there is no documented teacher-forcing
       / `labels=` loss interface, and no standard way to pull intermediate encoder hidden
       states the way WhisperModel.encoder(..., output_hidden_states=True) does. This class is
       therefore build_cache()-only: it produces a static, FROZEN {wav_rel_path: text} lookup,
       exactly like ASRTranscriber, and is used ONLY to populate `asr_cache` -- it cannot serve
       as (and is not intended to replace) WhisperAcousticEncoder / JointWhisperASR as the
       TRAINABLE acoustic-embedding path feeding the SLU heads.

    2. generate() takes FILE PATHS, not pre-decoded waveform arrays. Unlike ASRTranscriber
       (which reuses this pipeline's own parallel-decoded numpy waveforms), FunASR handles its
       own audio loading internally, so this class does its own file-path batching rather than
       consuming _load_and_resample_safe() output. Each batch call is wrapped in a try/except
       so a single corrupt/unsupported file can't take down the whole caching pass (mirroring
       the defensive philosophy of validate_audio_files() elsewhere in this file) -- on
       failure the WHOLE batch falls back to per-file retries, and any file that still fails is
       skipped and logged, never crashing the run.

    3. `language` takes one of Fun-ASR-Nano-2512's documented language NAMES (e.g. "英文" for
       English, "中文" for Chinese, "日文" for Japanese) -- NOT an ISO code like "en". Passing
       "en" is not a documented value for this checkpoint.

    4. ITN (inverse text normalisation, e.g. "five" -> "5", added punctuation/casing) is on by
       default in FunASR and may not match SLURP's raw, non-ITN'd ground-truth sentence
       formatting. A formatting mismatch here would inflate WER / corrupt slot-tag alignment
       for reasons that have nothing to do with actual transcription quality. If you see
       unexpectedly high WER after switching engines, try `funasr_itn=False` first and/or add
       a light text-normalisation step (lowercase, strip punctuation) before WER/alignment.
    """

    def __init__(self, model_name: str, hub: str = "hf", device: str = "cuda:0",
                 language: str = "英文", itn: bool = True):
        try:
            from funasr import AutoModel as _FunASRAutoModel
        except ImportError as e:
            raise ImportError(
                "asr_engine='funasr_nano' requires the `funasr` package. Install with: "
                "pip install -U funasr"
            ) from e
        self.model = _FunASRAutoModel(
            model=model_name, hub=hub, trust_remote_code=True, device=device,
        )
        self.model_name = model_name
        self.language = language
        self.itn = itn

    def _generate_batch(self, full_paths: List[str]) -> List[Optional[str]]:
        """Returns one hypothesis string per path, or None for any path that failed. Falls
        back to one-at-a-time retries if the whole-batch call raises, so a single bad file
        doesn't discard an entire batch's results."""
        try:
            res = self.model.generate(
                input=full_paths, cache={}, batch_size=len(full_paths),
                language=self.language, itn=self.itn,
            )
            return [r.get("text", "").strip() if r else None for r in res]
        except Exception as e:
            print(f"[warn] [FunASRTranscriber] batch of {len(full_paths)} failed ({e}); "
                  f"retrying one-by-one...")
            out = []
            for p in full_paths:
                try:
                    r = self.model.generate(input=[p], cache={}, batch_size=1,
                                             language=self.language, itn=self.itn)
                    out.append(r[0].get("text", "").strip() if r else None)
                except Exception as e2:
                    print(f"[warn] [FunASRTranscriber] failed to transcribe, will be excluded: {p} ({e2})")
                    out.append(None)
            return out

    def build_cache(self, wav_paths: List[str], root_dir: str, cache_path: str,
                     batch_size: int = 16) -> Dict[str, str]:
        """Same on-disk caching contract as ASRTranscriber.build_cache() (skip already-cached
        paths, checkpoint the cache file periodically) so it's a drop-in replacement at the
        call site in build_dataloaders()."""
        cache = {}
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                cache = json.load(f)

        todo = [p for p in wav_paths if p not in cache]
        print(f"[FunASRTranscriber] {len(todo)} / {len(wav_paths)} utterances need decoding "
              f"(model={self.model_name}, language={self.language}, itn={self.itn})")
        if not todo:
            return cache

        for i in range(0, len(todo), batch_size):
            batch_paths = todo[i:i + batch_size]
            full_paths = [os.path.join(root_dir, p) for p in batch_paths]
            hyps = self._generate_batch(full_paths)
            for p, h in zip(batch_paths, hyps):
                if h is not None:
                    cache[p] = h
            if (i // batch_size) % 20 == 0:
                with open(cache_path, "w") as f:
                    json.dump(cache, f)
                print(f"  ...{i + len(batch_paths)}/{len(todo)} decoded")

        with open(cache_path, "w") as f:
            json.dump(cache, f)
        return cache


# =========================================================================== #
# 6. Acoustic feature caching (PERF: run the frozen Whisper encoder ONCE
#    per recording, not once per training step per epoch)
# =========================================================================== #
def build_acoustic_feature_cache(
    encoder: "WhisperAcousticEncoder",
    feature_extractor: WhisperFeatureExtractor,
    wav_paths: List[str],
    root_dir: str,
    sample_rate: int,
    cache_dir: str,
    device: str,
    batch_size: int = 32,
    num_workers: int = 4,
) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    todo = [p for p in wav_paths if not os.path.exists(os.path.join(cache_dir, _cache_key(p)))]
    print(f"[acoustic-cache] {len(todo)} / {len(wav_paths)} recordings need frozen-encoder features")
    if not todo:
        return

    encoder.eval()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        full_paths = [os.path.join(root_dir, p) for p in todo]
        futures = [executor.submit(_load_and_resample_safe, fp, sample_rate) for fp in full_paths]

        for i in range(0, len(todo), batch_size):
            batch_paths = todo[i:i + batch_size]
            batch_futures = futures[i:i + batch_size]
            results = [f.result() for f in batch_futures]
            ok_paths = [p for p, w in zip(batch_paths, results) if w is not None]
            waveforms = [w for w in results if w is not None]
            if not waveforms:
                continue

            inputs = feature_extractor(waveforms, sampling_rate=sample_rate, return_tensors="pt")
            input_features = inputs.input_features.to(device)

            with torch.no_grad():
                hidden = encoder(input_features)   # [B, T, H], fixed T since Whisper pads to 30s internally

            hidden = hidden.half().cpu()
            for j, p in enumerate(ok_paths):
                torch.save(hidden[j], os.path.join(cache_dir, _cache_key(p)))

            if (i // batch_size) % 20 == 0:
                print(f"  ...{i + len(batch_paths)}/{len(todo)} features cached")


# =========================================================================== #
# 7. Slot-tag alignment caching (PERF: computed once, in parallel, instead
#    of being recomputed inside collate_fn every batch of every epoch)
# =========================================================================== #
def build_slot_tag_cache(
    examples: List[dict],           # each has wav_rel_path, words, tags, asr_text
    cache_path: str,
    num_workers: int = 4,
) -> Dict[str, List[str]]:
    cache: Dict[str, List[str]] = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)

    todo = [ex for ex in examples if ex["wav_rel_path"] not in cache]
    print(f"[slot-tag-cache] {len(todo)} / {len(examples)} recordings need tag alignment")
    if todo:
        args = [(ex["words"], ex["tags"], ex["asr_text"].split() or ["<empty>"]) for ex in todo]
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(_align_worker, args, chunksize=64))
        for ex, tags in zip(todo, results):
            cache[ex["wav_rel_path"]] = tags
        with open(cache_path, "w") as f:
            json.dump(cache, f)
    return cache


# =========================================================================== #
# 8. Dataset
# =========================================================================== #
class SLURPDataset(Dataset):
    def __init__(
        self,
        jsonl_paths: List[Tuple[str, str]],
        root_dir: str,
        vocab: SLURPLabelVocab,
        sample_rate: int = 16000,
        max_audio_seconds: float = 11.0,
        asr_cache: Optional[Dict[str, str]] = None,
        slot_tag_cache: Optional[Dict[str, List[str]]] = None,
        use_ground_truth_transcript: bool = False,
        recording_filter: str = "correct_only",
        recording_type_filter: Optional[str] = None,
        use_precomputed_acoustic_features: bool = True,
        acoustic_feature_cache_dir: Optional[str] = None,
        bad_files: Optional[set] = None,
    ):
        self.root_dir = root_dir
        self.vocab = vocab
        self.sample_rate = sample_rate
        self.max_samples = int(max_audio_seconds * sample_rate)
        self.asr_cache = asr_cache or {}
        self.slot_tag_cache = slot_tag_cache or {}
        self.use_ground_truth_transcript = use_ground_truth_transcript
        self.use_precomputed_acoustic_features = use_precomputed_acoustic_features
        self.acoustic_feature_cache_dir = acoustic_feature_cache_dir
        self.bad_files = bad_files or set()

        self.examples = []
        n_skipped_bad = 0
        for jsonl_path, audio_subdir in jsonl_paths:
            for entry in _read_jsonl(jsonl_path):
                if entry["intent"] not in vocab.intent2id:
                    continue
                words, tags = parse_sentence_annotation(entry.get("sentence_annotation", entry["sentence"]))

                for rec in entry.get("recordings", []):
                    if recording_filter == "correct_only" and rec.get("status") != "correct":
                        continue
                    fname = rec["file"]
                    if recording_type_filter == "headset" and "-headset" not in fname:
                        continue
                    if recording_type_filter == "non_headset" and "-headset" in fname:
                        continue
                    wav_rel_path = os.path.join(audio_subdir, fname)
                    if wav_rel_path in self.bad_files:
                        # corrupt/undecodable file, already flagged by validate_audio_files()
                        n_skipped_bad += 1
                        continue

                    self.examples.append({
                        "wav_rel_path": wav_rel_path,
                        "words": words,
                        "tags": tags,
                        "sentence": entry["sentence"],
                        "intent_id": vocab.intent2id[entry["intent"]],
                    })

        if n_skipped_bad:
            print(f"[SLURPDataset] skipped {n_skipped_bad} recordings flagged as undecodable by validate_audio_files()")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        ex = self.examples[idx]
        wav_rel_path = ex["wav_rel_path"]

        asr_text = ex["sentence"] if self.use_ground_truth_transcript else \
            self.asr_cache.get(wav_rel_path, ex["sentence"])

        if self.use_ground_truth_transcript:
            hyp_words, hyp_tags = ex["words"], ex["tags"]
        else:
            hyp_words = asr_text.split() or ["<empty>"]
            # PERF: pull the precomputed alignment instead of recomputing it here/in collate_fn.
            hyp_tags = self.slot_tag_cache.get(wav_rel_path)
            if hyp_tags is None:
                # fallback (e.g. cache not built yet) -- computed lazily, still cheap per-item
                hyp_tags = _align_and_project_tags(ex["words"], ex["tags"], hyp_words)

        item = {
            "wav_rel_path": wav_rel_path,
            "words": hyp_words,
            "tags": hyp_tags,
            "intent_id": ex["intent_id"],
            "gt_transcript": ex["sentence"],
        }

        if self.use_precomputed_acoustic_features and self.acoustic_feature_cache_dir is not None:
            feat_path = os.path.join(self.acoustic_feature_cache_dir, _cache_key(wav_rel_path))
            if os.path.exists(feat_path):
                # PERF: load a small cached tensor instead of decoding audio + running Whisper encoder.
                item["acoustic_features"] = torch.load(feat_path).float()
                return item
            # fall through to raw audio if this recording wasn't cached for any reason

        # Defensive fallback (should be rare -- bad_files filtering in __init__ is meant to
        # catch this ahead of time): never let a single corrupt file raise inside a DataLoader
        # worker and take down the whole training/eval loop. Substitute silence instead and warn.
        waveform = _load_and_resample_safe(os.path.join(self.root_dir, wav_rel_path), self.sample_rate)
        if waveform is None:
            print(f"[warn] undecodable at runtime (not caught during pre-validation), "
                  f"substituting silence: {wav_rel_path}")
            waveform = np.zeros(int(0.5 * self.sample_rate), dtype=np.float32)   # 0.5s of silence as a safe placeholder
        if len(waveform) > self.max_samples:
            start = random.randint(0, len(waveform) - self.max_samples)
            waveform = waveform[start:start + self.max_samples]
        item["waveform"] = waveform
        return item


# =========================================================================== #
# 9. Collator
# =========================================================================== #
class SLURPCollator:
    def __init__(self, feature_extractor, tokenizer, slot2id: Dict[str, int],
                 sample_rate: int = 16000, max_text_len: int = 64,
                 whisper_tokenizer=None, asr_max_len: int = 64):
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.slot2id = slot2id
        self.sample_rate = sample_rate
        self.max_text_len = max_text_len
        # JOINT-ASR: when set, ground-truth transcripts are tokenized with Whisper's own
        # tokenizer (NOT the BERT tokenizer above) to build teacher-forcing labels for the
        # ASR loss. Only needed in joint-fine-tuning mode; left None otherwise so existing
        # non-joint training paths are completely unaffected.
        self.whisper_tokenizer = whisper_tokenizer
        self.asr_max_len = asr_max_len

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}

        if "acoustic_features" in batch[0]:
            # PERF: all cached features share the same fixed sequence length (Whisper pads
            # every clip to 30s internally), so this is a plain stack -- no encoder forward,
            # no feature extraction, no padding logic needed at train time.
            out["acoustic_states"] = torch.stack([b["acoustic_features"] for b in batch], dim=0)
        else:
            waveforms = [b["waveform"] for b in batch]
            audio_inputs = self.feature_extractor(waveforms, sampling_rate=self.sample_rate, return_tensors="pt")
            out["input_features"] = audio_inputs.input_features

        all_words = [b["words"] for b in batch]
        all_tags = [b["tags"] for b in batch]

        text_inputs = self.tokenizer(
            all_words, is_split_into_words=True, padding=True, truncation=True,
            max_length=self.max_text_len, return_tensors="pt",
        )

        slot_labels = torch.full(text_inputs["input_ids"].shape, -100, dtype=torch.long)
        for i, tags in enumerate(all_tags):
            word_ids = text_inputs.word_ids(batch_index=i)
            prev_word_idx = None
            for pos, w_idx in enumerate(word_ids):
                if w_idx is None:
                    continue
                if w_idx != prev_word_idx:
                    tag = tags[w_idx] if w_idx < len(tags) else "O"
                    slot_labels[i, pos] = self.slot2id.get(tag, self.slot2id["O"])
                prev_word_idx = w_idx

        out["input_ids"] = text_inputs["input_ids"]
        out["text_attention_mask"] = text_inputs["attention_mask"]
        out["slot_labels"] = slot_labels
        out["intent_labels"] = torch.tensor([b["intent_id"] for b in batch], dtype=torch.long)
        out["paths"] = [b["wav_rel_path"] for b in batch]
        # SLU-F1: keep the actual (hypothesis) words around so evaluate() can reconstruct
        # entity filler TEXT (not just BIO tag spans) -- SLU-F1's dist() function needs the
        # real filler strings to compute word/char-level distance between gold and predicted
        # entities, not just their label.
        out["words"] = all_words

        # JOINT-ASR: build teacher-forcing labels for the ASR seq2seq loss from the
        # ground-truth SLURP sentence, using Whisper's tokenizer/prefix tokens (NOT the
        # BERT tokenizer used for the SLU text branch above -- different vocabulary).
        # Padding positions are set to -100 so they don't contribute to the CE loss,
        # matching how WhisperForConditionalGeneration expects `labels`.
        if self.whisper_tokenizer is not None:
            gt_texts = [b["gt_transcript"] for b in batch]
            asr_label_inputs = self.whisper_tokenizer(
                gt_texts, padding=True, truncation=True, max_length=self.asr_max_len, return_tensors="pt",
            )
            asr_labels = asr_label_inputs["input_ids"]
            asr_labels[asr_labels == self.whisper_tokenizer.pad_token_id] = -100
            out["asr_labels"] = asr_labels

        return out


# =========================================================================== #
# 10. Model components
# =========================================================================== #
class ResidualAdapter(nn.Module):
    """Bottleneck residual adapter (Houlsby et al., 2019 style): down-project ->
    nonlinearity -> up-project, added back to the input via a residual connection.
    `up_proj` is zero-initialized so the adapter starts as an exact identity function --
    training begins at the frozen pretrained baseline and only gradually learns a
    noise-adaptation delta on top of it, rather than perturbing the representation from
    the first step. Vastly fewer trainable params than unfreezing whole transformer
    layers (bottleneck_dim << hidden_size), and crucially the pretrained weights
    themselves are never touched -- the original clean-speech representation is always
    still recoverable, so there's no risk of catastrophically forgetting it."""

    def __init__(self, hidden_size: int, bottleneck_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.down_proj = nn.Linear(hidden_size, bottleneck_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up_proj = nn.Linear(bottleneck_dim, hidden_size)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layer_norm(x)
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        return residual + x


def _make_adapter_hook(adapter: nn.Module):
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            hidden_states = adapter(output[0])
            return (hidden_states,) + tuple(output[1:])
        return adapter(output)
    return hook


def _configure_encoder_finetuning(
    encoder_module: nn.Module,
    hidden_size: int,
    mode: str,
    unfreeze_last_n_layers: int = 0,
    adapter_bottleneck_dim: int = 64,
    adapter_dropout: float = 0.1,
    owner_name: str = "encoder",
) -> Optional[nn.ModuleList]:
    """Shared freeze/adapter-attachment policy, factored out so both WhisperAcousticEncoder
    (the "frozen"/"last_n_layers"/"full"/"adapters" standalone path) and JointWhisperASR
    (the "joint" path, which wraps the SAME kind of WhisperEncoder inside a
    WhisperForConditionalGeneration) apply IDENTICAL freezing logic instead of two
    independently-maintained copies that could silently drift apart.

    Returns the trainable nn.ModuleList of ResidualAdapters when mode == "adapters"
    (the CALLER is responsible for storing it as an attribute so its parameters are
    included in the owning module's .parameters()); returns None otherwise.
    """
    assert mode in ("frozen", "last_n_layers", "full", "adapters"), mode

    if mode == "frozen":
        for p in encoder_module.parameters():
            p.requires_grad = False
        return None

    if mode == "full":
        for p in encoder_module.parameters():
            p.requires_grad = True
        return None

    if mode == "last_n_layers":
        for p in encoder_module.parameters():
            p.requires_grad = False
        encoder_layers = encoder_module.layers
        n = min(unfreeze_last_n_layers, len(encoder_layers))
        if n <= 0:
            print(f"[warn] [{owner_name}] mode='last_n_layers' but unfreeze_last_n_layers<=0; "
                  f"encoder will stay fully frozen.")
        for layer_module in encoder_layers[len(encoder_layers) - n:]:
            for p in layer_module.parameters():
                p.requires_grad = True
        if hasattr(encoder_module, "layer_norm") and n > 0:
            for p in encoder_module.layer_norm.parameters():
                p.requires_grad = True
        return None

    # mode == "adapters"
    for p in encoder_module.parameters():
        p.requires_grad = False   # pretrained weights: frozen, untouched
    adapters = nn.ModuleList([
        ResidualAdapter(hidden_size, adapter_bottleneck_dim, adapter_dropout)
        for _ in encoder_module.layers
    ])
    for layer_module, adapter in zip(encoder_module.layers, adapters):
        layer_module.register_forward_hook(_make_adapter_hook(adapter))
    return adapters


class WhisperAcousticEncoder(nn.Module):
    """
    `finetune_mode` controls how much of the frozen-pretrained Whisper encoder is
    allowed to adapt during training:

      "frozen"        -- (original behavior) all encoder params frozen. Fastest,
                          but the encoder's features were learned on clean-ish
                          pretraining data and never adapt to your domain -- at
                          severe SNRs (e.g. 0dB) these features may simply not
                          carry much signal, since nothing ever taught the
                          encoder what your noise looks like.
      "last_n_layers" -- only the last `unfreeze_last_n_layers` transformer
                          layers of the encoder (+ its final layer norm) are
                          trainable; everything before stays frozen. A middle
                          ground: lets the encoder specialize its late
                          representations to noisy conditions without the
                          compute/overfitting cost of fine-tuning the whole
                          stack, and without destroying the low-level acoustic
                          features the early layers already learned well. This
                          DOES directly modify the pretrained weights, though,
                          so it can drift away from (and potentially forget)
                          the original clean-speech representation.
      "adapters"      -- pretrained weights stay 100% frozen; a small trainable
                          ResidualAdapter (see above) is attached to every
                          encoder layer via a forward hook (not by wrapping/
                          replacing the layer -- see note below). Far fewer
                          trainable params than "last_n_layers", no risk of
                          forgetting the pretrained representation (its
                          weights are never touched), and since up_proj is
                          zero-initialized, training starts exactly at the
                          frozen baseline. Generally the safer default to try
                          before "full" fine-tuning.
      "full"          -- the entire encoder is trainable. Most capacity to
                          adapt, most risk of overfitting/forgetting, most
                          compute.

    NOTE: this class only ever runs the ENCODER -- it has no ASR decoding capability
    and its output for a given waveform is fixed unless it's being fine-tuned itself.
    For joint SLU + ASR fine-tuning (encoder trained using a real ASR loss against
    ground-truth transcripts, with the SAME adapted weights feeding the SLU heads),
    see JointWhisperASR below instead -- that's what whisper_finetune_mode == "joint"
    selects in build_model().

    NOTE: acoustic-feature caching (`use_precomputed_acoustic_features`) is only
    valid when finetune_mode == "frozen", since that's the only case where the
    encoder's output for a given waveform is guaranteed not to change during
    training. build_dataloaders() disables the cache automatically otherwise.
    """

    def __init__(
        self,
        model_name: str = "openai/whisper-small",
        layer: int = 6,
        finetune_mode: str = "frozen",
        unfreeze_last_n_layers: int = 0,
        adapter_bottleneck_dim: int = 64,
        adapter_dropout: float = 0.1,
    ):
        super().__init__()
        assert finetune_mode in ("frozen", "last_n_layers", "full", "adapters"), finetune_mode
        self.whisper = WhisperModel.from_pretrained(model_name, local_files_only=True)
        self.layer = layer
        self.hidden_size = self.whisper.config.d_model
        self.finetune_mode = finetune_mode
        self.unfreeze_last_n_layers = unfreeze_last_n_layers
        self.adapter_bottleneck_dim = adapter_bottleneck_dim
        self.adapter_dropout = adapter_dropout
        self.adapters = _configure_encoder_finetuning(
            self.whisper.encoder, self.hidden_size, finetune_mode,
            unfreeze_last_n_layers, adapter_bottleneck_dim, adapter_dropout,
            owner_name="WhisperAcousticEncoder",
        )
        self._warn_if_layer_choice_wastes_trainable_layers()

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        print(f"[WhisperAcousticEncoder] finetune_mode={self.finetune_mode} "
              f"trainable_params={n_trainable:,}/{n_total:,} ({n_trainable / max(n_total, 1):.2%})")

    def _warn_if_layer_choice_wastes_trainable_layers(self):
        if self.finetune_mode == "frozen":
            return
        num_encoder_layers = len(self.whisper.encoder.layers)
        if self.layer < num_encoder_layers:
            print(f"[warn] whisper_layer={self.layer} selects a hidden state that only depends on "
                  f"encoder layers 0..{self.layer - 1} of {num_encoder_layers}. With "
                  f"finetune_mode='{self.finetune_mode}', any trainable parameters in layers "
                  f"{self.layer}..{num_encoder_layers - 1} will never receive gradient (they still run "
                  f"forward, just wastefully). Consider whisper_layer={num_encoder_layers} (the final "
                  f"layer-normed output) to make full use of every trainable layer/adapter.")

    def train(self, mode: bool = True):
        super().train(mode)
        if self.finetune_mode in ("frozen", "adapters"):
            self.whisper.eval()
        return self

    def forward(self, input_features: torch.Tensor, asr_labels: Optional[torch.Tensor] = None):
        """Returns (acoustic_states, asr_loss). asr_loss is always None here -- this class
        has no decoder / no ASR capability. `asr_labels` is accepted (and ignored) purely so
        this class has the SAME call signature as JointWhisperASR.forward, letting
        SemiCascadedSLU_SLURP.forward call self.acoustic_encoder(...) identically regardless
        of which encoder variant is in use."""
        any_trainable = any(p.requires_grad for p in self.parameters())
        ctx = torch.no_grad() if not any_trainable else torch.enable_grad()
        with ctx:
            enc_out = self.whisper.encoder(input_features, output_hidden_states=True, return_dict=True)
        return enc_out.hidden_states[self.layer], None


class JointWhisperASR(nn.Module):
    """
    Unified Whisper wrapper for JOINT SLU + ASR fine-tuning (whisper_finetune_mode == "joint").

    Owns ONE WhisperForConditionalGeneration (encoder + decoder). The same encoder weights
    are used for three purposes that all share a SINGLE encoder forward pass per training step:

      1. SLU features: `enc_out.hidden_states[self.layer]` feeds the cross-attention fusion /
         intent / slot heads, exactly like WhisperAcousticEncoder does.
      2. ASR loss: `enc_out.hidden_states[-1]` (the final, layer-normed encoder output -- what
         the decoder is actually trained to cross-attend into) is passed to the decoder via
         `encoder_outputs=`, together with ground-truth transcript token ids as `labels`, to
         get a standard teacher-forced seq2seq cross-entropy loss. This is NOT `.generate()` --
         it's a single differentiable forward pass, so it's cheap enough to run every step.
      3. Periodic hypothesis refresh: `generate_transcripts()` runs the real autoregressive
         `.generate()` loop (non-differentiable, no_grad) to produce updated ASR hypotheses
         for the BERT text branch and the slot-tag alignment cache. This is far more expensive
         than (1)+(2), so it is NOT run every step -- see `asr_refresh_every_n_epochs` in
         SLURPConfig and `refresh_asr_and_slot_caches()` below, called between epochs.

    Freezing policy: `encoder_finetune_mode` controls the ENCODER via the same shared
    `_configure_encoder_finetuning` helper WhisperAcousticEncoder uses (so "adapters" /
    "last_n_layers" / "full" behave identically here). The DECODER is controlled separately
    by `decoder_trainable` (default False -- see SLURPConfig.asr_decoder_trainable docstring
    for why). Critically, even with the decoder fully frozen (requires_grad=False on all its
    params), the ASR loss computed via decoder cross-entropy still backprops gradient INTO
    the encoder's output (`encoder_outputs.last_hidden_state`), which is the mechanism that
    lets the encoder adapt to noise using ASR supervision even without touching decoder
    weights at all.
    """

    def __init__(
        self,
        model_name: str = "openai/whisper-small",
        layer: int = 6,
        encoder_finetune_mode: str = "adapters",
        unfreeze_last_n_layers: int = 0,
        adapter_bottleneck_dim: int = 64,
        adapter_dropout: float = 0.1,
        decoder_trainable: bool = False,
        language: str = "en",
    ):
        super().__init__()
        self.whisper = WhisperForConditionalGeneration.from_pretrained(model_name, local_files_only=True)
        self.whisper.generation_config.forced_decoder_ids = None
        # Whisper's own tokenizer -- separate from the BERT tokenizer used for the SLU text
        # branch. Needed both to tokenize ground-truth labels for the ASR loss and to decode
        # generate()'d ids back into hypothesis strings.
        self.processor = WhisperProcessor.from_pretrained(model_name, local_files_only=True)
        self.tokenizer = self.processor.tokenizer

        self.layer = layer
        self.hidden_size = self.whisper.config.d_model
        self.encoder_finetune_mode = encoder_finetune_mode
        self.decoder_trainable = decoder_trainable
        self.language = language

        encoder_module = self.whisper.get_encoder()
        self.adapters = _configure_encoder_finetuning(
            encoder_module, self.hidden_size, encoder_finetune_mode,
            unfreeze_last_n_layers, adapter_bottleneck_dim, adapter_dropout,
            owner_name="JointWhisperASR(encoder)",
        )

        # Decoder + lm_head: frozen unless explicitly requested. This is INDEPENDENT of
        # encoder_finetune_mode -- the encoder can be adapting via "adapters" while the
        # decoder stays completely frozen (the default and recommended combination).
        for p in self.whisper.get_decoder().parameters():
            p.requires_grad = decoder_trainable
        if hasattr(self.whisper, "proj_out"):
            for p in self.whisper.proj_out.parameters():
                p.requires_grad = decoder_trainable

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        print(f"[JointWhisperASR] encoder_finetune_mode={encoder_finetune_mode} "
              f"decoder_trainable={decoder_trainable} "
              f"trainable_params={n_trainable:,}/{n_total:,} ({n_trainable / max(n_total, 1):.2%})")

    def train(self, mode: bool = True):
        # Mirrors WhisperAcousticEncoder.train(): pin frozen sub-modules to eval() (killing
        # their internal dropout) independent of the outer .train()/.eval() calls, since a
        # plain nn.Module.train() recurses regardless of requires_grad.
        super().train(mode)
        if self.encoder_finetune_mode in ("frozen", "adapters"):
            self.whisper.get_encoder().eval()
        if not self.decoder_trainable:
            self.whisper.get_decoder().eval()
        return self

    def forward(self, input_features: torch.Tensor, asr_labels: Optional[torch.Tensor] = None):
        """Returns (acoustic_states, asr_loss). `acoustic_states` = hidden_states[self.layer]
        for the SLU heads (identical role to WhisperAcousticEncoder.forward). If `asr_labels`
        is supplied, also computes the teacher-forced ASR seq2seq loss by feeding the encoder's
        FINAL hidden state (hidden_states[-1], not hidden_states[self.layer]) to the decoder via
        `encoder_outputs=` -- this reuses the single encoder forward pass already computed
        above instead of running the encoder twice."""
        any_encoder_trainable = any(p.requires_grad for p in self.whisper.get_encoder().parameters())
        enc_ctx = torch.enable_grad() if (any_encoder_trainable or asr_labels is not None) else torch.no_grad()
        with enc_ctx:
            enc_out = self.whisper.get_encoder()(input_features, output_hidden_states=True, return_dict=True)

        acoustic_states = enc_out.hidden_states[self.layer]

        asr_loss = None
        if asr_labels is not None:
            from transformers.modeling_outputs import BaseModelOutput
            final_encoder_states = enc_out.hidden_states[-1]
            encoder_outputs = BaseModelOutput(last_hidden_state=final_encoder_states)
            # Always compute this under grad-enabled: even when decoder params are frozen
            # (requires_grad=False), we still need gradient to FLOW THROUGH the decoder's
            # computation graph back into `final_encoder_states` -- freezing is enforced by
            # requires_grad on the decoder's own parameters, not by cutting the graph here.
            asr_out = self.whisper(encoder_outputs=encoder_outputs, labels=asr_labels)
            asr_loss = asr_out.loss

        return acoustic_states, asr_loss

    @torch.no_grad()
    def generate_transcripts(self, input_features: torch.Tensor) -> List[str]:
        """Real autoregressive decode (greedy, num_beams=1 -- see ASRTranscriber for the same
        rationale). Non-differentiable; used only for the periodic hypothesis refresh, never
        inside the training step itself."""
        was_training = self.training
        self.eval()
        generated_ids = self.whisper.generate(
            input_features, language=self.language, task="transcribe", num_beams=1,
        )
        texts = [t.strip() for t in self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)]
        if was_training:
            self.train()
        return texts


class TextEncoder(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased", freeze: bool = False):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name, local_files_only=True)
        self.hidden_size = self._get_hidden_size(self.bert.config)
        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False

    @staticmethod
    def _get_hidden_size(config) -> int:
        for attr in ("hidden_size", "dim"):
            if hasattr(config, attr):
                return getattr(config, attr)
        raise AttributeError(f"Could not determine hidden size from config {type(config).__name__}")

    def forward(self, input_ids, attention_mask):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


class CrossAttentionBlock(nn.Module):
    """
    Generic single-direction cross-attention block: `query_states` attends into
    `key_states`. Which modality plays which role is decided by the caller --
    this lets the same class implement both "text queries audio" (original
    design) and "audio queries text" (the swap) by just swapping which
    tensor is passed as query vs. key.
    """

    def __init__(self, query_dim, key_dim, fusion_dim=768, num_heads=8, ffn_dim=2048, dropout=0.1):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, fusion_dim)
        self.key_proj = nn.Linear(key_dim, fusion_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(fusion_dim)
        self.ffn = nn.Sequential(
            nn.Linear(fusion_dim, ffn_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_dim, fusion_dim)
        )
        self.norm2 = nn.LayerNorm(fusion_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query_states, key_states, key_padding_mask=None):
        Q = self.query_proj(query_states)
        K = self.key_proj(key_states)
        attn_out, attn_weights = self.cross_attn(
            query=Q, key=K, value=K, key_padding_mask=key_padding_mask,
            need_weights=True, average_attn_weights=True,
        )
        fused = self.norm1(Q + self.dropout(attn_out))
        fused = self.norm2(fused + self.dropout(self.ffn(fused)))
        return fused, attn_weights


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).float()
    return (x * m).sum(1) / m.sum(1).clamp(min=1e-6)


def _focal_loss(
    logits: torch.Tensor, target: torch.Tensor,
    weight: Optional[torch.Tensor] = None, gamma: float = 2.0, ignore_index: int = -100,
) -> torch.Tensor:
    """Focal loss (Lin et al., 2017): down-weights easy/already-confident examples so
    the loss concentrates on hard/rare classes -- complements (or substitutes for) static
    inverse-frequency class weights, since weighting alone can't fix a class the model
    already predicts with high confidence for the wrong label; focal loss keeps pushing on
    exactly those hard cases throughout training. Setting gamma=0 recovers plain
    (optionally class-weighted) cross-entropy.
    Works for both the intent head (no padding, ignore_index never matches a real label)
    and the slot head (per-token, with -100 marking non-first-subword positions to ignore).
    """
    logp = F.log_softmax(logits, dim=-1)
    ce = F.nll_loss(logp, target, weight=weight, ignore_index=ignore_index, reduction="none")
    valid_mask = target != ignore_index
    pt = torch.exp(-ce)                       # model's predicted probability of the true class
    focal_term = (1.0 - pt).clamp(min=0.0) ** gamma
    per_element_loss = focal_term * ce
    denom = valid_mask.sum().clamp(min=1)
    return per_element_loss.sum() / denom


# =========================================================================== #
# 11. Full model
# =========================================================================== #
class SemiCascadedSLU_SLURP(nn.Module):
    """
    `fusion_mode` controls which modality is the cross-attention query:

      "text_query"  -- text queries audio (original design). Fused output is
                        word-level: feeds both slot_head directly and, pooled
                        over text positions, intent_head.

      "audio_query" -- audio queries text. Fused output is frame-level: feeds
                        intent_head directly (mean-pooled over frames, which is
                        valid since Whisper always produces a fixed-length
                        frame sequence). A second block then has TEXT query
                        into that frame-level representation to recover a
                        word-level representation for slot_head, since the
                        slot task inherently needs one prediction per word
                        and a frame-indexed sequence has no such alignment.

      "dual_branch" -- both blocks run independently: the text-query branch
                        feeds slot_head (identical to "text_query"), the
                        audio-query branch feeds intent_head (identical
                        pooling to "audio_query"). No secondary projection is
                        needed here because each branch only serves the head
                        that matches its native granularity.

    `whisper_finetune_mode == "joint"` swaps self.acoustic_encoder from
    WhisperAcousticEncoder to JointWhisperASR (see that class's docstring) -- this is
    the only change needed to enable joint SLU + ASR fine-tuning, since both classes
    share the same `forward(input_features, asr_labels=None) -> (acoustic_states,
    asr_loss)` call signature.
    """

    def __init__(
        self,
        num_intents: int,
        num_slot_labels: int,
        whisper_model_name: str = "openai/whisper-small",
        bert_model_name: str = "bert-base-uncased",
        whisper_layer: int = 6,
        fusion_dim: int = 768,
        whisper_finetune_mode: str = "frozen",
        whisper_unfreeze_last_n_layers: int = 0,
        adapter_bottleneck_dim: int = 64,
        adapter_dropout: float = 0.1,
        freeze_bert: bool = False,
        dropout: float = 0.1,
        fusion_mode: str = "text_query",
        intent_class_weights: Optional[torch.Tensor] = None,
        slot_class_weights: Optional[torch.Tensor] = None,
        loss_type: str = "cross_entropy",
        focal_gamma: float = 2.0,
        # --- JOINT-ASR params (only used when whisper_finetune_mode == "joint") ---
        joint_encoder_finetune_mode: str = "adapters",
        asr_decoder_trainable: bool = False,
        asr_loss_weight: float = 1.0,
    ):
        super().__init__()
        assert fusion_mode in ("text_query", "audio_query", "dual_branch"), fusion_mode
        assert loss_type in ("cross_entropy", "focal"), loss_type
        assert whisper_finetune_mode in ("frozen", "last_n_layers", "full", "adapters", "joint"), whisper_finetune_mode
        self.fusion_mode = fusion_mode
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
        self.whisper_finetune_mode = whisper_finetune_mode
        self.asr_loss_weight = asr_loss_weight

        if whisper_finetune_mode == "joint":
            self.acoustic_encoder = JointWhisperASR(
                whisper_model_name, whisper_layer,
                encoder_finetune_mode=joint_encoder_finetune_mode,
                unfreeze_last_n_layers=whisper_unfreeze_last_n_layers,
                adapter_bottleneck_dim=adapter_bottleneck_dim, adapter_dropout=adapter_dropout,
                decoder_trainable=asr_decoder_trainable,
            )
        else:
            self.acoustic_encoder = WhisperAcousticEncoder(
                whisper_model_name, whisper_layer,
                finetune_mode=whisper_finetune_mode, unfreeze_last_n_layers=whisper_unfreeze_last_n_layers,
                adapter_bottleneck_dim=adapter_bottleneck_dim, adapter_dropout=adapter_dropout,
            )
        self.text_encoder = TextEncoder(bert_model_name, freeze_bert)
        text_dim = self.text_encoder.hidden_size
        acoustic_dim = self.acoustic_encoder.hidden_size

        if fusion_mode in ("text_query", "dual_branch"):
            self.text_query_block = CrossAttentionBlock(text_dim, acoustic_dim, fusion_dim, dropout=dropout)
            self.text_gate = nn.Sequential(nn.Linear(fusion_dim * 2, fusion_dim), nn.Sigmoid())

        if fusion_mode in ("audio_query", "dual_branch"):
            self.audio_query_block = CrossAttentionBlock(acoustic_dim, text_dim, fusion_dim, dropout=dropout)
            self.audio_gate = nn.Sequential(nn.Linear(fusion_dim * 2, fusion_dim), nn.Sigmoid())

        if fusion_mode == "audio_query":
            # secondary block: text queries into the (fusion_dim-sized) frame-level
            # audio-query output, to recover a word-level representation for slot_head.
            self.slot_projection_block = CrossAttentionBlock(text_dim, fusion_dim, fusion_dim, dropout=dropout)
            self.slot_gate2 = nn.Sequential(nn.Linear(fusion_dim * 2, fusion_dim), nn.Sigmoid())

        self.dropout = nn.Dropout(dropout)
        self.intent_head = nn.Linear(fusion_dim, num_intents)
        self.slot_head = nn.Linear(fusion_dim, num_slot_labels)

        # PERF/quality: SLURP's intent/slot label sets are long-tailed; unweighted
        # cross-entropy lets the model coast on frequent classes. register_buffer so these
        # move with the model across .to(device) calls automatically. None is fine too --
        # F.cross_entropy(weight=None) is just unweighted.
        self.register_buffer("intent_class_weights", intent_class_weights, persistent=False)
        self.register_buffer("slot_class_weights", slot_class_weights, persistent=False)

    def _compute_loss(self, logits: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor]) -> torch.Tensor:
        if self.loss_type == "focal":
            return _focal_loss(logits, target, weight=weight, gamma=self.focal_gamma, ignore_index=-100)
        return F.cross_entropy(logits, target, weight=weight, ignore_index=-100)

    def forward(
        self,
        input_features: Optional[torch.Tensor] = None,
        acoustic_states: Optional[torch.Tensor] = None,   # PERF: precomputed frozen-encoder output
        input_ids: torch.Tensor = None,
        text_attention_mask: torch.Tensor = None,
        acoustic_attention_mask: torch.Tensor = None,
        intent_labels: torch.Tensor = None,
        slot_labels: torch.Tensor = None,
        slot_loss_weight: float = 1.0,
        asr_labels: Optional[torch.Tensor] = None,   # JOINT-ASR: ground-truth transcript token ids
        **kwargs,
    ):
        asr_loss = None
        if acoustic_states is None:
            assert input_features is not None, "must supply either input_features or precomputed acoustic_states"
            acoustic_states, asr_loss = self.acoustic_encoder(input_features, asr_labels=asr_labels)
        # else: precomputed/cached features path (only ever used in "frozen" mode) never
        # has a live ASR loss, since there's no encoder forward pass to attach a decoder to.

        text_states = self.text_encoder(input_ids, text_attention_mask)

        text_key_padding_mask = ~text_attention_mask.bool()
        acoustic_key_padding_mask = ~acoustic_attention_mask.bool() if acoustic_attention_mask is not None else None

        intent_logits = None
        slot_logits = None
        extra: Dict[str, torch.Tensor] = {}

        if self.fusion_mode in ("text_query", "dual_branch"):
            fused_t, attn_t = self.text_query_block(text_states, acoustic_states, key_padding_mask=acoustic_key_padding_mask)
            text_proj = self.text_query_block.query_proj(text_states)
            g_t = self.text_gate(torch.cat([fused_t, text_proj], dim=-1))
            mixed_t = self.dropout(g_t * fused_t + (1 - g_t) * text_proj)
            slot_logits = self.slot_head(mixed_t)
            extra["text_query_attn"] = attn_t
            extra["text_gate"] = g_t
            if self.fusion_mode == "text_query":
                pooled = _masked_mean(mixed_t, text_attention_mask)
                intent_logits = self.intent_head(pooled)

        if self.fusion_mode in ("audio_query", "dual_branch"):
            frame_fused, attn_a = self.audio_query_block(acoustic_states, text_states, key_padding_mask=text_key_padding_mask)
            acoustic_proj = self.audio_query_block.query_proj(acoustic_states)
            g_a = self.audio_gate(torch.cat([frame_fused, acoustic_proj], dim=-1))
            mixed_a = self.dropout(g_a * frame_fused + (1 - g_a) * acoustic_proj)
            pooled_a = mixed_a.mean(dim=1)   # fixed-length frame sequence, no padding to mask out
            intent_logits = self.intent_head(pooled_a)
            extra["audio_query_attn"] = attn_a
            extra["audio_gate"] = g_a

            if self.fusion_mode == "audio_query":
                word_fused, attn_t2 = self.slot_projection_block(text_states, mixed_a)
                text_proj2 = self.slot_projection_block.query_proj(text_states)
                g_t2 = self.slot_gate2(torch.cat([word_fused, text_proj2], dim=-1))
                mixed_t2 = self.dropout(g_t2 * word_fused + (1 - g_t2) * text_proj2)
                slot_logits = self.slot_head(mixed_t2)
                extra["slot_projection_attn"] = attn_t2

        output = {
            "intent_logits": intent_logits,
            "slot_logits": slot_logits,
            **extra,
        }

        if intent_labels is not None and slot_labels is not None:
            intent_loss = self._compute_loss(intent_logits, intent_labels, self.intent_class_weights)
            slot_loss = self._compute_loss(
                slot_logits.view(-1, slot_logits.size(-1)), slot_labels.view(-1), self.slot_class_weights,
            )
            total_loss = intent_loss + slot_loss_weight * slot_loss
            output["intent_loss"] = intent_loss
            output["slot_loss"] = slot_loss
            if asr_loss is not None:
                # JOINT-ASR: fold the teacher-forced ASR seq2seq loss into the joint objective.
                # Gradient from this term flows into the shared encoder (and, if
                # asr_decoder_trainable=True, the decoder too), adapting it to noisy audio
                # using strong ground-truth ASR supervision -- the SAME adapted encoder
                # representation is what intent_logits/slot_logits above were computed from.
                total_loss = total_loss + self.asr_loss_weight * asr_loss
                output["asr_loss"] = asr_loss
            output["loss"] = total_loss

        return output


# =========================================================================== #
# 12. Metrics (unchanged)
# =========================================================================== #
def _intent_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "intent_acc": acc,
        "intent_precision_macro": p_macro, "intent_recall_macro": r_macro, "intent_f1_macro": f1_macro,
        "intent_precision_weighted": p_w, "intent_recall_weighted": r_w, "intent_f1_weighted": f1_w,
    }


def _bio_to_spans(tags: List[str]) -> List[Tuple[str, int, int]]:
    spans = []
    start, ent_type = None, None
    for i, tag in enumerate(tags + ["O"]):
        if tag.startswith("B-"):
            if start is not None:
                spans.append((ent_type, start, i))
            start, ent_type = i, tag[2:]
        elif tag.startswith("I-") and ent_type == tag[2:]:
            continue
        else:
            if start is not None:
                spans.append((ent_type, start, i))
                start, ent_type = None, None
            if tag.startswith("I-"):
                start, ent_type = i, tag[2:]
    return spans


def _slot_span_metrics(all_true_tags: List[List[str]], all_pred_tags: List[List[str]]) -> Dict[str, float]:
    tp = fp = fn = 0
    for true_tags, pred_tags in zip(all_true_tags, all_pred_tags):
        true_spans = set(_bio_to_spans(true_tags))
        pred_spans = set(_bio_to_spans(pred_tags))
        tp += len(true_spans & pred_spans)
        fp += len(pred_spans - true_spans)
        fn += len(true_spans - pred_spans)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"slot_precision": precision, "slot_recall": recall, "slot_f1": f1}


def _slot_type_span_metrics(
    all_true_tags: List[List[str]], all_pred_tags: List[List[str]]
) -> Dict[str, Dict[str, float]]:
    """Same as _slot_span_metrics but broken out PER ENTITY TYPE (not overall) --
    this is what actually explains a low aggregate slot_f1: a handful of entity
    types the model essentially never gets right."""
    from collections import defaultdict
    tp: Dict[str, int] = defaultdict(int)
    fp: Dict[str, int] = defaultdict(int)
    fn: Dict[str, int] = defaultdict(int)
    support: Dict[str, int] = defaultdict(int)

    for true_tags, pred_tags in zip(all_true_tags, all_pred_tags):
        true_spans = set(_bio_to_spans(true_tags))
        pred_spans = set(_bio_to_spans(pred_tags))
        for s in true_spans:
            support[s[0]] += 1
        for s in true_spans & pred_spans:
            tp[s[0]] += 1
        for s in pred_spans - true_spans:
            fp[s[0]] += 1
        for s in true_spans - pred_spans:
            fn[s[0]] += 1

    all_types = set(tp) | set(fp) | set(fn) | set(support)
    result = {}
    for t in all_types:
        p = tp[t] / (tp[t] + fp[t]) if (tp[t] + fp[t]) > 0 else 0.0
        r = tp[t] / (tp[t] + fn[t]) if (tp[t] + fn[t]) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        result[t] = {"precision": p, "recall": r, "f1": f1, "support": support[t]}
    return result


# =========================================================================== #
# 12a. SLU-F1 (Bastianelli et al., 2020, EMNLP -- "SLURP: A Spoken Language
#      Understanding Resource Package", Section 4, Algorithm 1)
#
#      Unlike exact-span slot F1 (_slot_span_metrics above), SLU-F1 does NOT require an
#      exact match between predicted and gold entity fillers to count a match. It relaxes
#      the equality e_i =:= e_hat_k to only require the LABEL to match (l_i = l_k); a
#      matched pair still contributes +1 TP, but FP/FN are additionally incremented by a
#      distance dist(gold_filler, pred_filler) in [0, ~1] measuring how far off the filler
#      TEXT is. This means an entity that's correctly typed but has an ASR-mangled filler
#      is scored as "mostly right" rather than either a hard hit or a hard miss -- exactly
#      the ASR-misalignment-tolerant behavior the paper designed this metric for.
#
#      Two distance functions are used and then combined:
#        - Word-F1: dist = word-level WER on the filler (token-level, strict about e.g.
#          singular/plural mismatches -- shows how much ASR is hurting NLU).
#        - Char-F1: dist = normalised character-level Levenshtein on the filler (much less
#          sensitive to small transcription noise -- shows NLU quality despite noise).
#        - SLU-F1: combines Word-F1 and Char-F1 by SUMMING their confusion matrices
#          (TP/FP/FN) before computing the final precision/recall/F1, exactly as described
#          in the paper ("we combine Word-F1 and Char-F1 in a single number SLU-F1, which
#          evaluates the final performance over the sum of the confusion matrices").
# =========================================================================== #
def _bio_to_entities(words: List[str], tags: List[str]) -> List[Tuple[str, List[str]]]:
    """Like _bio_to_spans, but returns (label, filler_words) pairs -- the actual entity
    text -- since SLU-F1's dist() function needs to compare filler CONTENT, not just index
    spans."""
    spans = _bio_to_spans(tags)
    return [(label, words[start:end]) for label, start, end in spans]


def _char_levenshtein_normalized(ref_words: List[str], hyp_words: List[str]) -> float:
    """Character-level Levenshtein distance between two filler word lists (joined with
    spaces, lowercased), normalised by the longer string's length so the result is in
    [0, 1]. This is the "Char-F1" distance function from the paper -- much less sensitive
    to small transcription noise than word-level WER."""
    ref_str = " ".join(ref_words).lower()
    hyp_str = " ".join(hyp_words).lower()
    n, m = len(ref_str), len(hyp_str)
    if n == 0 and m == 0:
        return 0.0
    if n == 0 or m == 0:
        return 1.0
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_str[i - 1] == hyp_str[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m] / max(n, m)


def _dist_f1_confusion(
    all_true_entities: List[List[Tuple[str, List[str]]]],
    all_pred_entities: List[List[Tuple[str, List[str]]]],
    dist_fn,
) -> Tuple[float, float, float]:
    """Implements Algorithm 1 ("dist-F1") from the SLURP paper, aggregated (summed) over
    every utterance in the corpus -- i.e. a corpus-level micro-averaged confusion matrix,
    matching how Word-F1/Char-F1/SLU-F1 are reported in the paper's results table.

    Per utterance: each predicted entity is greedily matched, in order, against the closest
    (by `dist_fn`) remaining gold entity of the SAME label. A match: TP += 1, FP/FN each +=
    dist(gold_filler, pred_filler) -- so an exact filler match costs nothing beyond the +1
    TP, while a partially-wrong filler is penalised proportionally instead of being scored
    as a hard miss. A predicted entity with no remaining gold entity of that label (wrong
    label, or all gold entities of that label already consumed by an earlier, better-
    matching prediction) is a full FP (+1). Any gold entities left unmatched at the end of
    the utterance are full FNs (+1 each)."""
    TP = FP = FN = 0.0
    for true_entities, pred_entities in zip(all_true_entities, all_pred_entities):
        gold_remaining = list(true_entities)
        for label, pred_filler in pred_entities:
            candidates = [i for i, (g_label, _) in enumerate(gold_remaining) if g_label == label]
            if candidates:
                best_i = min(candidates, key=lambda i: dist_fn(gold_remaining[i][1], pred_filler))
                d = dist_fn(gold_remaining[best_i][1], pred_filler)
                TP += 1
                FP += d
                FN += d
                del gold_remaining[best_i]
            else:
                FP += 1
        FN += len(gold_remaining)
    return TP, FP, FN


def _prf1_from_confusion(TP: float, FP: float, FN: float) -> Dict[str, float]:
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_slu_f1(
    all_true_tags: List[List[str]],
    all_pred_tags: List[List[str]],
    all_words: List[List[str]],
) -> Dict[str, Dict[str, float]]:
    """SLU-F1 (Bastianelli et al., 2020). Computes Word-F1 and Char-F1 separately (each its
    own dist-F1 confusion matrix -- see _dist_f1_confusion), then combines them into the
    final SLU-F1 by summing confusion matrices before computing precision/recall/F1, exactly
    as specified in the paper.

    `all_words[i]` must be the SAME word sequence that `all_true_tags[i]`/`all_pred_tags[i]`
    were assigned against (one tag per word, in order) -- see evaluate() for how this is
    reconstructed from the collated batch.

    Returns {"word_f1": {precision,recall,f1}, "char_f1": {...}, "slu_f1": {...}}."""
    all_true_entities = [_bio_to_entities(words, tags) for words, tags in zip(all_words, all_true_tags)]
    all_pred_entities = [_bio_to_entities(words, tags) for words, tags in zip(all_words, all_pred_tags)]

    tp_w, fp_w, fn_w = _dist_f1_confusion(all_true_entities, all_pred_entities, _word_error_rate)
    tp_c, fp_c, fn_c = _dist_f1_confusion(all_true_entities, all_pred_entities, _char_levenshtein_normalized)

    return {
        "word_f1": _prf1_from_confusion(tp_w, fp_w, fn_w),
        "char_f1": _prf1_from_confusion(tp_c, fp_c, fn_c),
        "slu_f1": _prf1_from_confusion(tp_w + tp_c, fp_w + fp_c, fn_w + fn_c),
    }


def print_intent_per_class_report(
    all_intent_true: List[int], all_intent_pred: List[int], vocab: SLURPLabelVocab, top_n: int = 15,
) -> List[Tuple[str, int, float, float, float]]:
    """Prints the worst `top_n` intents by F1 (ties in with why macro metrics can be much
    lower than accuracy/weighted-F1: macro gives every class equal weight, so a long tail
    of near-zero-F1 rare intents drags the average down even when common intents are fine)."""
    from sklearn.metrics import precision_recall_fscore_support
    labels = list(range(vocab.num_intents))
    p, r, f1, support = precision_recall_fscore_support(
        all_intent_true, all_intent_pred, labels=labels, average=None, zero_division=0
    )
    rows = [(vocab.id2intent[i], int(support[i]), float(p[i]), float(r[i]), float(f1[i])) for i in labels]
    rows.sort(key=lambda x: x[4])   # ascending F1 -- worst first

    print(f"\n--- worst {top_n} intents by F1 (support = # test examples for that intent) ---")
    header = f"{'intent':<32}{'support':>8}{'precision':>10}{'recall':>8}{'f1':>8}"
    print(header)
    print("-" * len(header))
    for name, sup, pp, rr, ff in rows[:top_n]:
        print(f"{name:<32}{sup:>8}{pp:>10.3f}{rr:>8.3f}{ff:>8.3f}")

    zero_f1_with_support = [row for row in rows if row[4] == 0.0 and row[1] > 0]
    print(f"\n{len(zero_f1_with_support)}/{len(rows)} intents have F1=0.0 despite having test "
          f"examples -- these alone pull the macro average down substantially.")
    return rows


def print_slot_per_class_report(
    all_slot_true_tags: List[List[str]], all_slot_pred_tags: List[List[str]], top_n: int = 15,
) -> List[Tuple[str, Dict[str, float]]]:
    """Prints the worst `top_n` slot ENTITY TYPES by F1 (span-level, not per-BIO-tag)."""
    type_metrics = _slot_type_span_metrics(all_slot_true_tags, all_slot_pred_tags)
    rows = sorted(type_metrics.items(), key=lambda kv: kv[1]["f1"])

    print(f"\n--- worst {top_n} slot types by F1 (support = # true spans of that type in test set) ---")
    header = f"{'slot_type':<32}{'support':>8}{'precision':>10}{'recall':>8}{'f1':>8}"
    print(header)
    print("-" * len(header))
    for name, m in rows[:top_n]:
        print(f"{name:<32}{m['support']:>8}{m['precision']:>10.3f}{m['recall']:>8.3f}{m['f1']:>8.3f}")

    zero_f1_with_support = [row for row in rows if row[1]["f1"] == 0.0 and row[1]["support"] > 0]
    print(f"\n{len(zero_f1_with_support)}/{len(rows)} slot types have F1=0.0 despite having true spans in "
          f"the test set.")
    return rows


def _forward_batch(model: nn.Module, batch: Dict) -> Dict:
    kwargs = dict(
        input_ids=batch["input_ids"],
        text_attention_mask=batch["text_attention_mask"],
        intent_labels=batch["intent_labels"],
        slot_labels=batch["slot_labels"],
    )
    if "acoustic_states" in batch:
        kwargs["acoustic_states"] = batch["acoustic_states"]
    else:
        kwargs["input_features"] = batch["input_features"]
    # JOINT-ASR: pass ground-truth transcript token ids through so the model can compute
    # the teacher-forced ASR loss. Absent entirely (via SLURPCollator) whenever
    # whisper_finetune_mode != "joint", so this is a no-op for every other mode.
    if "asr_labels" in batch:
        kwargs["asr_labels"] = batch["asr_labels"]
    return model(**kwargs)


_WER_BUCKETS = [("low (WER<0.25)", 0.0, 0.25), ("medium (0.25-0.5)", 0.25, 0.5),
                ("high (0.5-0.75)", 0.5, 0.75), ("catastrophic (>=0.75)", 0.75, float("inf"))]


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: str, vocab: SLURPLabelVocab,
    wer_lookup: Optional[Dict[str, float]] = None,
    per_class_report: bool = False,
    per_class_report_top_n: int = 15,
) -> Dict[str, float]:
    """If `wer_lookup` (wav_rel_path -> ASR word error rate, from compute_asr_wer_report)
    is supplied, metrics are ALSO broken out per WER bucket under the "by_wer" key --
    e.g. to see directly whether a fusion_mode holds up better on the utterances whose
    ASR hypothesis was most garbled, which is exactly the failure mode expected at
    severe SNRs like 0dB.

    If `per_class_report=True`, also prints the worst-performing intents/slot-types by
    F1 -- this is the direct way to check whether a low macro-F1/slot_f1 is being driven
    by a handful of near-zero-F1 rare classes (very common on SLURP's long-tailed label
    set) rather than broadly poor performance."""
    model.eval()
    total, correct_intent = 0, 0
    total_loss = 0.0
    all_intent_true, all_intent_pred = [], []
    all_slot_true_tags, all_slot_pred_tags, all_slot_words = [], [], []
    all_paths: List[str] = []

    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = _forward_batch(model, batch)

        bsz = batch["intent_labels"].size(0)
        total_loss += out["loss"].item() * bsz
        total += bsz

        pred_intent = out["intent_logits"].argmax(-1)
        correct_intent += (pred_intent == batch["intent_labels"]).sum().item()
        all_intent_true += batch["intent_labels"].cpu().tolist()
        all_intent_pred += pred_intent.cpu().tolist()
        all_paths += batch["paths"]

        pred_slot_ids = out["slot_logits"].argmax(-1)
        slot_labels = batch["slot_labels"]
        mask = slot_labels != -100
        for i in range(bsz):
            true_ids = slot_labels[i][mask[i]].cpu().tolist()
            pred_ids = pred_slot_ids[i][mask[i]].cpu().tolist()
            all_slot_true_tags.append([vocab.id2slot[t] for t in true_ids])
            all_slot_pred_tags.append([vocab.id2slot[p] for p in pred_ids])
            # The unmasked (first-subword) positions above appear in the SAME left-to-right
            # order as the original word list (see SLURPCollator) -- one entry per word, so
            # batch["words"][i] lines up positionally with true_ids/pred_ids. Sliced
            # defensively in case max_text_len truncation dropped trailing words.
            all_slot_words.append(batch["words"][i][:len(true_ids)])

    metrics = {"loss": total_loss / total, "intent_acc": correct_intent / total}
    metrics.update(_intent_metrics(all_intent_true, all_intent_pred))

    # SLU-F1 (Bastianelli et al., 2020) -- the primary slot-filling metric used from here on
    # (checkpoint selection, comparison tables, WER-bucket breakdowns). See compute_slu_f1.
    slu = compute_slu_f1(all_slot_true_tags, all_slot_pred_tags, all_slot_words)
    metrics["word_f1"] = slu["word_f1"]["f1"]
    metrics["word_f1_precision"] = slu["word_f1"]["precision"]
    metrics["word_f1_recall"] = slu["word_f1"]["recall"]
    metrics["char_f1"] = slu["char_f1"]["f1"]
    metrics["char_f1_precision"] = slu["char_f1"]["precision"]
    metrics["char_f1_recall"] = slu["char_f1"]["recall"]
    metrics["slu_f1"] = slu["slu_f1"]["f1"]
    metrics["slu_f1_precision"] = slu["slu_f1"]["precision"]
    metrics["slu_f1_recall"] = slu["slu_f1"]["recall"]
    # Retained purely as a secondary diagnostic: the OLD strict exact-span-match metric (no
    # partial credit for filler mismatches). No longer used for checkpoint selection,
    # comparisons, or WER-bucket breakdowns -- SLU-F1 (above) replaces it there.
    metrics.update({f"exact_span_{k}": v for k, v in _slot_span_metrics(all_slot_true_tags, all_slot_pred_tags).items()})

    if wer_lookup is not None:
        by_wer: Dict[str, Dict[str, float]] = {}
        for label, lo, hi in _WER_BUCKETS:
            idxs = [i for i, p in enumerate(all_paths) if lo <= wer_lookup.get(p, 1.0) < hi]
            if not idxs:
                continue
            bucket_intent_true = [all_intent_true[i] for i in idxs]
            bucket_intent_pred = [all_intent_pred[i] for i in idxs]
            bucket_true_tags = [all_slot_true_tags[i] for i in idxs]
            bucket_pred_tags = [all_slot_pred_tags[i] for i in idxs]
            bucket_words = [all_slot_words[i] for i in idxs]
            bucket_metrics = {"n": len(idxs)}
            bucket_metrics.update(_intent_metrics(bucket_intent_true, bucket_intent_pred))
            bucket_slu = compute_slu_f1(bucket_true_tags, bucket_pred_tags, bucket_words)
            bucket_metrics["slu_f1"] = bucket_slu["slu_f1"]["f1"]
            bucket_metrics["word_f1"] = bucket_slu["word_f1"]["f1"]
            bucket_metrics["char_f1"] = bucket_slu["char_f1"]["f1"]
            by_wer[label] = bucket_metrics
        metrics["by_wer"] = by_wer

    if per_class_report:
        print_intent_per_class_report(all_intent_true, all_intent_pred, vocab, top_n=per_class_report_top_n)
        print_slot_per_class_report(all_slot_true_tags, all_slot_pred_tags, top_n=per_class_report_top_n)

    return metrics


def print_wer_bucket_table(by_wer: Dict[str, Dict[str, float]], title: str = ""):
    if title:
        print(f"\n--- {title} (metrics by ASR-hypothesis WER bucket) ---")
    header = f"{'wer_bucket':<24}{'n':>7}{'intent_acc':>12}{'slu_f1':>10}{'word_f1':>10}{'char_f1':>10}"
    print(header)
    print("-" * len(header))
    for label, m in by_wer.items():
        print(f"{label:<24}{m['n']:>7}{m['intent_acc']:>12.4f}{m['slu_f1']:>10.4f}{m['word_f1']:>10.4f}{m['char_f1']:>10.4f}")


# =========================================================================== #
# 12b. ASR WER diagnostic (quantifies how much signal the text branch is
#      actually getting -- critical to check before tuning anything else at
#      severe SNRs, since the ASR hypothesis feeding the text branch is never
#      fine-tuned in this pipeline: it's a frozen, off-the-shelf Whisper
#      decode, cached once. If its WER is catastrophic, no amount of
#      downstream training/hyperparameter tuning can fix garbage input.)
#
#      NOTE: in "joint" mode this is still useful, but the cached
#      slurp_asr_cache.json it reads gets OVERWRITTEN by refresh_asr_and_slot_caches()
#      during training -- re-run this after training finishes to see whether the
#      real, live WER actually dropped as the encoder adapted.
# =========================================================================== #
def _word_error_rate(ref_words: List[str], hyp_words: List[str]) -> float:
    """Standard word error rate: word-level Levenshtein distance / len(ref_words).
    Can exceed 1.0 if the hypothesis has many insertions."""
    n, m = len(ref_words), len(hyp_words)
    if n == 0:
        return 0.0 if m == 0 else 1.0
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1].lower() == hyp_words[j - 1].lower() else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m] / n


def compute_asr_wer_report(
    cfg: SLURPConfig,
    splits: Optional[List[str]] = None,
    save_csv_path: str = "slurp_asr_wer_report.csv",
    save_hist_path: str = "slurp_asr_wer_histogram.png",
) -> Dict[str, float]:
    """
    Standalone diagnostic: compares the cached ASR hypotheses (`cfg.asr_cache_path`,
    built by ASRTranscriber -- an off-the-shelf, NEVER fine-tuned Whisper decode, OR, in
    joint mode, periodically refreshed by the live adapting model -- see
    refresh_asr_and_slot_caches) against the SLURP ground-truth sentences, and reports
    the word error rate distribution.

    Run this BEFORE tuning hyperparameters when results are poor at a severe SNR. If mean/
    median WER is high (rule of thumb: >50-60%), the text branch's input is close to random
    relative to what it's meant to represent, and the bottleneck is ASR quality, not model
    architecture or learning rate. In that case: fine-tune/upgrade the ASR model (see
    whisper_finetune_mode="joint"), or lean on fusion_mode="audio_query"/"dual_branch" so
    the intent head doesn't depend on this text at all (see evaluate(..., wer_lookup=...)
    for a way to directly measure whether that actually helps, bucketed by how bad each
    utterance's ASR was).

    Returns a dict of {wav_rel_path: wer} which can be passed straight into
    evaluate(..., wer_lookup=...) for WER-bucketed accuracy/F1 breakdowns.
    """
    splits = splits or ["train", "valid", "test"]
    split_jsonl = {
        "train": (os.path.join(cfg.root_dir, cfg.train_jsonl), cfg.audio_real_subdir),
        "valid": (os.path.join(cfg.root_dir, cfg.valid_jsonl), cfg.audio_real_subdir),
        "test": (os.path.join(cfg.root_dir, cfg.test_jsonl), cfg.audio_real_subdir),
    }

    if not os.path.exists(cfg.asr_cache_path):
        raise FileNotFoundError(
            f"{cfg.asr_cache_path} not found. Run build_dataloaders(cfg) at least once first "
            f"(with use_ground_truth_transcript=False) to build the ASR cache, then re-run this."
        )
    with open(cfg.asr_cache_path) as f:
        asr_cache: Dict[str, str] = json.load(f)

    per_path_wer: Dict[str, float] = {}
    per_split_wers: Dict[str, List[float]] = {}
    n_empty_hyp = 0

    for split in splits:
        jsonl_path, audio_subdir = split_jsonl[split]
        if not os.path.exists(jsonl_path):
            continue
        wers = []
        for entry in _read_jsonl(jsonl_path):
            ref_words = entry["sentence"].split()
            for rec in entry.get("recordings", []):
                if cfg.recording_filter == "correct_only" and rec.get("status") != "correct":
                    continue
                wav_rel_path = os.path.join(audio_subdir, rec["file"])
                hyp_text = asr_cache.get(wav_rel_path)
                if hyp_text is None:
                    continue   # not decoded (e.g. excluded as a corrupt/bad file)
                hyp_words = hyp_text.split()
                if not hyp_words:
                    n_empty_hyp += 1
                wer = _word_error_rate(ref_words, hyp_words)
                per_path_wer[wav_rel_path] = wer
                wers.append(wer)
        per_split_wers[split] = wers

    all_wers = [w for ws in per_split_wers.values() for w in ws]
    if not all_wers:
        print("[wer-report] no cached ASR hypotheses found for the requested splits.")
        return per_path_wer

    all_wers_sorted = sorted(all_wers)
    n = len(all_wers_sorted)
    mean_wer = sum(all_wers_sorted) / n
    median_wer = all_wers_sorted[n // 2]
    pct_above_50 = sum(1 for w in all_wers_sorted if w >= 0.5) / n
    pct_above_75 = sum(1 for w in all_wers_sorted if w >= 0.75) / n
    pct_at_or_above_100 = sum(1 for w in all_wers_sorted if w >= 1.0) / n

    print(f"\n{'=' * 60}\nASR WER report (cached hypotheses vs. ground truth)\n{'=' * 60}")
    print(f"utterances analyzed:      {n}")
    print(f"empty hypotheses:         {n_empty_hyp} ({n_empty_hyp / n:.1%})")
    print(f"mean WER:                 {mean_wer:.3f}")
    print(f"median WER:               {median_wer:.3f}")
    print(f"min / max WER:            {all_wers_sorted[0]:.3f} / {all_wers_sorted[-1]:.3f}")
    print(f"% utterances WER >= 0.50: {pct_above_50:.1%}")
    print(f"% utterances WER >= 0.75: {pct_above_75:.1%}")
    print(f"% utterances WER >= 1.00: {pct_at_or_above_100:.1%}  (hypothesis as bad as random/empty)")
    for split, ws in per_split_wers.items():
        if ws:
            print(f"  [{split}] n={len(ws)} mean_wer={sum(ws) / len(ws):.3f}")

    if mean_wer >= 0.5:
        print("\n[verdict] Mean WER is very high. The text branch's input is largely uncorrelated "
              "with the true sentence at this SNR -- this is very likely your main bottleneck, not "
              "hyperparameters. Consider: whisper_finetune_mode='joint' (fine-tune the ASR encoder "
              "jointly with SLU against ground-truth transcripts), or fusion_mode="
              "'audio_query'/'dual_branch' so intent doesn't depend on this text at all.")
    elif mean_wer >= 0.25:
        print("\n[verdict] Moderate WER. The text branch carries a noticeably degraded signal; "
              "worth comparing fusion_mode='dual_branch' against the current mode to see whether "
              "decoupling intent from text recovers accuracy.")
    else:
        print("\n[verdict] WER is relatively low -- the ASR/text branch is probably not your main "
              "bottleneck. Look at the acoustic side (whisper_finetune_mode) instead.")

    with open(save_csv_path, "w") as f:
        f.write("wav_rel_path,wer\n")
        for p, w in per_path_wer.items():
            f.write(f"{p},{w:.4f}\n")
    print(f"\nper-utterance WER saved to {save_csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4))
        plt.hist(all_wers, bins=40, range=(0, max(2.0, max(all_wers))))
        plt.xlabel("word error rate")
        plt.ylabel("count")
        plt.title(f"ASR hypothesis WER distribution (n={n}, mean={mean_wer:.2f})")
        plt.axvline(mean_wer, color="red", linestyle="--", label=f"mean={mean_wer:.2f}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_hist_path, dpi=150)
        plt.close()
        print(f"WER histogram saved to {save_hist_path}")
    except ImportError:
        print("[wer-report] matplotlib not available -- skipping histogram plot (CSV was still saved).")

    return per_path_wer


# =========================================================================== #
# 12c. Class weights (long-tail intents/slots -- see SLURPConfig.use_class_weights)
# =========================================================================== #
def compute_class_weights(
    examples: List[dict],   # SLURPDataset.examples: each has intent_id, tags
    num_intents: int,
    slot2id: Dict[str, int],
    cap: float = 10.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inverse-frequency class weights from the TRAINING split only (weighting from
    valid/test would leak split-specific info into the loss). Weights are normalized to
    mean 1.0 and capped at `cap` so a handful of near-singleton classes (SLURP's slot
    label set has several very rare entity types) don't dominate the gradient."""
    intent_counts = [0] * num_intents
    slot_counts = [0] * len(slot2id)
    for ex in examples:
        intent_counts[ex["intent_id"]] += 1
        for t in ex["tags"]:
            slot_counts[slot2id.get(t, slot2id["O"])] += 1

    def _weights_from_counts(counts: List[int]) -> torch.Tensor:
        counts = [c if c > 0 else 1 for c in counts]   # avoid div-by-zero for unseen classes
        total = sum(counts)
        n_classes = len(counts)
        raw = [total / (n_classes * c) for c in counts]
        mean_w = sum(raw) / len(raw)
        capped = [min(w / mean_w, cap) for w in raw]
        return torch.tensor(capped, dtype=torch.float)

    intent_weights = _weights_from_counts(intent_counts)
    slot_weights = _weights_from_counts(slot_counts)
    print(f"[class-weights] intent weights: min={intent_weights.min():.2f} max={intent_weights.max():.2f} "
          f"(capped at {cap}); slot weights: min={slot_weights.min():.2f} max={slot_weights.max():.2f}")
    return intent_weights, slot_weights


# =========================================================================== #
# 12d. JOINT-ASR periodic hypothesis / slot-tag refresh
# =========================================================================== #
def refresh_asr_and_slot_caches(
    model: nn.Module,
    all_recordings: List[dict],   # each has wav_rel_path, words, tags, sentence (mutated: adds "asr_text")
    root_dir: str,
    sample_rate: int,
    feature_extractor: WhisperFeatureExtractor,
    asr_cache: Dict[str, str],              # mutated IN PLACE
    slot_tag_cache: Dict[str, List[str]],   # mutated IN PLACE
    device: str,
    batch_size: int = 16,
    num_workers: int = 4,
    asr_cache_path: Optional[str] = None,
    slot_tag_cache_path: Optional[str] = None,
) -> None:
    """JOINT-ASR: regenerates ASR hypotheses for every recording using the model's CURRENT,
    adapting acoustic_encoder weights (not a frozen baseline), then re-runs the word-level
    alignment against those fresh hypotheses to update the slot-tag labels used for training.

    Both `asr_cache` and `slot_tag_cache` are mutated IN PLACE (not replaced) so every
    SLURPDataset instance sharing these dict objects (train/valid/test all receive the SAME
    object via build_dataloaders' common_kwargs) sees the update automatically on the next
    __getitem__ call -- no need to rebuild the Dataset objects.

    IMPORTANT: DataLoader `persistent_workers=True` pickles a snapshot of the dataset (and
    hence these dicts) into long-lived worker processes at DataLoader-construction time, so
    in-place mutation here would NOT be visible to already-running persistent workers.
    build_dataloaders() disables persistent_workers automatically whenever
    whisper_finetune_mode == "joint" so this refresh takes effect correctly every epoch
    (workers are simply re-forked, picking up the current dict contents, each time a
    DataLoader is iterated)."""
    assert isinstance(model.acoustic_encoder, JointWhisperASR), \
        "refresh_asr_and_slot_caches requires a JointWhisperASR acoustic_encoder (whisper_finetune_mode='joint')"

    all_paths = [r["wav_rel_path"] for r in all_recordings]
    print(f"[joint-asr-refresh] regenerating ASR hypotheses for {len(all_paths)} recordings "
          f"using the current (adapting) model weights...")

    was_training = model.training
    model.eval()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        full_paths = [os.path.join(root_dir, p) for p in all_paths]
        futures = [executor.submit(_load_and_resample_safe, fp, sample_rate) for fp in full_paths]

        for i in range(0, len(all_paths), batch_size):
            batch_paths = all_paths[i:i + batch_size]
            batch_futures = futures[i:i + batch_size]
            results = [f.result() for f in batch_futures]
            ok_paths = [p for p, w in zip(batch_paths, results) if w is not None]
            waveforms = [w for w in results if w is not None]
            if not waveforms:
                continue
            inputs = feature_extractor(waveforms, sampling_rate=sample_rate, return_tensors="pt")
            input_features = inputs.input_features.to(device)
            hyps = model.acoustic_encoder.generate_transcripts(input_features)
            for p, h in zip(ok_paths, hyps):
                asr_cache[p] = h
            if (i // batch_size) % 20 == 0:
                print(f"  ...{i + len(batch_paths)}/{len(all_paths)} hypotheses refreshed")
    if was_training:
        model.train()

    if asr_cache_path:
        with open(asr_cache_path, "w") as f:
            json.dump(asr_cache, f)

    print("[joint-asr-refresh] re-aligning slot tags against refreshed hypotheses...")
    for r in all_recordings:
        r["asr_text"] = asr_cache.get(r["wav_rel_path"], r["sentence"])
    args = [(r["words"], r["tags"], r["asr_text"].split() or ["<empty>"]) for r in all_recordings]
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(_align_worker, args, chunksize=64))
    slot_tag_cache.clear()
    for r, tags in zip(all_recordings, results):
        slot_tag_cache[r["wav_rel_path"]] = tags

    if slot_tag_cache_path:
        with open(slot_tag_cache_path, "w") as f:
            json.dump(slot_tag_cache, f)

    print("[joint-asr-refresh] done.")


# =========================================================================== #
# 13. Data pipeline
# =========================================================================== #
def build_dataloaders(cfg: SLURPConfig) -> Tuple[DataLoader, DataLoader, DataLoader, SLURPLabelVocab, Dict[str, torch.Tensor], Optional[Dict]]:
    """Returns (train_loader, valid_loader, test_loader, vocab, class_weights, joint_refresh_ctx).

    `joint_refresh_ctx` is None unless cfg.whisper_finetune_mode == "joint", in which case it's
    a dict with everything refresh_asr_and_slot_caches() needs to be called again between
    epochs: {"all_recordings", "root_dir", "sample_rate", "feature_extractor", "asr_cache",
    "slot_tag_cache", "asr_cache_path", "slot_tag_cache_path"}.
    """
    is_joint = cfg.whisper_finetune_mode == "joint"

    train_jsonl = os.path.join(cfg.root_dir, cfg.train_jsonl)
    valid_jsonl = os.path.join(cfg.root_dir, cfg.valid_jsonl)
    test_jsonl = os.path.join(cfg.root_dir, cfg.test_jsonl)
    synth_jsonl = os.path.join(cfg.root_dir, cfg.train_synthetic_jsonl)

    vocab_source_paths = [train_jsonl, valid_jsonl, test_jsonl]
    if cfg.include_synthetic and os.path.exists(synth_jsonl):
        vocab_source_paths.append(synth_jsonl)
    vocab = SLURPLabelVocab(vocab_source_paths)
    vocab.save("slurp_label_vocab.json")
    print(f"[vocab] intents={vocab.num_intents} slot_labels={vocab.num_slot_labels}")

    # --- gather (wav_rel_path, words, tags, sentence) for every recording up front ---
    all_jsonl_specs = [(train_jsonl, cfg.audio_real_subdir),
                        (valid_jsonl, cfg.audio_real_subdir),
                        (test_jsonl, cfg.audio_real_subdir)]
    if cfg.include_synthetic and os.path.exists(synth_jsonl):
        all_jsonl_specs.append((synth_jsonl, cfg.audio_synth_subdir))

    all_recordings = []   # list of dicts: wav_rel_path, words, tags, sentence
    for jp, subdir in all_jsonl_specs:
        for entry in _read_jsonl(jp):
            if entry["intent"] not in vocab.intent2id:
                continue
            words, tags = parse_sentence_annotation(entry.get("sentence_annotation", entry["sentence"]))
            for rec in entry.get("recordings", []):
                if cfg.recording_filter == "correct_only" and rec.get("status") != "correct":
                    continue
                all_recordings.append({
                    "wav_rel_path": os.path.join(subdir, rec["file"]),
                    "words": words,
                    "tags": tags,
                    "sentence": entry["sentence"],
                })

    all_paths = [r["wav_rel_path"] for r in all_recordings]

    # --- 0) validate audio up front: one bad/corrupt file must never be able to crash a
    # parallel caching pass or a DataLoader worker mid-training. Excluded permanently from
    # every split below; verdicts are cached to disk so repeat runs skip re-checking. ---
    bad_files = validate_audio_files(
        all_paths, cfg.root_dir, cfg.sample_rate,
        cache_path="slurp_audio_validity_cache.json", num_workers=cfg.prep_num_workers,
    )
    if bad_files:
        all_recordings = [r for r in all_recordings if r["wav_rel_path"] not in bad_files]
        all_paths = [r["wav_rel_path"] for r in all_recordings]

    # --- 1) ASR transcript cache (parallel decode) ---
    # `cfg.asr_engine` picks WHICH model produces these cached hypotheses; it is independent
    # of `cfg.whisper_finetune_mode`, which controls the separate, trainable acoustic-embedding
    # path. See SLURPConfig.asr_engine / FunASRTranscriber docstrings.
    asr_cache = {}
    if not cfg.use_ground_truth_transcript:
        if cfg.asr_engine == "funasr_nano":
            # Fun-ASR-Nano-2512 is inference-only (no training support -- see FunASRTranscriber
            # docstring), so it's always used the same way regardless of whisper_finetune_mode:
            # build a full, static cache up front. There is nothing to "warm start" here since
            # this engine doesn't adapt during training.
            transcriber = FunASRTranscriber(
                cfg.funasr_model_name, hub=cfg.funasr_hub, device=cfg.funasr_device,
                language=cfg.funasr_language, itn=cfg.funasr_itn,
            )
            asr_cache = transcriber.build_cache(
                all_paths, cfg.root_dir, cfg.asr_cache_path, batch_size=cfg.prep_batch_size,
            )
            del transcriber
        elif is_joint:
            # JOINT-ASR (whisper engine only): do NOT spin up a separate ASRTranscriber (a
            # second full Whisper copy) just to build an initial cache. SLURPDataset.__getitem__
            # already falls back to the ground-truth sentence whenever a path is missing from
            # asr_cache (see `self.asr_cache.get(wav_rel_path, ex["sentence"])`), so leaving
            # asr_cache empty here means epoch 0 trains with PERFECT hypotheses/slot-tag
            # alignment as a warm start -- a reasonable and cheap way to begin. The first
            # end-of-epoch refresh (see train_model) then replaces these with the model's own,
            # adapting generate() output for epoch 1 onward.
            print("[data] whisper_finetune_mode='joint' with asr_engine='whisper': skipping "
                  "standalone ASRTranscriber cache build. Training starts with ground-truth "
                  "transcripts as a warm start; real model-generated ASR hypotheses populate "
                  "after the first asr_refresh_every_n_epochs epoch(s) (see "
                  "refresh_asr_and_slot_caches).")
        else:
            transcriber = ASRTranscriber(cfg.whisper_model_name, cfg.device)
            asr_cache = transcriber.build_cache(
                all_paths, cfg.root_dir, cfg.sample_rate, cfg.asr_cache_path,
                batch_size=cfg.prep_batch_size, num_workers=cfg.prep_num_workers,
            )
            del transcriber
            if cfg.device == "cuda":
                torch.cuda.empty_cache()

    feature_extractor = WhisperFeatureExtractor.from_pretrained(cfg.whisper_model_name, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.bert_model_name, local_files_only=True)

    # JOINT-ASR: Whisper's OWN tokenizer, needed by the collator to build teacher-forcing
    # `labels` for the ASR loss from ground-truth transcripts. Loading just the tokenizer
    # (not the full model) is cheap -- the actual Whisper weights are already owned by
    # model.acoustic_encoder (JointWhisperASR) and are not duplicated here. Unaffected by
    # asr_engine: the ASR loss is always computed against Whisper's own tokenization of the
    # ground-truth sentence, regardless of which engine produced the cached hypotheses.
    whisper_tokenizer = None
    if is_joint:
        whisper_tokenizer = WhisperProcessor.from_pretrained(cfg.whisper_model_name, local_files_only=True).tokenizer

    # --- 2) frozen acoustic feature cache -- only valid when the encoder is fully frozen,
    # since that's the only mode where its output for a given waveform never changes
    # during training. ---
    if cfg.use_precomputed_acoustic_features and cfg.whisper_finetune_mode == "frozen":
        cache_encoder = WhisperAcousticEncoder(
            cfg.whisper_model_name, cfg.whisper_layer, finetune_mode="frozen",
        ).to(cfg.device)
        build_acoustic_feature_cache(
            cache_encoder, feature_extractor, all_paths, cfg.root_dir, cfg.sample_rate,
            cfg.acoustic_feature_cache_dir, cfg.device,
            batch_size=cfg.prep_batch_size, num_workers=cfg.prep_num_workers,
        )
        del cache_encoder
        if cfg.device == "cuda":
            torch.cuda.empty_cache()
    elif cfg.use_precomputed_acoustic_features and cfg.whisper_finetune_mode != "frozen":
        print(f"[warn] use_precomputed_acoustic_features=True but whisper_finetune_mode="
              f"'{cfg.whisper_finetune_mode}' -- acoustic encoder is (partially) trainable, so its "
              f"output can't be cached. Disabling feature cache; training will decode audio + run "
              f"the encoder fresh every step (slower, but necessary for the encoder to adapt).")
        cfg.use_precomputed_acoustic_features = False

    # --- 3) slot-tag alignment cache ---
    slot_tag_cache = {}
    if not cfg.use_ground_truth_transcript:
        for r in all_recordings:
            r["asr_text"] = asr_cache.get(r["wav_rel_path"], r["sentence"])
        if is_joint and cfg.asr_engine == "whisper" and not asr_cache:
            # Ground-truth warm-start case ONLY (see step 1): asr_cache is genuinely empty, so
            # every r["asr_text"] just equals r["sentence"] here. Still must go through proper
            # alignment rather than assigning r["tags"] directly -- asr_text.split() (whitespace
            # split on the raw sentence) is not guaranteed to be identical, word-for-word, to
            # r["words"] from parse_sentence_annotation() (punctuation handling can differ), so
            # skipping alignment would silently stamp tags onto the wrong words. Cheap here since
            # alignment against a near-identical hypothesis converges almost immediately.
            args = [(r["words"], r["tags"], r["asr_text"].split() or ["<empty>"]) for r in all_recordings]
            with ProcessPoolExecutor(max_workers=cfg.prep_num_workers) as executor:
                results = list(executor.map(_align_worker, args, chunksize=64))
            for r, tags in zip(all_recordings, results):
                slot_tag_cache[r["wav_rel_path"]] = tags
        else:
            # Covers: non-joint modes (any asr_engine), AND joint mode with asr_engine=
            # "funasr_nano" (asr_cache is already populated with real hypotheses from step 1,
            # so this must align against THOSE, not assume a ground-truth warm start).
            slot_tag_cache = build_slot_tag_cache(all_recordings, cfg.slot_tag_cache_path, cfg.prep_num_workers)

    collator = SLURPCollator(
        feature_extractor, tokenizer, vocab.slot2id,
        sample_rate=cfg.sample_rate, max_text_len=cfg.max_text_len,
        whisper_tokenizer=whisper_tokenizer,
    )

    common_kwargs = dict(
        root_dir=cfg.root_dir, vocab=vocab, sample_rate=cfg.sample_rate,
        max_audio_seconds=cfg.max_audio_seconds, asr_cache=asr_cache, slot_tag_cache=slot_tag_cache,
        use_ground_truth_transcript=cfg.use_ground_truth_transcript,
        recording_filter=cfg.recording_filter, recording_type_filter=cfg.recording_type_filter,
        use_precomputed_acoustic_features=cfg.use_precomputed_acoustic_features,
        acoustic_feature_cache_dir=cfg.acoustic_feature_cache_dir,
        bad_files=bad_files,
    )

    train_jsonl_paths = [(train_jsonl, cfg.audio_real_subdir)]
    if cfg.include_synthetic and os.path.exists(synth_jsonl):
        train_jsonl_paths.append((synth_jsonl, cfg.audio_synth_subdir))

    train_ds = SLURPDataset(train_jsonl_paths, **common_kwargs)
    valid_ds = SLURPDataset([(valid_jsonl, cfg.audio_real_subdir)], **common_kwargs)
    test_ds = SLURPDataset([(test_jsonl, cfg.audio_real_subdir)], **common_kwargs)
    print(f"[data] train={len(train_ds)} valid={len(valid_ds)} test={len(test_ds)} examples (per-recording)")

    class_weights: Dict[str, torch.Tensor] = {}
    if cfg.use_class_weights:
        intent_w, slot_w = compute_class_weights(
            train_ds.examples, vocab.num_intents, vocab.slot2id, cap=cfg.class_weight_cap,
        )
        class_weights = {"intent": intent_w, "slot": slot_w}

    # PERF: persistent_workers avoids re-spawning worker processes every epoch;
    # pin_memory speeds up host->GPU transfer. Worker cost is now trivial
    # (load one small cached tensor + tokenize text) since audio decode /
    # Whisper-encoder forward / tag alignment all happened once above.
    #
    # JOINT-ASR EXCEPTION: persistent_workers=True pickles a snapshot of `asr_cache` and
    # `slot_tag_cache` into long-lived worker processes at DataLoader-construction time.
    # refresh_asr_and_slot_caches() mutates those dicts IN PLACE between epochs, and
    # already-running persistent workers would never see the update -- they'd keep using
    # their stale, construction-time snapshot for the rest of training. So persistent_workers
    # is forced off whenever whisper_finetune_mode == "joint"; workers are then re-forked
    # (and re-pickle the current dict contents) every time a DataLoader is iterated, which is
    # what makes each epoch actually see the previous epoch's refreshed hypotheses/tags.
    dl_kwargs = dict(num_workers=cfg.num_workers, collate_fn=collator, pin_memory=(cfg.device == "cuda"))
    if cfg.num_workers > 0 and not is_joint:
        dl_kwargs["persistent_workers"] = True
    elif is_joint and cfg.num_workers > 0:
        print("[data] whisper_finetune_mode='joint': disabling persistent_workers so periodic "
              "ASR/slot-tag cache refreshes are visible to DataLoader workers every epoch.")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, **dl_kwargs)
    valid_loader = DataLoader(valid_ds, batch_size=cfg.batch_size, shuffle=False, **dl_kwargs)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, **dl_kwargs)

    joint_refresh_ctx = None
    if is_joint and cfg.asr_engine == "whisper":
        joint_refresh_ctx = dict(
            all_recordings=all_recordings, root_dir=cfg.root_dir, sample_rate=cfg.sample_rate,
            feature_extractor=feature_extractor, asr_cache=asr_cache, slot_tag_cache=slot_tag_cache,
            asr_cache_path=cfg.asr_cache_path, slot_tag_cache_path=cfg.slot_tag_cache_path,
        )
    elif is_joint and cfg.asr_engine == "funasr_nano":
        print("[data] whisper_finetune_mode='joint' with asr_engine='funasr_nano': the ASR "
              "hypothesis engine (FunASR) is not trained, so there is nothing to periodically "
              "refresh -- the text branch and slot-tag cache stay fixed at FunASR's output for "
              "the whole run. Whisper's encoder still adapts via its own ASR loss against "
              "ground truth; only the TEXT-BRANCH INPUT is now fixed instead of co-adapting.")

    return train_loader, valid_loader, test_loader, vocab, class_weights, joint_refresh_ctx


# =========================================================================== #
# 14. Training loop
# =========================================================================== #
def build_model(
    cfg: SLURPConfig, vocab: SLURPLabelVocab, fusion_mode: Optional[str] = None,
    class_weights: Optional[Dict[str, torch.Tensor]] = None,
) -> nn.Module:
    mode = fusion_mode or cfg.fusion_mode
    class_weights = class_weights or {}
    return SemiCascadedSLU_SLURP(
        num_intents=vocab.num_intents,
        num_slot_labels=vocab.num_slot_labels,
        whisper_model_name=cfg.whisper_model_name,
        bert_model_name=cfg.bert_model_name,
        whisper_layer=cfg.whisper_layer,
        whisper_finetune_mode=cfg.whisper_finetune_mode,
        whisper_unfreeze_last_n_layers=cfg.whisper_unfreeze_last_n_layers,
        adapter_bottleneck_dim=cfg.adapter_bottleneck_dim,
        adapter_dropout=cfg.adapter_dropout,
        freeze_bert=False,
        fusion_mode=mode,
        intent_class_weights=class_weights.get("intent"),
        slot_class_weights=class_weights.get("slot"),
        loss_type=cfg.loss_type,
        focal_gamma=cfg.focal_gamma,
        # JOINT-ASR params -- inert (unused) unless whisper_finetune_mode == "joint"
        joint_encoder_finetune_mode=cfg.joint_encoder_finetune_mode,
        asr_decoder_trainable=cfg.asr_decoder_trainable,
        asr_loss_weight=cfg.asr_loss_weight,
    ).to(cfg.device)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    test_loader: DataLoader,
    vocab: SLURPLabelVocab,
    cfg: SLURPConfig,
    run_tag: str = "",
    wer_lookup: Optional[Dict[str, float]] = None,
    joint_refresh_ctx: Optional[Dict] = None,
) -> Dict[str, float]:
    """Runs the epoch loop for an already-built model and returns final test metrics.
    Split out from `train()` so the same cached dataloaders can be reused to train
    several fusion_mode variants back-to-back without rebuilding ASR/feature/tag
    caches each time -- see `run_comparison`.

    If `wer_lookup` is supplied (from compute_asr_wer_report), the final test evaluation
    is also broken out by ASR-hypothesis WER bucket -- useful for seeing whether a given
    fusion_mode holds up better specifically on the noisiest/most-garbled-transcript
    utterances, which is exactly what severe-SNR conditions (e.g. 0dB) stress.

    If `joint_refresh_ctx` is supplied (only when cfg.whisper_finetune_mode == "joint",
    from build_dataloaders), ASR hypotheses and slot-tag alignments are regenerated using
    the model's own, currently-adapting weights every `cfg.asr_refresh_every_n_epochs`
    epochs -- see refresh_asr_and_slot_caches. This is what lets the text branch and slot
    labels actually track the encoder's improving noise-robustness over the course of
    training, instead of staying pinned to a single fixed (ground-truth-warm-start or
    pre-training) snapshot."""
    ckpt_name = f"slurp_semi_cascaded_slu_best_{run_tag}.pt" if run_tag else "slurp_semi_cascaded_slu_best.pt"
    is_joint = cfg.whisper_finetune_mode == "joint"
    if is_joint and joint_refresh_ctx is None:
        print("[warn] whisper_finetune_mode='joint' but joint_refresh_ctx is None -- ASR hypotheses "
              "and slot tags will NEVER be refreshed during training (they'll stay at whatever "
              "build_dataloaders() initialized them to, i.e. the ground-truth warm start). Pass "
              "the joint_refresh_ctx returned by build_dataloaders() to get real periodic refresh.")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr)

    # PERF/quality: LR warmup + decay. A flat LR with no warmup tends to underfit a
    # freshly-initialized fusion module stacked on pretrained BERT/Whisper; gradient
    # accumulation reaches a larger effective batch size without more GPU memory.
    accum = max(1, cfg.grad_accum_steps)
    optim_steps_per_epoch = math.ceil(len(train_loader) / accum)
    epochs_to_run = cfg.epochs
    total_optim_steps = optim_steps_per_epoch * epochs_to_run

    # Guard against silently undertraining: grad_accum_steps reduces optimizer updates per
    # epoch by that same factor. If the resulting total update count falls below the floor,
    # bump epochs up to compensate instead of quietly training with fewer updates than
    # cfg.epochs alone would suggest to someone reading the config.
    if cfg.min_total_optimizer_steps is not None and total_optim_steps < cfg.min_total_optimizer_steps:
        needed_epochs = math.ceil(cfg.min_total_optimizer_steps / optim_steps_per_epoch)
        print(f"[train] NOTE: grad_accum_steps={accum} means only {optim_steps_per_epoch} optimizer "
              f"updates/epoch. cfg.epochs={cfg.epochs} would give just {total_optim_steps} total updates "
              f"(< min_total_optimizer_steps={cfg.min_total_optimizer_steps}). Increasing epochs actually "
              f"run: {epochs_to_run} -> {needed_epochs}. Set cfg.min_total_optimizer_steps=None to disable "
              f"this and use cfg.epochs exactly as given.")
        epochs_to_run = needed_epochs
        total_optim_steps = optim_steps_per_epoch * epochs_to_run

    warmup_steps = int(cfg.warmup_ratio * total_optim_steps)
    scheduler = get_scheduler(
        cfg.lr_scheduler_type, optimizer=optimizer,
        num_warmup_steps=warmup_steps, num_training_steps=total_optim_steps,
    )
    print(f"[train] effective_batch_size={cfg.batch_size * accum} epochs={epochs_to_run} "
          f"optim_steps/epoch={optim_steps_per_epoch} total_optim_steps={total_optim_steps} "
          f"warmup_steps={warmup_steps} ({cfg.lr_scheduler_type} schedule) loss_type={cfg.loss_type} "
          f"joint_asr={'on (asr_loss_weight=' + str(cfg.asr_loss_weight) + ')' if is_joint else 'off'}")

    best_valid_f1 = -1.0
    for epoch in range(epochs_to_run):
        model.train()
        running_loss = 0.0
        running_asr_loss = 0.0
        optimizer.zero_grad()
        n_batches = len(train_loader)
        for step, batch in enumerate(train_loader):
            batch = {k: (v.to(cfg.device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out = _forward_batch(model, batch)
            out_loss = out["loss"]
            (out_loss / accum).backward()   # scale so accumulated grads match a single large-batch step

            is_last_in_epoch = (step + 1) == n_batches
            if (step + 1) % accum == 0 or is_last_in_epoch:
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            running_loss += out_loss.item()   # log the unscaled per-batch loss
            if "asr_loss" in out and out["asr_loss"] is not None:
                running_asr_loss += out["asr_loss"].item()
            if step % 50 == 0:
                tag = f"[{run_tag}] " if run_tag else ""
                asr_part = f" asr={out['asr_loss'].item():.4f}" if "asr_loss" in out and out["asr_loss"] is not None else ""
                print(f"{tag}epoch {epoch} step {step} loss {out_loss.item():.4f} lr={scheduler.get_last_lr()[0]:.2e} "
                      f"(intent={out['intent_loss'].item():.4f} slot={out['slot_loss'].item():.4f}{asr_part})")

        valid_metrics = evaluate(model, valid_loader, cfg.device, vocab)
        tag = f"[{run_tag}] " if run_tag else ""
        print(f"{tag}[epoch {epoch}] train_loss={running_loss / len(train_loader):.4f} valid={valid_metrics}")

        combined = 0.5 * valid_metrics["intent_acc"] + 0.5 * valid_metrics["slu_f1"]
        if combined > best_valid_f1:
            best_valid_f1 = combined
            torch.save(model.state_dict(), ckpt_name)
            print(f"{tag}  -> new best (intent_acc+slu_f1 avg={best_valid_f1:.4f}), checkpoint saved to {ckpt_name}")

        # JOINT-ASR: refresh ASR hypotheses / slot-tag alignment using the model's current,
        # adapted weights so subsequent epochs train the text branch and slot labels against
        # up-to-date (not stale/pretraining-era) transcripts. Skipped on the very last epoch
        # since there's no further training left to benefit from it.
        is_last_epoch = (epoch + 1) == epochs_to_run
        if is_joint and joint_refresh_ctx is not None and not is_last_epoch \
                and (epoch + 1) % max(1, cfg.asr_refresh_every_n_epochs) == 0:
            refresh_asr_and_slot_caches(
                model,
                all_recordings=joint_refresh_ctx["all_recordings"],
                root_dir=joint_refresh_ctx["root_dir"],
                sample_rate=joint_refresh_ctx["sample_rate"],
                feature_extractor=joint_refresh_ctx["feature_extractor"],
                asr_cache=joint_refresh_ctx["asr_cache"],
                slot_tag_cache=joint_refresh_ctx["slot_tag_cache"],
                device=cfg.device,
                batch_size=cfg.asr_refresh_batch_size,
                num_workers=cfg.prep_num_workers,
                asr_cache_path=joint_refresh_ctx["asr_cache_path"],
                slot_tag_cache_path=joint_refresh_ctx["slot_tag_cache_path"],
            )

    test_metrics = evaluate(model, test_loader, cfg.device, vocab, wer_lookup=wer_lookup,
                             per_class_report=cfg.print_per_class_report,
                             per_class_report_top_n=cfg.per_class_report_top_n)
    print(f"{tag}[final test] {test_metrics}")
    if wer_lookup is not None and "by_wer" in test_metrics:
        print_wer_bucket_table(test_metrics["by_wer"], title=f"{run_tag or 'model'} test set")
    return test_metrics


def train(cfg: SLURPConfig, wer_lookup: Optional[Dict[str, float]] = None):
    """Single-architecture training entry point (uses cfg.fusion_mode)."""
    train_loader, valid_loader, test_loader, vocab, class_weights, joint_refresh_ctx = build_dataloaders(cfg)
    model = build_model(cfg, vocab, class_weights=class_weights)
    test_metrics = train_model(model, train_loader, valid_loader, test_loader, vocab, cfg,
                                run_tag=cfg.fusion_mode, wer_lookup=wer_lookup,
                                joint_refresh_ctx=joint_refresh_ctx)
    return model, vocab, test_metrics


def run_comparison(
    cfg: SLURPConfig,
    fusion_modes: Optional[List[str]] = None,
    bucket_by_wer: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Trains each fusion_mode variant on identical data/splits (dataloaders and all
    caches -- ASR transcripts, frozen acoustic features, slot-tag alignments -- are
    built ONCE and reused across variants) and prints a comparison table at the end.

    When `bucket_by_wer=True` (default), also computes the ASR WER report once up front
    and prints a per-mode, per-WER-bucket breakdown -- this is the direct way to check
    whether e.g. "audio_query"/"dual_branch" actually degrade less than "text_query" on
    the utterances whose ASR transcript was most garbled, which is the concrete failure
    mode expected at severe SNRs like 0dB.

    JOINT-ASR CAVEAT: when cfg.whisper_finetune_mode == "joint", each fusion_mode variant
    trains its OWN separate JointWhisperASR weights, but they all share the SAME
    joint_refresh_ctx (asr_cache / slot_tag_cache dicts), since dataloaders are built once
    and reused. That means a later fusion_mode's training starts from whatever hypotheses
    the PREVIOUS fusion_mode's model last generated, not a clean ground-truth warm start --
    the comparison is then confounded by refresh history/order, not just architecture. If
    you need a clean apples-to-apples joint-ASR comparison across fusion_mode, call train()
    separately per mode (each gets its own fresh build_dataloaders() call) instead of
    run_comparison()."""
    modes = list(fusion_modes or cfg.comparison_modes)
    if cfg.whisper_finetune_mode == "joint" and len(modes) > 1:
        print("[warn] run_comparison() with whisper_finetune_mode='joint' and multiple fusion_modes: "
              "ASR hypothesis/slot-tag refresh state is SHARED and carries over between variants "
              "(see run_comparison docstring). Prefer calling train() separately per fusion_mode for "
              "a clean joint-ASR comparison.")

    train_loader, valid_loader, test_loader, vocab, class_weights, joint_refresh_ctx = build_dataloaders(cfg)

    wer_lookup = None
    if bucket_by_wer and not cfg.use_ground_truth_transcript and cfg.whisper_finetune_mode != "joint":
        # In joint mode the cache starts empty/ground-truth (see build_dataloaders), so a WER
        # report at this point would trivially show ~0 WER and isn't meaningful; run
        # compute_asr_wer_report(cfg) manually AFTER training if you want the live WER.
        try:
            wer_lookup = compute_asr_wer_report(cfg)
        except FileNotFoundError as e:
            print(f"[warn] skipping WER bucketing: {e}")

    results: Dict[str, Dict[str, float]] = {}
    for mode in modes:
        print(f"\n{'=' * 70}\n=== training fusion_mode={mode} ===\n{'=' * 70}")
        model = build_model(cfg, vocab, fusion_mode=mode, class_weights=class_weights)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[{mode}] trainable params: {n_params:,}")
        results[mode] = train_model(model, train_loader, valid_loader, test_loader, vocab, cfg,
                                     run_tag=mode, wer_lookup=wer_lookup, joint_refresh_ctx=joint_refresh_ctx)
        del model
        if cfg.device == "cuda":
            torch.cuda.empty_cache()

    print(f"\n{'=' * 70}\n=== comparison: {', '.join(modes)} ===\n{'=' * 70}")
    header = f"{'fusion_mode':<14}{'intent_acc':>12}{'intent_f1_macro':>18}{'slu_f1':>10}{'word_f1':>10}{'char_f1':>10}"
    print(header)
    print("-" * len(header))
    for mode in modes:
        m = results[mode]
        print(f"{mode:<14}{m['intent_acc']:>12.4f}{m['intent_f1_macro']:>18.4f}"
              f"{m['slu_f1']:>10.4f}{m['word_f1']:>10.4f}{m['char_f1']:>10.4f}")

    if wer_lookup is not None:
        print(f"\n{'=' * 70}\n=== comparison by ASR-hypothesis WER bucket ===\n{'=' * 70}")
        for label, _, _ in _WER_BUCKETS:
            print(f"\n[{label}]")
            row_header = f"{'fusion_mode':<14}{'n':>7}{'intent_acc':>12}{'slu_f1':>10}"
            print(row_header)
            print("-" * len(row_header))
            for mode in modes:
                bucket = results[mode].get("by_wer", {}).get(label)
                if bucket is None:
                    continue
                print(f"{mode:<14}{bucket['n']:>7}{bucket['intent_acc']:>12.4f}{bucket['slu_f1']:>10.4f}")

    best_mode = max(results, key=lambda k: 0.5 * results[k]["intent_acc"] + 0.5 * results[k]["slu_f1"])
    print(f"\nBest overall by 0.5*intent_acc + 0.5*slu_f1: {best_mode}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["compare", "train", "wer_report"], default="train",
                         help="'compare' trains all cfg.comparison_modes and prints a comparison "
                              "(incl. WER-bucketed breakdown; avoid combining with "
                              "whisper_finetune_mode='joint', see run_comparison docstring); "
                              "'train' trains just cfg.fusion_mode (recommended entry point for "
                              "joint ASR fine-tuning); 'wer_report' only runs the ASR WER "
                              "diagnostic (requires the ASR cache to already exist).")
    args = parser.parse_args()

    cfg = SLURPConfig(
        root_dir="slurp_dataset",
        use_ground_truth_transcript=False,
        recording_filter="correct_only",
        use_precomputed_acoustic_features=True,
        batch_size=8,
        epochs=10,

        include_synthetic=True,
        # --- JOINT-ASR fine-tuning: encoder adapts to noise using a real ASR loss against
        # ground-truth transcripts, sharing weights with (and therefore also improving) the
        # intent/slot heads. See JointWhisperASR / refresh_asr_and_slot_caches docstrings. ---
        whisper_finetune_mode="joint",
        joint_encoder_finetune_mode="adapters",   # safest: pretrained weights never touched
        asr_decoder_trainable=False,              # keep decoder frozen; encoder still gets ASR gradient
        asr_loss_weight=1.0,
        asr_refresh_every_n_epochs=1,
        asr_refresh_batch_size=16,

        fusion_mode="dual_branch",   # recommended alongside joint ASR: intent doesn't solely
                                      # depend on the (still-imperfect, even after adaptation) text branch

        # --- ASR hypothesis engine for the text branch (independent of whisper_finetune_mode
        # above, which controls the separate TRAINABLE acoustic-embedding path). Default stays
        # "whisper" for backward compatibility / the joint-warm-start behavior described above.
        # To use Fun-ASR-Nano-2512 instead (much stronger noise robustness, but inference-only
        # -- see FunASRTranscriber docstring), set:
        #   asr_engine="funasr_nano",
        #   funasr_model_name="FunAudioLLM/Fun-ASR-Nano-2512",
        #   funasr_language="英文",   # NOT "en" -- see SLURPConfig.funasr_language docstring
        # and `pip install -U funasr` first. With asr_engine="funasr_nano", the periodic
        # refresh in joint mode is automatically skipped (nothing to refresh -- see
        # build_dataloaders), so text-branch hypotheses stay fixed at FunASR's output for the
        # whole run while Whisper's encoder still adapts via its own ASR loss.
        asr_engine="whisper",

        warmup_ratio=0.06,
        lr_scheduler_type="linear",
        grad_accum_steps=4,
        use_class_weights=True,
        class_weight_cap=10.0,
        loss_type="focal",
        focal_gamma=2.0,
    )

    if args.mode == "wer_report":
        compute_asr_wer_report(cfg)
    elif args.mode == "train":
        train(cfg)
    else:
        run_comparison(cfg)
