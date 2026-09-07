"""
dataset.py
PyTorch Dataset for the SLURP corpus (https://github.com/pswietojanski/slurp).

Expected layout (standard SLURP release):

    slurp/
      dataset/slurp/{train,devel,test}.jsonl
      audio/slurp_real/*.flac
      audio/slurp_synth/*.flac

Each line of the jsonl files is one *sentence* record with a `recordings`
list (several speakers/mics reading the same sentence). This Dataset
flattens that into one example per audio recording.

Each SLURP record looks like:
{
  "slurp_id": 1234,
  "sentence": "wake me up at five am tomorrow",
  "sentence_annotation": "wake me up at [time : five am] [date : tomorrow]",
  "intent": "alarm_set",
  "action": "set",
  "scenario": "alarm",
  "recordings": [{"file": "audio-id.flac", ...}, ...]
}
"""
import json
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset

from vocab import annotation_to_tagged_text

TARGET_SR = 16000


class SlurpDataset(Dataset):
    def __init__(self, jsonl_path, audio_root, ctc_vocab, intent2id,
                 max_audio_seconds=15.0):
        self.audio_root = Path(audio_root)
        self.ctc_vocab = ctc_vocab
        self.intent2id = intent2id
        self.max_samples = int(max_audio_seconds * TARGET_SR)

        self.examples = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                scenario, action = rec.get("scenario"), rec.get("action")
                if scenario is None or action is None:
                    continue
                intent = rec.get("intent", f"{scenario}_{action}")
                if intent not in self.intent2id:
                    continue  # unseen intent (shouldn't happen if maps built from train)
                ann = rec.get("sentence_annotation") or rec.get("sentence", "")
                tagged_text = annotation_to_tagged_text(ann)
                ctc_ids = ctc_vocab.encode(tagged_text)
                for r in rec.get("recordings", []):
                    fname = r.get("file")
                    if fname is None:
                        continue
                    self.examples.append({
                        "audio_file": fname,
                        "intent_id": self.intent2id[intent],
                        "ctc_ids": ctc_ids,
                        "tagged_text": tagged_text,
                    })

    def __len__(self):
        return len(self.examples)

    def _find_audio(self, fname):
        # SLURP splits real/synth audio across two subdirectories.
        for sub in ("slurp_real", "slurp_synth"):
            candidate = self.audio_root / sub / fname
            if candidate.exists():
                return candidate
        # fall back: search directly under audio_root
        candidate = self.audio_root / fname
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Could not locate audio file: {fname}")

    def __getitem__(self, idx):
        ex = self.examples[idx]
        path = self._find_audio(ex["audio_file"])
        waveform, sr = torchaudio.load(str(path))
        if waveform.shape[0] > 1:  # downmix to mono
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != TARGET_SR:
            waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
        waveform = waveform.squeeze(0)
        if waveform.numel() > self.max_samples:
            waveform = waveform[: self.max_samples]

        return {
            "input_values": waveform,
            "intent_id": ex["intent_id"],
            "ctc_ids": torch.tensor(ex["ctc_ids"], dtype=torch.long),
            "tagged_text": ex["tagged_text"],
        }


def collate_fn(batch, pad_id):
    """Pads raw waveforms and CTC label sequences; returns attention mask
    for the audio encoder and lengths needed for CTC loss."""
    audio_lens = [b["input_values"].shape[0] for b in batch]
    max_audio_len = max(audio_lens)
    input_values = torch.zeros(len(batch), max_audio_len)
    attention_mask = torch.zeros(len(batch), max_audio_len, dtype=torch.long)
    for i, b in enumerate(batch):
        L = b["input_values"].shape[0]
        input_values[i, :L] = b["input_values"]
        attention_mask[i, :L] = 1

    ctc_lens = [b["ctc_ids"].shape[0] for b in batch]
    max_ctc_len = max(ctc_lens)
    ctc_targets = torch.full((len(batch), max_ctc_len), pad_id, dtype=torch.long)
    for i, b in enumerate(batch):
        L = b["ctc_ids"].shape[0]
        ctc_targets[i, :L] = b["ctc_ids"]

    intent_ids = torch.tensor([b["intent_id"] for b in batch], dtype=torch.long)

    return {
        "input_values": input_values,
        "attention_mask": attention_mask,
        "intent_ids": intent_ids,
        "ctc_targets": ctc_targets,
        "ctc_target_lens": torch.tensor(ctc_lens, dtype=torch.long),
        "tagged_texts": [b["tagged_text"] for b in batch],
    }
