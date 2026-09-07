"""
infer.py
Run inference on a single audio file with a trained checkpoint.

Usage:
    python infer.py --ckpt ckpt/best_model.pt --ckpt_dir ckpt \
        --wav path/to/utterance.wav --hubert_name facebook/hubert-base-ls960 \
        --semantic_layer 8
"""
import argparse
import json
from pathlib import Path

import torch
import torchaudio

from vocab import CTCVocab, extract_slots_from_tagged_text, strip_tags
from model import HubertSLUModel
from dataset import TARGET_SR


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--ckpt_dir", required=True, help="Dir with intent_labels.json / ctc_vocab.json")
    p.add_argument("--wav", required=True)
    p.add_argument("--hubert_name", default="facebook/hubert-base-ls960")
    p.add_argument("--semantic_layer", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_audio(path):
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
    return waveform  # (1, T)


def main():
    args = parse_args()
    ckpt_dir = Path(args.ckpt_dir)
    intent_list = json.loads((ckpt_dir / "intent_labels.json").read_text())
    ctc_vocab = CTCVocab.load(ckpt_dir / "ctc_vocab.json")

    model = HubertSLUModel(
        num_intents=len(intent_list),
        ctc_vocab_size=len(ctc_vocab),
        pad_id=ctc_vocab.pad_id,
        hubert_name=args.hubert_name,
        semantic_layer=args.semantic_layer,
    )
    model.set_blank_id(ctc_vocab.blank_id)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.to(args.device).eval()

    waveform = load_audio(args.wav).to(args.device)
    attention_mask = torch.ones(waveform.shape, dtype=torch.long, device=args.device)

    with torch.no_grad():
        outputs = model(waveform, attention_mask)

    intent_id = outputs["intent_logits"].argmax(-1).item()
    intent = intent_list[intent_id]

    greedy_ids = outputs["ctc_log_probs"].argmax(-1).transpose(0, 1)[0]
    L = int(outputs["feat_lens"][0].item())
    tagged_text = ctc_vocab.decode(greedy_ids[:L].tolist())
    transcript = strip_tags(tagged_text)
    slots = extract_slots_from_tagged_text(tagged_text)

    print("Intent   :", intent)
    print("Transcript:", transcript)
    print("Slots    :", slots)


if __name__ == "__main__":
    main()
