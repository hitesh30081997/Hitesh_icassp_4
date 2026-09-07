"""
train.py
End-to-end training for the HuBERT SLURP SLU model.

Usage:
    python train.py \
        --slurp_jsonl_dir /path/to/slurp/dataset/slurp \
        --audio_root /path/to/slurp/audio \
        --output_dir ./ckpt \
        --hubert_name facebook/hubert-base-ls960 \
        --semantic_layer 8 \
        --epochs 20 --batch_size 8 --lr 3e-5
"""
import argparse
import functools
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from vocab import CTCVocab, build_label_maps, extract_slots_from_tagged_text
from dataset import SlurpDataset, collate_fn
from model import HubertSLUModel
from metrics import intent_accuracy, slu_f1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--slurp_jsonl_dir", required=True,
                    help="Directory containing train.jsonl / devel.jsonl / test.jsonl")
    p.add_argument("--audio_root", required=True,
                    help="Directory containing slurp_real/ and slurp_synth/")
    p.add_argument("--output_dir", default="./ckpt")
    p.add_argument("--hubert_name", default="facebook/hubert-base-ls960")
    p.add_argument("--semantic_layer", type=int, default=8,
                    help="Index into HuBERT hidden_states to read out (1..num_layers)")
    p.add_argument("--freeze_feature_extractor", action="store_true", default=True)
    p.add_argument("--freeze_encoder_layers", type=int, default=0,
                    help="Freeze the first N Transformer blocks (<= semantic_layer makes sense)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--ctc_loss_weight", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_audio_seconds", type=float, default=15.0)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_or_load_vocabs(args, out_dir):
    intent_map_path = out_dir / "intent_labels.json"
    ctc_vocab_path = out_dir / "ctc_vocab.json"
    train_jsonl = Path(args.slurp_jsonl_dir) / "train.jsonl"

    if intent_map_path.exists() and ctc_vocab_path.exists():
        intent_list = json.loads(intent_map_path.read_text())
        ctc_vocab = CTCVocab.load(ctc_vocab_path)
    else:
        intent_list, slot_types = build_label_maps([train_jsonl])
        ctc_vocab = CTCVocab(slot_types)
        out_dir.mkdir(parents=True, exist_ok=True)
        intent_map_path.write_text(json.dumps(intent_list, indent=2))
        ctc_vocab.save(ctc_vocab_path)

    intent2id = {label: i for i, label in enumerate(intent_list)}
    return intent_list, intent2id, ctc_vocab


@torch.no_grad()
def evaluate(model, loader, device, ctc_vocab, intent_list):
    model.eval()
    pred_intents, gold_intents = [], []
    pred_texts, gold_texts = [], []
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        outputs = model(batch["input_values"], batch["attention_mask"])

        pred_intents.extend(outputs["intent_logits"].argmax(-1).tolist())
        gold_intents.extend(batch["intent_ids"].tolist())

        greedy_ids = outputs["ctc_log_probs"].argmax(-1).transpose(0, 1).tolist()  # (B, T)
        feat_lens = outputs["feat_lens"].tolist()
        for ids, L in zip(greedy_ids, feat_lens):
            pred_texts.append(ctc_vocab.decode(ids[:L]))
        gold_texts.extend(batch["tagged_texts"])

    acc = intent_accuracy(pred_intents, gold_intents)
    f1 = slu_f1(pred_texts, gold_texts)
    model.train()
    return {"intent_accuracy": acc, **f1}


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    intent_list, intent2id, ctc_vocab = build_or_load_vocabs(args, out_dir)
    print(f"[vocab] {len(intent_list)} intents, {len(ctc_vocab)} CTC symbols")

    train_ds = SlurpDataset(
        Path(args.slurp_jsonl_dir) / "train.jsonl", args.audio_root,
        ctc_vocab, intent2id, args.max_audio_seconds,
    )
    dev_ds = SlurpDataset(
        Path(args.slurp_jsonl_dir) / "devel.jsonl", args.audio_root,
        ctc_vocab, intent2id, args.max_audio_seconds,
    )
    print(f"[data] train={len(train_ds)} dev={len(dev_ds)}")

    collate = functools.partial(collate_fn, pad_id=ctc_vocab.pad_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, collate_fn=collate)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=collate)

    model = HubertSLUModel(
        num_intents=len(intent_list),
        ctc_vocab_size=len(ctc_vocab),
        pad_id=ctc_vocab.pad_id,
        hubert_name=args.hubert_name,
        semantic_layer=args.semantic_layer,
        freeze_feature_extractor=args.freeze_feature_extractor,
        freeze_encoder_layers=args.freeze_encoder_layers,
    )
    model.set_blank_id(ctc_vocab.blank_id)
    model.to(args.device)

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr
    )
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=max(1, total_steps)
    )

    best_f1 = -1.0
    step = 0
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for batch in pbar:
            batch = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            outputs = model(batch["input_values"], batch["attention_mask"])
            loss, intent_loss, ctc_loss = model.compute_loss(
                batch, outputs, ctc_loss_weight=args.ctc_loss_weight
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            step += 1
            if step % args.log_every == 0:
                pbar.set_postfix(loss=float(loss), intent=float(intent_loss), ctc=float(ctc_loss))

        metrics = evaluate(model, dev_loader, args.device, ctc_vocab, intent_list)
        print(f"[epoch {epoch}] dev intent_acc={metrics['intent_accuracy']:.4f} "
              f"slu_f1={metrics['f1']:.4f} (P={metrics['precision']:.4f} R={metrics['recall']:.4f})")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            print(f"[epoch {epoch}] new best model saved (slu_f1={best_f1:.4f})")

    torch.save(model.state_dict(), out_dir / "last_model.pt")


if __name__ == "__main__":
    main()
