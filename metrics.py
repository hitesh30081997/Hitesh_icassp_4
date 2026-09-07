"""
metrics.py
Intent accuracy + a simplified span-based SLU-F1, matching SLURP's evaluation
notion of "an entity is correct if both its type and filler text match"
(word-error tolerant exact match here; swap in the official `slurp_eval`
scorer for challenge-comparable numbers).
"""
from vocab import extract_slots_from_tagged_text


def intent_accuracy(pred_ids, gold_ids):
    correct = sum(int(p == g) for p, g in zip(pred_ids, gold_ids))
    return correct / max(1, len(gold_ids))


def _normalize(text):
    return " ".join(text.lower().split())


def slu_f1(pred_tagged_texts, gold_tagged_texts):
    """Micro-averaged precision/recall/F1 over (slot_type, filler) pairs."""
    tp = fp = fn = 0
    for pred_text, gold_text in zip(pred_tagged_texts, gold_tagged_texts):
        pred_slots = [(t, _normalize(f)) for t, f in extract_slots_from_tagged_text(pred_text)]
        gold_slots = [(t, _normalize(f)) for t, f in extract_slots_from_tagged_text(gold_text)]

        gold_remaining = gold_slots.copy()
        for slot in pred_slots:
            if slot in gold_remaining:
                gold_remaining.remove(slot)
                tp += 1
            else:
                fp += 1
        fn += len(gold_remaining)

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
