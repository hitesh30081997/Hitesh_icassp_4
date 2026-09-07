"""
vocab.py
Vocabulary / label utilities for HuBERT-based end-to-end SLU on SLURP.

Two label spaces are used:

1. Intent classes: `scenario + "_" + action` (SLURP's own `intent` field
   already stores this, e.g. "alarm_set", "calendar_query").

2. A CTC token vocabulary for the joint transcription + slot-tagging
   branch. SLURP's bracketed slot annotations, e.g.

       "wake me up at [time : five am]"

   are converted into a flat token stream with explicit slot-boundary
   markers so a single CTC head can transcribe speech AND tag slots at
   the same time:

       "wake me up at <fill_time> five am <end_fill>"

   Decoding this string back with `extract_slots_from_tagged_text`
   recovers (slot_type, filler) pairs, which is the format SLURP's
   official SLU-F1 scorer expects.
"""
import json
import re
from collections import Counter
from pathlib import Path

PAD_TOKEN = "<pad>"
BLANK_TOKEN = "<blank>"    # CTC blank symbol
UNK_TOKEN = "<unk>"
SPACE_TOKEN = "|"          # word boundary (wav2vec2/HuBERT convention)
END_FILL_TOKEN = "<end_fill>"

SLOT_SPAN_RE = re.compile(r"\[\s*([a-zA-Z_0-9]+)\s*:\s*([^\]]+?)\s*\]")


def annotation_to_tagged_text(sentence_annotation: str) -> str:
    """"wake me up at [time : five am]" -> "wake me up at <fill_time> five am <end_fill>" """
    def _sub(m):
        slot_type, filler = m.group(1), m.group(2).strip()
        return f"<fill_{slot_type}> {filler} {END_FILL_TOKEN}"

    tagged = SLOT_SPAN_RE.sub(_sub, sentence_annotation)
    return re.sub(r"\s+", " ", tagged).strip()


def extract_slots_from_tagged_text(tagged_text: str):
    """Inverse of `annotation_to_tagged_text`. Robust to minor decoding
    noise (e.g. a missing closing tag)."""
    tokens = tagged_text.split()
    slots, cur_type, cur_words = [], None, []
    for tok in tokens:
        if tok.startswith("<fill_") and tok.endswith(">"):
            if cur_type is not None and cur_words:
                slots.append((cur_type, " ".join(cur_words)))
            cur_type, cur_words = tok[len("<fill_"):-1], []
        elif tok == END_FILL_TOKEN:
            if cur_type is not None and cur_words:
                slots.append((cur_type, " ".join(cur_words)))
            cur_type, cur_words = None, []
        elif cur_type is not None:
            cur_words.append(tok)
    if cur_type is not None and cur_words:
        slots.append((cur_type, " ".join(cur_words)))
    return slots


def strip_tags(tagged_text: str) -> str:
    """Remove slot tags, leaving plain transcript text."""
    keep = []
    for tok in tagged_text.split():
        if (tok.startswith("<fill_") and tok.endswith(">")) or tok == END_FILL_TOKEN:
            continue
        keep.append(tok)
    return " ".join(keep)


class CTCVocab:
    """Character-level CTC vocab, extended with atomic slot-tag tokens.

    Ordinary words are spelled character-by-character (kept vocab small);
    slot tags (<fill_TYPE>, <end_fill>) are single atomic units so the
    model only needs to predict one symbol per slot boundary.
    """

    def __init__(self, slot_types):
        specials = [PAD_TOKEN, BLANK_TOKEN, UNK_TOKEN, SPACE_TOKEN, END_FILL_TOKEN]
        fill_tokens = [f"<fill_{s}>" for s in sorted(slot_types)]
        chars = list("abcdefghijklmnopqrstuvwxyz0123456789'")
        self.itos = specials + fill_tokens + chars
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        self.blank_id = self.stoi[BLANK_TOKEN]
        self.pad_id = self.stoi[PAD_TOKEN]

    def __len__(self):
        return len(self.itos)

    def encode(self, tagged_text: str):
        ids = []
        for word in tagged_text.strip().split(" "):
            if not word:
                continue
            if (word.startswith("<fill_") and word.endswith(">")) or word == END_FILL_TOKEN:
                ids.append(self.stoi.get(word, self.stoi[UNK_TOKEN]))
            else:
                for ch in word.lower():
                    ids.append(self.stoi.get(ch, self.stoi[UNK_TOKEN]))
            ids.append(self.stoi[SPACE_TOKEN])
        if ids and ids[-1] == self.stoi[SPACE_TOKEN]:
            ids.pop()
        return ids

    def decode(self, ids):
        """Standard CTC greedy collapse (dedupe + drop blanks), then
        regroup characters into words / keep tags atomic."""
        collapsed, prev = [], None
        for i in ids:
            if i == prev:
                continue
            prev = i
            if i == self.blank_id:
                continue
            collapsed.append(i)

        out_words, cur_word = [], []
        for i in collapsed:
            tok = self.itos[i]
            if tok == SPACE_TOKEN:
                if cur_word:
                    out_words.append("".join(cur_word)); cur_word = []
            elif (tok.startswith("<fill_") and tok.endswith(">")) or tok == END_FILL_TOKEN:
                if cur_word:
                    out_words.append("".join(cur_word)); cur_word = []
                out_words.append(tok)
            else:
                cur_word.append(tok)
        if cur_word:
            out_words.append("".join(cur_word))
        return " ".join(out_words)

    def save(self, path):
        Path(path).write_text(json.dumps(self.itos, indent=2))

    @classmethod
    def load(cls, path):
        itos = json.loads(Path(path).read_text())
        obj = cls.__new__(cls)
        obj.itos = itos
        obj.stoi = {t: i for i, t in enumerate(itos)}
        obj.blank_id = obj.stoi[BLANK_TOKEN]
        obj.pad_id = obj.stoi[PAD_TOKEN]
        return obj


def build_label_maps(jsonl_paths):
    """Scan SLURP jsonl file(s) and build:
      - sorted list of intent labels ("scenario_action")
      - sorted list of slot types seen in `sentence_annotation`
    """
    intents = Counter()
    slot_types = set()
    for path in jsonl_paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                scenario, action = rec.get("scenario"), rec.get("action")
                if scenario is None or action is None:
                    continue
                intents[f"{scenario}_{action}"] += 1
                ann = rec.get("sentence_annotation", "") or ""
                for m in SLOT_SPAN_RE.finditer(ann):
                    slot_types.add(m.group(1))
    return sorted(intents.keys()), sorted(slot_types)
