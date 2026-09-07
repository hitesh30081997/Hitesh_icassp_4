"""
model.py
HuBERT-based end-to-end SLU model.

Design
------
HuBERT's intermediate layers are known to carry different information:
lower layers ~ acoustic/phonetic, middle layers ~ phonetic-to-semantic
transition, and layer ~8 (of 12 in HuBERT-base) is commonly reported as a
strong layer for semantic/word-content tasks (this is where SUPERB-style
probing studies show word content and intent-relevant information peaks
for HuBERT-base). We freeze/keep the full HuBERT encoder but only read out
`hidden_states[8]` (the output of the 8th Transformer block) and route it
into two task heads:

  1. Intent classification head
     Attention pooling over time on the layer-8 features -> linear
     classifier over the SLURP intent set (scenario_action).

  2. Slot filling head
     A light BiLSTM refines the layer-8 features temporally, followed by
     a linear CTC projection over a vocabulary that includes both text
     characters and slot-boundary tags (see vocab.py). CTC lets us train
     directly against speech <-> tagged-transcript pairs without any
     forced alignment.

Total loss = intent_ce_loss + ctc_loss_weight * ctc_loss
"""
import torch
import torch.nn as nn
from transformers import HubertModel


class AttentionPool(nn.Module):
    """Simple additive attention pooling over the time dimension."""

    def __init__(self, hidden_size):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, x, mask):
        # x: (B, T, H), mask: (B, T) with 1 for valid frames
        scores = self.score(torch.tanh(self.proj(x))).squeeze(-1)  # (B, T)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)      # (B, T, 1)
        pooled = (x * weights).sum(dim=1)                          # (B, H)
        return pooled


class HubertSLUModel(nn.Module):
    def __init__(
        self,
        num_intents: int,
        ctc_vocab_size: int,
        pad_id: int,
        hubert_name: str = "facebook/hubert-base-ls960",
        semantic_layer: int = 8,
        freeze_feature_extractor: bool = True,
        freeze_encoder_layers: int = 0,
        slot_lstm_hidden: int = 256,
        slot_lstm_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.semantic_layer = semantic_layer
        self.pad_id = pad_id

        self.hubert = HubertModel.from_pretrained(hubert_name)
        hidden_size = self.hubert.config.hidden_size

        if freeze_feature_extractor:
            self.hubert.feature_extractor._freeze_parameters()

        if freeze_encoder_layers > 0:
            for i, layer in enumerate(self.hubert.encoder.layers):
                if i < freeze_encoder_layers:
                    for p in layer.parameters():
                        p.requires_grad = False

        # --- Intent branch ---
        self.intent_pool = AttentionPool(hidden_size)
        self.intent_dropout = nn.Dropout(dropout)
        self.intent_head = nn.Linear(hidden_size, num_intents)

        # --- Slot filling branch (BiLSTM + CTC over tagged transcript) ---
        self.slot_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=slot_lstm_hidden,
            num_layers=slot_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if slot_lstm_layers > 1 else 0.0,
        )
        self.slot_dropout = nn.Dropout(dropout)
        self.ctc_head = nn.Linear(slot_lstm_hidden * 2, ctc_vocab_size)

    def _feature_mask_from_attention_mask(self, attention_mask):
        """Convert raw-waveform attention mask to the (shorter) feature-frame
        mask that matches HuBERT's CNN downsampling, using HF's helper."""
        feat_lens = self.hubert._get_feat_extract_output_lengths(
            attention_mask.sum(-1)
        )
        max_len = int(feat_lens.max().item())
        frame_mask = torch.arange(max_len, device=attention_mask.device)[None, :] < feat_lens[:, None]
        return frame_mask.long(), feat_lens

    def forward(self, input_values, attention_mask, ctc_target_lens=None):
        outputs = self.hubert(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # hidden_states is a tuple of length num_layers+1 (index 0 = CNN output
        # embeddings before any Transformer block). Index `semantic_layer`
        # therefore corresponds to the output *after* the 8th Transformer block.
        semantic_feats = outputs.hidden_states[self.semantic_layer]  # (B, T, H)

        frame_mask, feat_lens = self._feature_mask_from_attention_mask(attention_mask)
        # hidden_states time dim can be off by a few frames from our computed
        # lengths on some torch/audio versions; clip defensively.
        T = semantic_feats.shape[1]
        frame_mask = frame_mask[:, :T]
        feat_lens = feat_lens.clamp(max=T)

        # ---- Intent classification ----
        pooled = self.intent_pool(semantic_feats, frame_mask)
        intent_logits = self.intent_head(self.intent_dropout(pooled))

        # ---- Slot filling (CTC) ----
        slot_feats, _ = self.slot_lstm(semantic_feats)
        slot_feats = self.slot_dropout(slot_feats)
        ctc_logits = self.ctc_head(slot_feats)                      # (B, T, V)
        log_probs = torch.log_softmax(ctc_logits, dim=-1).transpose(0, 1)  # (T, B, V) for CTCLoss

        return {
            "intent_logits": intent_logits,
            "ctc_log_probs": log_probs,
            "feat_lens": feat_lens,
        }

    def compute_loss(self, batch, outputs, ctc_loss_weight=1.0):
        intent_loss = nn.functional.cross_entropy(
            outputs["intent_logits"], batch["intent_ids"]
        )
        ctc_loss = nn.functional.ctc_loss(
            outputs["ctc_log_probs"],
            batch["ctc_targets"],
            outputs["feat_lens"],
            batch["ctc_target_lens"],
            blank=self._blank_id(),
            zero_infinity=True,
        )
        total = intent_loss + ctc_loss_weight * ctc_loss
        return total, intent_loss.detach(), ctc_loss.detach()

    def _blank_id(self):
        # blank id is passed in at construction time via ctc_head size /
        # vocab; stored separately by the training script for clarity.
        return self._ctc_blank_id

    def set_blank_id(self, blank_id):
        self._ctc_blank_id = blank_id
