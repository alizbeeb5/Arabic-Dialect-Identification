"""Week 2 B/C: identical masked attention + MLP; only encoder trainability differs."""
import torch
from torch import nn


class MaskedAttentionPooling(nn.Module):
    """Scalar additive attention, a_t = softmax(v^T tanh(W h_t)), over real frames."""
    def __init__(self, dim=848, attention_dim=128):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, attention_dim), nn.Tanh(),
                                   nn.Linear(attention_dim, 1, bias=False))

    def forward(self, frames, mask):
        if mask.shape != frames.shape[:2] or mask.dtype != torch.bool or not mask.any(1).all():
            raise ValueError("Attention needs a boolean mask with real frames in every clip")
        # Zero padding before scoring AND summation (even NaN padding must not leak).
        frames = frames.masked_fill(~mask[..., None], 0)
        scores = self.score(frames).squeeze(-1).float().masked_fill(~mask, -torch.inf)
        weights = scores.softmax(-1)
        return (frames.float() * weights[..., None]).sum(1)


class AcousticDialectModel(nn.Module):
    def __init__(self, encoder, architecture, num_labels=20, attention_dim=128, hidden=256, dropout=0.2):
        super().__init__()
        if architecture not in ("B", "C"):
            raise ValueError("Use the existing frozen-linear runner for architecture A")
        self.encoder, self.architecture = encoder, architecture
        self.pooling = MaskedAttentionPooling(encoder.architecture.d_model, attention_dim)
        self.head = nn.Sequential(nn.Linear(encoder.architecture.d_model, hidden), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, num_labels))
        # Normalization and acoustic frontend settings remain fixed in both arms.
        self.encoder.requires_grad_(False)
        if architecture == "C":
            self.encoder.model.requires_grad_(True)
        self.train(True)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.normalize.eval()
        self.encoder.compute_fbank.eval()
        if self.architecture == "B":
            self.encoder.eval()
        return self

    def forward(self, waves):
        if self.architecture == "B":
            with torch.no_grad():
                frames, mask, _ = self.encoder.encode_final(waves)
        else:
            frames, mask, _ = self.encoder.encode_final(waves)
        return self.head(self.pooling(frames, mask))

    def parameter_groups(self, encoder_lr, head_lr, weight_decay):
        groups = []
        for role, modules, rate in (("pooling_and_head", (self.pooling, self.head), head_lr),
                                    ("encoder", (self.encoder,), encoder_lr)):
            params = [p for module in modules for p in module.parameters() if p.requires_grad]
            for decay in (True, False):
                selected = [p for p in params if (p.ndim >= 2) == decay]
                if selected:
                    groups.append({"params": selected, "lr": rate,
                                   "weight_decay": weight_decay if decay else 0.0,
                                   "name": role + ("_weights" if decay else "_bias_and_norm")})
        return groups
