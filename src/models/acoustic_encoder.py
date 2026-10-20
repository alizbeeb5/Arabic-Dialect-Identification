"""Frozen acoustic encoder with mask-aware pooled clip features.

One pooled representation per clip is shared by the dialect classifier and the
verifier. Processor, padding invariance, dimensions, cache keys and
tolerances are documented in docs/acoustic_features.md.
"""

import argparse
import hashlib
import json
import os
import types
from dataclasses import asdict, dataclass, field
from itertools import islice
from pathlib import Path, PurePosixPath

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

ENCODER_NAME = "arabest-rq-300m-6k"
REPO_ID = "Elyadata/AraBEST-RQ-300M-6k"
REVISION = "dff453e2b5f72f76211a523b1050691967e5a59b"
# Content hashes of the pinned revision's files, checked before loading.
CHECKPOINT_SHA256 = {
    "model.ckpt": "2e9dc6347b93ed7b55c3847c68cfe25947ff4fe21aa7faa9fe5048f15c0ca35a",
    "normalizer.ckpt": "30209fa89aad9f7fa007ea2ef17e33ba45c6eacbf7160b581d1946b49dfe6c04",
}
# Bump when a code change alters feature values for the same audio and config.
FEATURE_CODE_VERSION = "acoustic-features-v2"
DOWNSAMPLE = 4


@dataclass(frozen=True)
class Architecture:
    """Encoder shape. Defaults are the released 300M checkpoint."""
    d_model: int = 848
    nhead: int = 8
    num_layers: int = 24
    d_ffn: int = 2048

    @property
    def num_representations(self):
        # Front-end projection, every Conformer layer, then the final LayerNorm output.
        return self.num_layers + 2


@dataclass(frozen=True)
class FeatureConfig:
    """Everything that changes feature values. Its hash is part of the cache key."""
    encoder: str = ENCODER_NAME
    checkpoint_revision: str = REVISION
    sample_rate_hz: int = 16000
    channel_mix: str = "mean"
    resampler: str = "torchaudio.functional.resample:sinc_interp_hann"
    max_duration_sec: float | None = None  # None encodes the full clip
    crop: str = "start"
    n_fft: int = 400
    win_length_ms: int = 25
    hop_length_ms: int = 10
    n_mels: int = 80
    normalization: str = "sentence"
    attention: str = "full"  # no dynamic-chunk streaming mask
    padding_invariant_batching: bool = True
    layers: tuple[int, ...] | str = "all"
    pooling: str = "mean"  # "mean" or "mean_std"
    precision: str = "fp32"  # "fp32" or "bf16" (CUDA autocast)
    code_version: str = FEATURE_CODE_VERSION

    def __post_init__(self):
        if self.pooling not in ("mean", "mean_std"):
            raise ValueError(f"Unknown pooling: {self.pooling}")
        if self.precision not in ("fp32", "bf16"):
            raise ValueError(f"Unknown precision: {self.precision}")
        if self.channel_mix != "mean" or self.crop != "start" or self.normalization != "sentence":
            raise ValueError("Only mean channel mix, start crop and sentence normalization are implemented")
        if self.attention != "full" or not self.padding_invariant_batching:
            raise ValueError("Only full attention with padding-invariant batching is implemented")
        if isinstance(self.layers, list):
            object.__setattr__(self, "layers", tuple(self.layers))

    def layer_indices(self, architecture):
        count = architecture.num_representations
        if self.layers == "all":
            return tuple(range(count))
        if not self.layers or any(not 0 <= i < count for i in self.layers):
            raise ValueError(f"Layer indices must be in 0..{count - 1}")
        return tuple(self.layers)

    def to_dict(self):
        return asdict(self)

    def sha256(self):
        return hashlib.sha256(canonical_json(self.to_dict())).hexdigest()


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def layer_names(architecture, indices):
    names = ["frontend", *(f"conformer_{i}" for i in range(1, architecture.num_layers + 1)), "final"]
    return [names[i] for i in indices]


# ---------------------------------------------------------------- audio

def load_waveform(path, config):
    """Read a clip as a float32 mono tensor at the configured sample rate."""
    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    wave = torch.from_numpy(audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0])
    return prepare_waveform(wave, rate, config)


def prepare_waveform(wave, rate, config):
    wave = torch.as_tensor(wave, dtype=torch.float32)
    if wave.dim() == 2:  # [channels, samples]
        wave = wave.mean(dim=0)
    if rate != config.sample_rate_hz:
        import torchaudio.functional as AF
        wave = AF.resample(wave, rate, config.sample_rate_hz)
    if config.max_duration_sec is not None:
        wave = wave[: int(round(config.max_duration_sec * config.sample_rate_hz))]
    if wave.numel() == 0 or not torch.isfinite(wave).all():
        raise ValueError("Empty or nonfinite waveform")
    return wave.contiguous()


# ---------------------------------------------------------------- pooling

def frame_mask(lengths, max_len):
    """[B, T] boolean mask; True marks a real (non-padding) frame."""
    lengths = torch.as_tensor(lengths)
    return torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]


def masked_mean_pool(hidden, mask):
    """Average real frames only. hidden: [..., B, T, D]; mask: [B, T]."""
    weights = mask.to(hidden.dtype)[..., None]
    counts = weights.sum(dim=-2).clamp(min=1.0)
    return (hidden * weights).sum(dim=-2) / counts


def masked_mean_std_pool(hidden, mask, eps=1e-5):
    """Concatenate masked mean and population standard deviation: [..., B, 2D]."""
    mean = masked_mean_pool(hidden, mask)
    weights = mask.to(hidden.dtype)[..., None]
    counts = weights.sum(dim=-2).clamp(min=1.0)
    var = (((hidden - mean[..., None, :]) ** 2) * weights).sum(dim=-2) / counts
    return torch.cat([mean, var.clamp(min=eps).sqrt()], dim=-1)


# ---------------------------------------------------------------- model

def _masked_conv_forward(self, x, mask=None, dynchunktrain_config=None):
    """ConvolutionModule.forward with padded frames zeroed before the depthwise conv.

    Stock SpeechBrain masks only the output, so the depthwise conv reads padding
    into a shorter clip's last frames (docs/acoustic_features.md, section 3).
    """
    if dynchunktrain_config is not None or self.causal:
        raise ValueError("Padding-invariant convolution supports full-context, non-causal use only")
    out = self.bottleneck(self.layer_norm(x).transpose(1, 2))
    if mask is not None:
        out = out.masked_fill(mask.transpose(1, 2), 0.0)
    out = self.after_conv(self.conv(out).transpose(1, 2))
    if mask is not None:
        out = out.masked_fill(mask, 0.0)
    return out


def build_modules(architecture, n_mels):
    from speechbrain.lobes.models.convolution import ConvolutionFrontEnd
    from speechbrain.lobes.models.transformer.TransformerASR import EncoderWrapper, TransformerASR
    from speechbrain.processing.features import InputNormalization

    a = architecture
    cnn = ConvolutionFrontEnd(input_shape=(8, 10, n_mels), num_blocks=2, num_layers_per_block=1,
                              out_channels=(128, 64), kernel_sizes=(3, 3), strides=(2, 2),
                              residuals=(False, False))
    transformer = TransformerASR(
        input_size=64 * (n_mels // DOWNSAMPLE), tgt_vocab=5000, d_model=a.d_model, nhead=a.nhead,
        num_encoder_layers=a.num_layers, num_decoder_layers=0, d_ffn=a.d_ffn, dropout=0.1,
        activation=torch.nn.GELU, conformer_activation=torch.nn.GELU, encoder_module="conformer",
        attention_type="RelPosMHAXL", normalize_before=True, causal=False, layerdrop_prob=0.0,
        output_hidden_states=True)
    # Same parameter names as the released model.ckpt: ModuleList([CNN, wrapper]).
    model = torch.nn.ModuleList([cnn, EncoderWrapper(transformer)])
    for layer in transformer.encoder.layers:
        conv = layer.convolution_module
        conv.forward = types.MethodType(_masked_conv_forward, conv)
    return model, InputNormalization(norm_type="sentence")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_device(device=None):
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class ClipFeatures:
    """Pooled features in the caller's order. features: [N, L, D'] float32."""
    clip_ids: list
    features: np.ndarray
    n_frames: np.ndarray
    layer_names: list
    audio_sha256: list = field(default_factory=list)


class AcousticEncoder(torch.nn.Module):
    """Frozen AraBEST-RQ encoder returning frame features or pooled clip features."""

    def __init__(self, config=FeatureConfig(), architecture=Architecture(), device=None,
                 checkpoint_dir=None, expected_sha256=None):
        super().__init__()
        from speechbrain.lobes.features import Fbank

        self.config, self.architecture = config, architecture
        self.layers = config.layer_indices(architecture)
        self.layer_names = layer_names(architecture, self.layers)
        self.device = resolve_device(device)
        self.model, self.normalize = build_modules(architecture, config.n_mels)
        self.compute_fbank = Fbank(
            sample_rate=config.sample_rate_hz, n_fft=config.n_fft, n_mels=config.n_mels,
            win_length=config.win_length_ms, hop_length=config.hop_length_ms)
        self.checkpoint = {"repo_id": None, "revision": None, "files": {}}
        if checkpoint_dir is not None:
            self._load_checkpoint(Path(checkpoint_dir), expected_sha256)
        self.requires_grad_(False)
        self.to(self.device)
        self.eval()

    @classmethod
    def from_pretrained(cls, config=FeatureConfig(), device=None, cache_dir=None):
        """Download the pinned checkpoint (if needed) and load it strictly."""
        from huggingface_hub import hf_hub_download

        if config.encoder != ENCODER_NAME or config.checkpoint_revision != REVISION:
            raise ValueError("Only the pinned AraBEST-RQ-300M-6k revision is supported")
        paths = [hf_hub_download(REPO_ID, name, revision=config.checkpoint_revision,
                                 cache_dir=cache_dir) for name in CHECKPOINT_SHA256]
        encoder = cls(config, Architecture(), device, checkpoint_dir=Path(paths[0]).parent,
                      expected_sha256=CHECKPOINT_SHA256)
        encoder.checkpoint.update(repo_id=REPO_ID, revision=config.checkpoint_revision)
        return encoder

    def _load_checkpoint(self, directory, expected_sha256=None):
        hashes = {name: sha256_file(directory / name) for name in CHECKPOINT_SHA256}
        for name, digest in hashes.items():
            if expected_sha256 and digest != expected_sha256[name]:
                path = directory / name
                raise ValueError(
                    f"Corrupt or incomplete checkpoint file {path} ({path.stat().st_size} bytes, "
                    f"sha256 {digest}). Delete the download cache and retry, preferably on local "
                    "disk rather than a synced drive.")
        state = torch.load(directory / "model.ckpt", map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        # Sentence normalization uses per-clip statistics; the stored global
        # statistics are loaded for provenance but are not applied.
        self.normalize._load(str(directory / "normalizer.ckpt"))
        self.checkpoint["files"] = hashes

    def to(self, device):
        self.device = torch.device(device)
        super().to(self.device)
        self.normalize.to(self.device)  # its statistics are plain tensors, not buffers
        return self

    @property
    def output_dim(self):
        return self.architecture.d_model * (2 if self.config.pooling == "mean_std" else 1)

    def _autocast(self):
        enabled = self.config.precision == "bf16" and self.device.type == "cuda"
        return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=enabled)

    def _frontend(self, wave):
        """One clip: Fbank -> sentence norm -> pad to x4 -> CNN. Returns [T_enc, 1280]."""
        feats = self.compute_fbank(wave[None].to(self.device))
        feats = self.normalize(feats, torch.ones(1, device=self.device))
        pad = (-feats.shape[1]) % DOWNSAMPLE
        feats = F.pad(feats, (0, 0, 0, pad))
        with self._autocast():
            src = self.model[0](feats)  # [1, T_enc, F', C]
        return src.reshape(src.shape[1], -1)

    @torch.inference_mode()
    def encode(self, waves):
        """Frame features for a batch of prepared waveforms.

        Returns (hidden [L, B, T, D] float32, mask [B, T], n_frames [B]) where L
        follows config.layers and mask marks each clip's real encoder frames.
        """
        srcs = [self._frontend(wave) for wave in waves]
        with self._autocast():
            n_frames = torch.tensor([len(src) for src in srcs], device=self.device)
            longest = int(n_frames.max())
            batch = torch.stack([F.pad(src, (0, 0, 0, longest - len(src))) for src in srcs])
            # SpeechBrain reconstructs integer lengths and builds its attention
            # mask from these values. BF16 rounds frame counts above 256.
            final, hidden = self.model[1](batch, n_frames.float() / longest)
        stacked = torch.stack([*hidden, final])[list(self.layers)].float()
        return stacked, frame_mask(n_frames, longest), n_frames

    def pool(self, hidden, mask):
        """[L, B, T, D] -> [B, L, D'] using only real frames."""
        pooled = (masked_mean_pool(hidden, mask) if self.config.pooling == "mean"
                  else masked_mean_std_pool(hidden, mask))
        return pooled.transpose(0, 1)

    def embed(self, waves):
        hidden, mask, n_frames = self.encode(waves)
        return self.pool(hidden, mask), n_frames

    def extract(self, clips, batch_size=8):
        """Pool features for (clip_id, waveform) pairs; output order equals input order.

        Clips are grouped by length to limit padding work; batching does not
        change values beyond the documented tolerance.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        clip_ids = [clip_id for clip_id, _ in clips]
        if len(set(clip_ids)) != len(clip_ids):
            raise ValueError("Duplicate clip_id in extraction request")
        order = sorted(range(len(clips)), key=lambda i: clips[i][1].numel())
        features = np.zeros((len(clips), len(self.layers), self.output_dim), dtype=np.float32)
        n_frames = np.zeros(len(clips), dtype=np.int64)
        for start in range(0, len(order), batch_size):
            chunk = order[start:start + batch_size]
            pooled, frames = self.embed([clips[i][1] for i in chunk])
            features[chunk] = pooled.cpu().numpy()
            n_frames[chunk] = frames.cpu().numpy()
        if not np.isfinite(features).all():
            raise FloatingPointError("Nonfinite pooled features")
        return ClipFeatures(clip_ids, features, n_frames, list(self.layer_names))

    def iter_extract_manifest(self, rows, audio_root, batch_size=8, chunk_size=128):
        """Yield pooled chunks in input order, decoding at most chunk_size clips.

        Write each yielded result to FeatureCache before requesting the next
        chunk to keep both waveform and pooled-feature memory bounded.
        """
        if batch_size <= 0 or chunk_size <= 0:
            raise ValueError("batch_size and chunk_size must be positive")
        rows = iter(rows)
        seen = set()
        while chunk := list(islice(rows, chunk_size)):
            clips, hashes = [], []
            for row in chunk:
                clip_id = row["clip_id"]
                if clip_id in seen:
                    raise ValueError("Duplicate clip_id in extraction request")
                seen.add(clip_id)
                path = resolve_audio_path(audio_root, row["audio_path"])
                digest = sha256_file(path)
                if row.get("audio_sha256") not in (None, digest):
                    raise ValueError(f"Audio content differs from manifest: {clip_id}")
                clips.append((clip_id, load_waveform(path, self.config)))
                hashes.append(digest)
            result = self.extract(clips, batch_size=batch_size)
            result.audio_sha256 = hashes
            del clips  # release decoded waveforms before handing off pooled features
            yield result

    def extract_manifest(self, rows, audio_root, batch_size=8, chunk_size=128):
        """Collect pooled features for a small manifest; audio is decoded in chunks.

        The returned features still occupy memory for the entire request. Use
        iter_extract_manifest and cache.write for full-dataset extraction.
        """
        results = list(self.iter_extract_manifest(rows, audio_root, batch_size, chunk_size))
        if not results:
            return ClipFeatures([], np.empty((0, len(self.layers), self.output_dim), dtype=np.float32),
                                np.empty(0, dtype=np.int64), list(self.layer_names))
        return ClipFeatures(
            [clip_id for result in results for clip_id in result.clip_ids],
            np.concatenate([result.features for result in results]),
            np.concatenate([result.n_frames for result in results]), list(self.layer_names),
            [digest for result in results for digest in result.audio_sha256])

    def provenance(self):
        return {"encoder": self.config.encoder, "checkpoint": self.checkpoint,
                "architecture": asdict(self.architecture),
                "feature_config": self.config.to_dict(),
                "feature_config_sha256": self.config.sha256(),
                "layer_indices": list(self.layers), "layer_names": self.layer_names,
                "output_dim": self.output_dim, "frame_shift_sec": DOWNSAMPLE * self.config.hop_length_ms / 1000}


def resolve_audio_path(audio_root, audio_path):
    relative = PurePosixPath(audio_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"audio_path must stay under audio_root: {audio_path}")
    return Path(audio_root) / relative


# ---------------------------------------------------------------- cache

def clip_cache_key(clip_id, audio_sha256, provenance):
    """Identity of one cached vector: clip, audio content, preprocessing and checkpoint."""
    return hashlib.sha256(canonical_json({
        "clip_id": clip_id, "audio_sha256": audio_sha256,
        "feature_config_sha256": provenance["feature_config_sha256"],
        "checkpoint_revision": provenance["checkpoint"]["revision"]
        or provenance["feature_config"]["checkpoint_revision"],
        "checkpoint_files": provenance["checkpoint"]["files"],
        "code_version": provenance["feature_config"]["code_version"],
    })).hexdigest()


class FeatureCache:
    """Pooled features under OUTPUT_ROOT/features/<encoder>/<revision>/<config>/.

    config.json records the full provenance; index.jsonl maps each clip_id to
    its cache key and to a row of one part-*.npz file.
    """

    def __init__(self, output_root, provenance):
        # JSON round trip so tuples compare equal to the lists stored on disk.
        self.provenance = provenance = json.loads(json.dumps(provenance))
        config = provenance["feature_config"]
        self.root = (Path(output_root) / "features" / config["encoder"]
                     / config["checkpoint_revision"][:12] / provenance["feature_config_sha256"][:16])
        self.index_path = self.root / "index.jsonl"

    def index(self):
        self._check_config()
        if not self.index_path.exists():
            return {}
        rows = [json.loads(line) for line in self.index_path.read_text(encoding="utf-8").splitlines() if line]
        return {row["clip_id"]: row for row in rows}

    # Provenance fields that determine feature values; descriptive fields may change.
    IDENTITY_FIELDS = ("feature_config_sha256", "checkpoint", "layer_names", "output_dim")

    def _check_config(self, create=False):
        path = self.root / "config.json"
        if path.exists():
            stored = json.loads(path.read_text(encoding="utf-8"))
            differs = [key for key in self.IDENTITY_FIELDS if stored.get(key) != self.provenance[key]]
            if differs:
                raise ValueError(f"Cache {path} was built with different {', '.join(differs)}; "
                                 "use another OUTPUT_ROOT or remove that cache directory")
        elif self.index_path.exists():
            raise ValueError(f"Cache {self.root} has an index but no config.json; "
                             "restore its provenance or use another OUTPUT_ROOT")
        elif create:
            self.root.mkdir(parents=True, exist_ok=True)
            path.write_bytes(json.dumps(self.provenance, indent=2, sort_keys=True).encode("utf-8") + b"\n")

    def write(self, result):
        """Store a ClipFeatures result. Refuses to overwrite a clip with a different key."""
        if len(result.audio_sha256) != len(result.clip_ids):
            raise ValueError("Cache writes need the audio_sha256 of every clip")
        if result.layer_names != self.provenance["layer_names"]:
            raise ValueError("Feature layers differ from the cache provenance")
        self._check_config(create=True)
        existing = self.index()
        keys = [clip_cache_key(c, h, self.provenance) for c, h in zip(result.clip_ids, result.audio_sha256)]
        new = []
        for i, (clip_id, key) in enumerate(zip(result.clip_ids, keys)):
            if clip_id in existing:
                if existing[clip_id]["cache_key"] != key:
                    raise ValueError(f"Stale cache entry for {clip_id}; use a new cache or remove it")
                continue
            new.append(i)
        if not new:
            return 0
        part = "part-" + hashlib.sha256("".join(keys[i] for i in new).encode()).hexdigest()[:16] + ".npz"
        np.savez(self.root / part, clip_ids=np.array([result.clip_ids[i] for i in new]),
                 cache_keys=np.array([keys[i] for i in new]), features=result.features[new],
                 n_frames=result.n_frames[new])
        with self.index_path.open("a", encoding="utf-8") as stream:
            for row, i in enumerate(new):
                stream.write(json.dumps({"clip_id": result.clip_ids[i], "cache_key": keys[i],
                                         "audio_sha256": result.audio_sha256[i], "part": part,
                                         "row": row, "n_frames": int(result.n_frames[i])},
                                        sort_keys=True) + "\n")
        return len(new)

    def missing(self, rows):
        """Manifest rows whose audio or provenance is not cached yet."""
        existing = self.index()
        return [row for row in rows if row["clip_id"] not in existing or "audio_sha256" not in row
                or existing[row["clip_id"]]["cache_key"]
                != clip_cache_key(row["clip_id"], row["audio_sha256"], self.provenance)]

    def load(self, clip_ids=None):
        """Return cached ClipFeatures in the requested clip order."""
        index = self.index()
        clip_ids = list(index) if clip_ids is None else list(clip_ids)
        unknown = [c for c in clip_ids if c not in index]
        if unknown:
            raise KeyError(f"{len(unknown)} clips are not cached, e.g. {unknown[0]}")
        parts = {}
        features, n_frames = [], []
        for clip_id in clip_ids:
            entry = index[clip_id]
            if entry["cache_key"] != clip_cache_key(clip_id, entry["audio_sha256"], self.provenance):
                raise ValueError(f"Cache key does not match provenance for {clip_id}")
            data = parts.get(entry["part"])
            if data is None:
                data = parts[entry["part"]] = dict(np.load(self.root / entry["part"]))
            row = entry["row"]
            if data["clip_ids"][row] != clip_id or data["cache_keys"][row] != entry["cache_key"]:
                raise ValueError(f"Cache index and part file disagree for {clip_id}")
            features.append(data["features"][row])
            n_frames.append(data["n_frames"][row])
        values = (np.stack(features) if features else
                  np.empty((0, len(self.provenance["layer_names"]), self.provenance["output_dim"]),
                           dtype=np.float32))
        return ClipFeatures(clip_ids, values, np.array(n_frames, dtype=np.int64),
                            list(self.provenance["layer_names"]),
                            [index[c]["audio_sha256"] for c in clip_ids])


# ---------------------------------------------------------------- CLI

def read_manifest(path, limit=None):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def main(argv=None):
    parser = argparse.ArgumentParser(description="Extract pooled acoustic features into the cache.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=os.environ.get("OUTPUT_ROOT"))
    parser.add_argument("--cache-root", type=Path, default=os.environ.get("CACHE_ROOT"),
                        help="Hugging Face download cache")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=128,
                        help="Maximum clips decoded and cached at once (default: 128)")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--layers", default="all", help='"all" or comma-separated indices')
    parser.add_argument("--pooling", default="mean", choices=["mean", "mean_std"])
    parser.add_argument("--precision", default="fp32", choices=["fp32", "bf16"])
    parser.add_argument("--max-duration-sec", type=float)
    args = parser.parse_args(argv)
    if args.output_root is None:
        parser.error("--output-root or OUTPUT_ROOT is required")
    if args.batch_size <= 0 or args.chunk_size <= 0:
        parser.error("--batch-size and --chunk-size must be positive")
    layers = "all" if args.layers == "all" else tuple(int(i) for i in args.layers.split(","))
    config = FeatureConfig(layers=layers, pooling=args.pooling, precision=args.precision,
                           max_duration_sec=args.max_duration_sec)
    encoder = AcousticEncoder.from_pretrained(config, device=args.device, cache_dir=args.cache_root)
    cache = FeatureCache(args.output_root, encoder.provenance())
    rows = read_manifest(args.manifest, args.limit)
    todo = cache.missing(rows)
    for result in encoder.iter_extract_manifest(todo, args.audio_root, batch_size=args.batch_size,
                                               chunk_size=args.chunk_size):
        cache.write(result)
    print(f"Cache: {cache.root} ({len(rows)} clips requested, {len(todo)} extracted)")


if __name__ == "__main__":
    main()
