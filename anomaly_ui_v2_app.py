"""
Per-category files:
    type_conditioned_adapter_v3.pt (or the deployed alias type_conditioned_adapter_v2.pt)
    generalized_empirical_mask_bank_v3.pt (or empirical_mask_bank.pt)
    mvtec_<category>_diffusion_inpaint_improved.pt

Multi-category layout:
    weights/
      bottle/
        type_conditioned_adapter_v3.pt
        generalized_empirical_mask_bank_v3.pt
        mvtec_bottle_diffusion_inpaint_improved.pt
      capsule/
        type_conditioned_adapter_v3.pt
        generalized_empirical_mask_bank_v3.pt
        mvtec_capsule_diffusion_inpaint_improved.pt

Example:
    python -m venv .venv
    .venv\\Scripts\\activate
    python -m pip install torch torchvision "diffusers==0.30.3" "gradio==5.49.1" pillow "numpy==2.0.2" "huggingface-hub>=0.34.0,<1.0"
    python anomaly_ui_v2_app.py --weights-root ./weights --category auto

Single-category example:
    python anomaly_ui_v2_app.py \
        --category capsule \
        --weights-root ./capsule_weights \
        --host 0.0.0.0 \
        --port 7860

The mask gallery returns the exact binary core mask used for each generated
image. The mask bank stores masks only, not anomaly images.
"""

import argparse
import gc
import math
import random
import threading
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from diffusers import DDIMScheduler, DDPMScheduler, UNet2DModel


# -----------------------------------------------------------------------------
# Compatibility and global runtime state
# -----------------------------------------------------------------------------
try:
    import diffusers.models.modeling_utils as diffusers_modeling_utils

    def _safe_get_parameter_dtype(module):
        last_dtype = None
        for parameter in module.parameters():
            last_dtype = parameter.dtype
            if parameter.is_floating_point():
                return parameter.dtype
        return last_dtype if last_dtype is not None else torch.float32

    # Needed by older diffusers releases under Python 3.13/DataParallel-style
    # model internals. It is harmless for current diffusers versions.
    diffusers_modeling_utils.get_parameter_dtype = _safe_get_parameter_dtype
except Exception as error:
    print("Warning: diffusers dtype compatibility patch failed:", error)


IMAGE_SIZE = 256
APPEARANCE_Z_DIM_DEFAULT = 4
MIN_MASK_PIXELS = 10
DEFAULT_SEVERITY = 0.60
DEVICE = torch.device("cpu")
USE_AMP = False

image_transform = T.Compose([
    T.Resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=T.InterpolationMode.BICUBIC,
    ),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def choose_device(requested):
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def strip_module_prefix(state):
    if state and all(key.startswith("module.") for key in state):
        return {
            key.replace("module.", "", 1): value
            for key, value in state.items()
        }
    return state


def tensor_to_display(tensor):
    return (
        tensor.detach().cpu().clamp(-1, 1) * 0.5 + 0.5
    ).permute(1, 2, 0).numpy().clip(0, 1)


def estimate_foreground(image, threshold=0.985):
    rgb = (image * 0.5 + 0.5).clamp(0, 1)
    gray = rgb.mean(dim=1, keepdim=True)
    foreground = (gray < threshold).float()
    return F.max_pool2d(foreground, kernel_size=11, stride=1, padding=5)


def dilate(mask, radius):
    if radius <= 0:
        return mask
    kernel = 2 * radius + 1
    return F.max_pool2d(mask, kernel, stride=1, padding=radius)


def erode(mask, radius):
    if radius <= 0:
        return mask
    kernel = 2 * radius + 1
    return 1.0 - F.max_pool2d(
        1.0 - mask,
        kernel,
        stride=1,
        padding=radius,
    )


def soft_generation_mask(mask, radius=3):
    expanded = dilate(mask, radius)
    kernel = 2 * radius + 1
    return F.avg_pool2d(
        expanded,
        kernel,
        stride=1,
        padding=radius,
    ).clamp(0, 1)


def apply_affine_pair(
    image,
    mask,
    rng,
    *,
    max_translate,
    scale_low,
    scale_high,
    max_degrees,
    max_shear,
):
    angle = rng.uniform(-max_degrees, max_degrees)
    translate = [
        rng.randint(-int(max_translate), int(max_translate)),
        rng.randint(-int(max_translate), int(max_translate)),
    ]
    scale = rng.uniform(scale_low, scale_high)
    shear = [
        rng.uniform(-max_shear, max_shear),
        rng.uniform(-max_shear, max_shear),
    ]
    image = TF.affine(
        image,
        angle,
        translate,
        scale,
        shear,
        interpolation=TF.InterpolationMode.BILINEAR,
        fill=0.0,
    )
    mask = TF.affine(
        mask,
        angle,
        translate,
        scale,
        shear,
        interpolation=TF.InterpolationMode.NEAREST,
        fill=0.0,
    )
    return image, (mask > 0.5).float()


class EmpiricalMaskBankV3:
    """Loads and samples the Stage 3 training-only mask bank."""

    def __init__(self, checkpoint):
        self.type_names = list(checkpoint["type_names"])
        self.type_to_id = dict(checkpoint.get(
            "type_to_id",
            {name: index for index, name in enumerate(self.type_names)},
        ))
        self.masks_by_type = {
            name: tensor.float().cpu()
            for name, tensor in checkpoint["masks_by_type"].items()
        }
        self.source_paths_by_type = checkpoint.get(
            "source_paths_by_type", {}
        )
        jitter = checkpoint.get("jitter", {})
        self.translate_px = int(jitter.get("translate_px", 6))
        self.scale_low = float(jitter.get("scale_low", 0.95))
        self.scale_high = float(jitter.get("scale_high", 1.05))
        self.degrees = float(jitter.get("degrees", 8.0))
        self.shear = float(jitter.get("shear", 3.0))
        self.morph_probability = float(
            jitter.get("morph_probability", 0.30)
        )
        self.morph_radius = int(jitter.get("morph_radius", 1))
        generalization = checkpoint.get("generalization", {})
        # These metadata fields are retained for checkpoint compatibility.
        # The UI's explicit two-mask toggle does not reject unions by area.
        self.max_mix_components = int(
            generalization.get("max_mix_components", 2)
        )
        self.max_mask_area_fraction = float(
            generalization.get("max_area_fraction", 0.20)
        )

        missing = [
            name for name in self.type_names
            if name not in self.masks_by_type
        ]
        if missing:
            raise RuntimeError(
                f"Stage 3 mask bank is missing types: {missing}. "
                f"Available: {sorted(self.masks_by_type)}"
            )

    def _sample_one(self, defect_type, rng, exclude_index=None):
        masks = self.masks_by_type[defect_type]
        candidates = list(range(len(masks)))
        if exclude_index is not None and len(candidates) > 1:
            candidates.remove(int(exclude_index))
        source_index = rng.choice(candidates)
        source_mask = masks[source_index].clone()
        dummy_image = torch.zeros_like(source_mask).repeat(3, 1, 1)
        _, mask = apply_affine_pair(
            dummy_image,
            source_mask,
            rng,
            max_translate=self.translate_px,
            scale_low=self.scale_low,
            scale_high=self.scale_high,
            max_degrees=self.degrees,
            max_shear=self.shear,
        )
        if (
            self.morph_radius > 0
            and rng.random() < self.morph_probability
        ):
            if rng.random() < 0.5:
                mask = dilate(mask.unsqueeze(0), self.morph_radius)[0]
            else:
                mask = erode(mask.unsqueeze(0), self.morph_radius)[0]
            mask = (mask > 0.5).float()
        if int(mask.sum().item()) < MIN_MASK_PIXELS:
            mask = source_mask
        # Keep every sampled component binary. The union returned below is the
        # exact core mask consumed by the diffusion loop and shown by the UI.
        return (mask > 0.5).float(), source_index

    def sample(self, defect_type, seed, use_multiple_masks=False):
        if defect_type not in self.masks_by_type:
            raise KeyError(
                f"Unknown anomaly type {defect_type}. "
                f"Choose from {self.type_names}."
            )
        rng = random.Random(int(seed))
        first_mask, first_index = self._sample_one(defect_type, rng)
        masks = [first_mask]
        source_indices = [first_index]
        if use_multiple_masks and len(self.masks_by_type[defect_type]) > 1:
            # The UI toggle explicitly requests two distinct bank entries.
            # Do not reject the union on area, because that would silently
            # turn the enabled state back into one-mask generation.
            second_mask, second_index = self._sample_one(
                defect_type,
                rng,
                exclude_index=first_index,
            )
            masks.append(second_mask)
            source_indices.append(second_index)
        mask = masks[0]
        for component in masks[1:]:
            mask = torch.maximum(mask, component)
        return (mask > 0.5).float().cpu(), source_indices


# Backward-compatible symbol name for deployment wrappers that import it.
EmpiricalMaskBankV2 = EmpiricalMaskBankV3


# -----------------------------------------------------------------------------
# Stage 3 model definitions
# -----------------------------------------------------------------------------
class NoisePredictionUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet2DModel(
            sample_size=IMAGE_SIZE,
            in_channels=8,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(96, 192, 384, 768),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
            norm_num_groups=32,
            act_fn="silu",
        )

    def forward(self, sample, timestep):
        return self.unet(sample, timestep).sample


def group_count(channels):
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.dimension = dimension

    def forward(self, timesteps):
        half = self.dimension // 2
        scale = math.log(10000) / max(1, half - 1)
        frequencies = torch.exp(
            torch.arange(
                half,
                device=timesteps.device,
                dtype=torch.float32,
            ) * -scale
        )
        angles = timesteps.float().view(-1, 1) * frequencies.view(1, -1)
        embedding = torch.cat([angles.sin(), angles.cos()], dim=1)
        if self.dimension % 2:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class FiLMResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, condition_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.condition = nn.Linear(condition_dim, 2 * out_channels)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x, condition):
        hidden = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.condition(condition).chunk(2, dim=1)
        hidden = self.norm2(hidden)
        hidden = hidden * (1 + scale[:, :, None, None])
        hidden = hidden + shift[:, :, None, None]
        hidden = self.conv2(F.silu(hidden))
        return hidden + self.skip(x)


class TypeConditionedResidualAdapterV2(nn.Module):
    def __init__(
        self,
        num_types,
        type_dim=16,
        appearance_z_dim=4,
        base=48,
        time_dim=64,
        condition_dim=128,
    ):
        super().__init__()
        self.type_embedding = nn.Embedding(num_types, type_dim)
        self.appearance_z_dim = appearance_z_dim
        self.type_dim = type_dim
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        self.condition_mlp = nn.Sequential(
            nn.Linear(time_dim + type_dim + 1, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
        )
        # sample + core mask + soft gate + visible normal + foreground + z
        input_channels = 3 + 1 + 1 + 3 + 1 + appearance_z_dim
        self.input_projection = nn.Conv2d(input_channels, base, 3, padding=1)
        self.enc1 = FiLMResidualBlock(base, base, condition_dim)
        self.down1 = nn.Conv2d(base, base * 2, 4, stride=2, padding=1)
        self.enc2 = FiLMResidualBlock(base * 2, base * 2, condition_dim)
        self.down2 = nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1)
        self.enc3 = FiLMResidualBlock(base * 4, base * 4, condition_dim)
        self.bottleneck = FiLMResidualBlock(base * 4, base * 4, condition_dim)
        self.up2 = nn.ConvTranspose2d(
            base * 4,
            base * 2,
            4,
            stride=2,
            padding=1,
        )
        self.dec2 = FiLMResidualBlock(base * 4, base * 2, condition_dim)
        self.up1 = nn.ConvTranspose2d(
            base * 2,
            base,
            4,
            stride=2,
            padding=1,
        )
        self.dec1 = FiLMResidualBlock(base * 2, base, condition_dim)
        self.output_norm = nn.GroupNorm(group_count(base), base)
        self.output = nn.Conv2d(base, 3, 3, padding=1)

    def make_condition(self, timesteps, type_id, severity):
        type_vector = self.type_embedding(type_id)
        time_vector = self.time_embedding(timesteps)
        severity_vector = severity.view(-1, 1)
        return self.condition_mlp(
            torch.cat([time_vector, type_vector, severity_vector], dim=1)
        )

    def forward(
        self,
        sample,
        core_mask,
        gate,
        visible,
        foreground,
        appearance_z,
        timesteps,
        type_id,
        severity,
    ):
        condition = self.make_condition(timesteps, type_id, severity)
        x = torch.cat(
            [sample, core_mask, gate, visible, foreground, appearance_z],
            dim=1,
        )
        e1 = self.enc1(self.input_projection(x), condition)
        e2 = self.enc2(self.down1(e1), condition)
        e3 = self.enc3(self.down2(e2), condition)
        hidden = self.bottleneck(e3, condition)
        hidden = self.up2(hidden)
        hidden = self.dec2(torch.cat([hidden, e2], dim=1), condition)
        hidden = self.up1(hidden)
        hidden = self.dec1(torch.cat([hidden, e1], dim=1), condition)
        return self.output(F.silu(self.output_norm(hidden)))


# -----------------------------------------------------------------------------
# Stage 3 generator bundle
# -----------------------------------------------------------------------------
class V2Generator:
    def __init__(self, normal_checkpoint, adapter_checkpoint, mask_bank_path):
        self.lock = threading.Lock()
        self.normal_checkpoint_path = Path(normal_checkpoint)
        self.adapter_checkpoint_path = Path(adapter_checkpoint)
        self.mask_bank_path = Path(mask_bank_path)

        bank_checkpoint = torch.load(
            self.mask_bank_path,
            map_location="cpu",
        )
        self.mask_bank = EmpiricalMaskBankV3(bank_checkpoint)

        adapter_checkpoint = torch.load(
            self.adapter_checkpoint_path,
            map_location="cpu",
        )
        self.type_names = list(
            adapter_checkpoint.get("type_names", self.mask_bank.type_names)
        )
        self.type_to_id = dict(
            adapter_checkpoint.get(
                "type_to_id",
                {
                    name: index
                    for index, name in enumerate(self.type_names)
                },
            )
        )
        if self.type_names != self.mask_bank.type_names:
            raise RuntimeError(
                "Type mismatch between Stage 3 adapter and mask bank. "
                f"Adapter={self.type_names}; bank={self.mask_bank.type_names}"
            )

        self.type_dim = int(adapter_checkpoint.get("type_dim", 16))
        self.appearance_z_dim = int(
            adapter_checkpoint.get("appearance_z_dim", 4)
        )
        self.base_channels = int(
            adapter_checkpoint.get("base_channels", 48)
        )
        self.time_embed_dim = int(
            adapter_checkpoint.get("time_embed_dim", 64)
        )
        self.condition_dim = int(
            adapter_checkpoint.get("condition_dim", 128)
        )
        self.default_adapter_scale = float(
            adapter_checkpoint.get("adapter_scale", 1.5)
        )
        self.severity_residual_gain = bool(
            adapter_checkpoint.get("severity_residual_gain", True)
        )
        severity_range = adapter_checkpoint.get(
            "severity_range", [0.20, 1.00]
        )
        self.severity_low = float(severity_range[0])
        self.severity_high = float(severity_range[1])

        self.base_model = NoisePredictionUNet()
        normal_checkpoint_data = torch.load(
            self.normal_checkpoint_path,
            map_location="cpu",
        )
        base_state = normal_checkpoint_data.get(
            "model",
            normal_checkpoint_data,
        )
        base_state = strip_module_prefix(base_state)
        missing, unexpected = self.base_model.load_state_dict(
            base_state,
            strict=False,
        )
        if missing or unexpected:
            raise RuntimeError(
                "Normal backbone checkpoint mismatch. "
                f"Missing={missing[:5]}; unexpected={unexpected[:5]}"
            )
        self.base_model = self.base_model.to(DEVICE).eval()
        for parameter in self.base_model.parameters():
            parameter.requires_grad = False

        self.adapter = TypeConditionedResidualAdapterV2(
            num_types=len(self.type_names),
            type_dim=self.type_dim,
            appearance_z_dim=self.appearance_z_dim,
            base=self.base_channels,
            time_dim=self.time_embed_dim,
            condition_dim=self.condition_dim,
        )
        adapter_state = strip_module_prefix(adapter_checkpoint["adapter"])
        missing, unexpected = self.adapter.load_state_dict(
            adapter_state,
            strict=False,
        )
        if missing or unexpected:
            raise RuntimeError(
                "Stage 3 adapter checkpoint mismatch. "
                f"Missing={missing[:5]}; unexpected={unexpected[:5]}"
            )
        self.adapter = self.adapter.to(DEVICE).eval()

        self.train_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
            clip_sample=False,
        )
        self.sample_scheduler = DDIMScheduler.from_config(
            self.train_scheduler.config
        )
        self.aplas_cumprod = self.train_scheduler.alphas_cumprod.to(DEVICE)
        self.alphas_cumprod = self.train_scheduler.alphas_cumprod.to(DEVICE)

    @torch.inference_mode()
    def generate(
        self,
        context,
        defect_type,
        severity,
        steps,
        seed,
        scale=None,
        use_multiple_masks=False,
    ):
        if context.shape != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(
                f"Expected context shape (1, 3, {IMAGE_SIZE}, {IMAGE_SIZE}), "
                f"got {tuple(context.shape)}"
            )
        if defect_type not in self.type_to_id:
            raise KeyError(
                f"Unknown type {defect_type}; choose from {self.type_names}"
            )
        if scale is None:
            scale = self.default_adapter_scale

        severity = float(np.clip(
            severity,
            self.severity_low,
            self.severity_high,
        ))
        core_mask_cpu, source_indices = self.mask_bank.sample(
            defect_type,
            seed,
            use_multiple_masks=bool(use_multiple_masks),
        )
        core_mask = core_mask_cpu.unsqueeze(0).float().to(DEVICE)
        if int(core_mask.sum().item()) < MIN_MASK_PIXELS:
            raise RuntimeError("The Stage 3 mask bank returned an empty mask.")

        batch = context.shape[0]
        gate = soft_generation_mask(core_mask, radius=3)
        visible = context * (1 - core_mask)
        foreground = estimate_foreground(context)
        type_id = torch.tensor(
            [self.type_to_id[defect_type]],
            dtype=torch.long,
            device=DEVICE,
        )
        severity_tensor = torch.tensor(
            [float(severity)],
            dtype=torch.float32,
            device=DEVICE,
        )

        generator = torch.Generator(device=DEVICE)
        generator.manual_seed(int(seed))
        known_noise = torch.randn(
            context.shape,
            device=DEVICE,
            dtype=context.dtype,
            generator=generator,
        )
        sample = torch.randn(
            context.shape,
            device=DEVICE,
            dtype=context.dtype,
            generator=generator,
        )
        appearance_z = torch.randn(
            batch,
            self.appearance_z_dim,
            IMAGE_SIZE,
            IMAGE_SIZE,
            device=DEVICE,
            dtype=context.dtype,
            generator=generator,
        )

        self.sample_scheduler.set_timesteps(
            int(steps),
            device=DEVICE,
        )
        first_timestep = self.sample_scheduler.timesteps[0]
        first_batch = torch.full(
            (batch,),
            int(first_timestep.item()),
            device=DEVICE,
            dtype=torch.long,
        )
        known_at_first = self.sample_scheduler.add_noise(
            context,
            known_noise,
            first_batch,
        )
        sample = sample * gate + known_at_first * (1 - gate)

        for step_index, timestep in enumerate(self.sample_scheduler.timesteps):
            timestep_batch = torch.full(
                (batch,),
                int(timestep.item()),
                device=DEVICE,
                dtype=torch.long,
            )
            base_input = torch.cat(
                [sample, core_mask, visible, foreground],
                dim=1,
            )
            autocast_context = (
                torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                )
                if USE_AMP else nullcontext()
            )
            with autocast_context:
                base_noise = self.base_model(
                    base_input,
                    timestep_batch,
                )
                adapter_delta = self.adapter(
                    sample,
                    core_mask,
                    gate,
                    visible,
                    foreground,
                    appearance_z,
                    timestep_batch,
                    type_id,
                    severity_tensor,
                )
                severity_gain = (
                    severity_tensor.view(-1, 1, 1, 1)
                    if self.severity_residual_gain else 1.0
                )
                predicted_noise = (
                    base_noise
                    + float(scale) * severity_gain * adapter_delta * gate
                )

            previous = self.sample_scheduler.step(
                predicted_noise,
                timestep,
                sample,
                eta=0.0,
            ).prev_sample

            if step_index + 1 < len(self.sample_scheduler.timesteps):
                next_timestep = self.sample_scheduler.timesteps[step_index + 1]
                next_batch = torch.full(
                    (batch,),
                    int(next_timestep.item()),
                    device=DEVICE,
                    dtype=torch.long,
                )
                known_next = self.sample_scheduler.add_noise(
                    context,
                    known_noise,
                    next_batch,
                )
            else:
                known_next = context
            sample = previous * gate + known_next * (1 - gate)

        return sample.clamp(-1, 1), core_mask, source_indices


# -----------------------------------------------------------------------------
# Path resolution and UI callbacks
# -----------------------------------------------------------------------------
def existing_file(candidates):
    for candidate in candidates:
        candidate = Path(candidate).expanduser()
        if candidate.is_file():
            return candidate.resolve()
    return None


def resolve_paths(args, category=None):
    """Resolve one category's normal checkpoint, adapter, and mask bank.

    A shared root may contain either the files directly or category folders:

        weights/{bottle,capsule}/<checkpoint files>

    Explicit checkpoint arguments are intended for a single selected category.
    """
    category = category or getattr(args, "category", "bottle")
    if category not in {"bottle", "capsule"}:
        raise ValueError(f"Unsupported category: {category}")

    script_dir = Path(__file__).resolve().parent
    normal_name = f"mvtec_{category}_diffusion_inpaint_improved.pt"
    roots = []
    if args.weights_root:
        shared_root = Path(args.weights_root).expanduser()
        roots.extend([shared_root / category, shared_root])
    roots.extend([
        script_dir / f"type_conditioned_generalized_severity_v3_{category}",
        script_dir / f"{category}_type_conditioned_generalized_severity_v3",
        script_dir / "weights" / category,
        script_dir / category,
    ])
    # Also allow a shared root containing the two V3 category folders.
    if category == "bottle":
        roots.extend([
            script_dir / "type_conditioned_generalized_severity_v3_bottle",
            script_dir,
        ])

    adapter_candidates = []
    bank_candidates = []
    normal_candidates = []
    if args.adapter_checkpoint:
        adapter_candidates.append(Path(args.adapter_checkpoint))
    if args.mask_bank:
        bank_candidates.append(Path(args.mask_bank))
    if args.normal_checkpoint:
        normal_candidates.append(Path(args.normal_checkpoint))

    # V3 is the Stage 3 artifact name. The v2 filenames are retained as
    # deployment aliases because existing category bundles use them.
    adapter_names = (
        "type_conditioned_adapter_v3.pt",
        "type_conditioned_adapter_v2.pt",
    )
    bank_names = (
        "generalized_empirical_mask_bank_v3.pt",
        "empirical_mask_bank.pt",
    )
    for root in roots:
        adapter_candidates.extend(root / name for name in adapter_names)
        bank_candidates.extend(root / name for name in bank_names)
        normal_candidates.append(root / normal_name)
        normal_candidates.append(root.parent / normal_name)

    adapter_path = existing_file(adapter_candidates)
    bank_path = existing_file(bank_candidates)
    normal_path = existing_file(normal_candidates)

    missing = []
    if adapter_path is None:
        missing.append(
            "type_conditioned_adapter_v3.pt (or type_conditioned_adapter_v2.pt)"
        )
    if bank_path is None:
        missing.append(
            "generalized_empirical_mask_bank_v3.pt (or empirical_mask_bank.pt)"
        )
    if normal_path is None:
        missing.append(normal_name)
    if missing:
        raise FileNotFoundError(
            f"Could not resolve {category} Stage 3 weights: " + ", ".join(missing) +
            ". Pass --weights-root with category folders, or pass the three "
            "explicit checkpoint options for the selected category."
        )
    return normal_path, adapter_path, bank_path

def image_to_context(value):
    if value is None:
        raise ValueError("Upload a good/normal image first.")
    if isinstance(value, Image.Image):
        image = value.convert("RGB")
    else:
        array = np.asarray(value)
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=2)
        if array.shape[-1] == 4:
            array = array[..., :3]
        if array.dtype != np.uint8:
            if array.max() <= 1.0:
                array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
        image = Image.fromarray(array).convert("RGB")
    return image_transform(image).unsqueeze(0).to(DEVICE)


def create_app(generators):
    """Build a UI for one or more category-specific Stage 3 generators.

    ``generators`` may be a single V2Generator for backward compatibility or
    a mapping such as ``{"bottle": bottle_generator, "capsule": capsule_generator}``.
    Each category keeps its own frozen normal checkpoint, adapter, mask bank,
    and anomaly-type vocabulary.
    """
    try:
        import gradio as gr
    except ImportError as error:
        raise RuntimeError(
            "Gradio is required. Install it with: pip install gradio"
        ) from error

    if isinstance(generators, V2Generator):
        category = getattr(generators, "category", "bottle")
        generators = {category: generators}
    generators = dict(generators)
    if not generators:
        raise ValueError("At least one category-specific generator is required.")
    categories = list(generators)
    default_category = categories[0]

    def update_category(category):
        active = generators[category]
        return gr.update(
            choices=active.type_names,
            value=active.type_names[0],
        )

    def generate_callback(
        category,
        normal_image,
        defect_type,
        severity,
        use_multiple_masks,
        seed_start,
        number_of_outputs,
        inference_steps,
    ):
        try:
            if category not in generators:
                raise KeyError(
                    f"Unknown category {category}; available: {categories}"
                )
            generator = generators[category]
            context = image_to_context(normal_image)
            seed_start = int(seed_start)
            number_of_outputs = int(number_of_outputs)
            inference_steps = int(inference_steps)
            severity = float(severity)
            use_multiple_masks = bool(use_multiple_masks)

            image_gallery = []
            mask_gallery = []
            source_reports = []
            component_counts = []
            with generator.lock:
                for index in range(number_of_outputs):
                    seed = seed_start + index
                    generated, mask, source_indices = generator.generate(
                        context=context,
                        defect_type=defect_type,
                        severity=severity,
                        steps=inference_steps,
                        seed=seed,
                        use_multiple_masks=use_multiple_masks,
                    )
                    source_indices = [int(index) for index in source_indices]
                    source_text = "[" + ", ".join(map(str, source_indices)) + "]"
                    source_reports.append(f"seed={seed}: {source_text}")
                    component_counts.append(len(source_indices))
                    image_gallery.append((
                        tensor_to_display(generated[0]),
                        f"{category} / {defect_type} | seed={seed} | "
                        f"bank sources={source_text}",
                    ))
                    mask_array = (
                        (mask[0, 0] > 0.5).detach().cpu().numpy() * 255
                    ).astype(np.uint8)
                    mask_gallery.append((
                        mask_array,
                        f"{category} exact binary core mask | seed={seed} | "
                        f"bank sources={source_text}",
                    ))

            if use_multiple_masks and component_counts and all(
                count == 2 for count in component_counts
            ):
                mask_mode = "two distinct same-type mask components"
            elif use_multiple_masks:
                mask_mode = "the requested two-mask mode (one bank entry was available)"
            else:
                mask_mode = "one mask component"
            status = (
                f"Generated {len(image_gallery)} {category} / {defect_type} "
                f"sample(s), using {mask_mode}. "
                f"Mask-bank source indices by seed: {'; '.join(source_reports)}. "
                "Each mask shown on the right is the exact binary core mask "
                "used for the corresponding image. No anomaly reference image "
                "was used."
            )
            return image_gallery, mask_gallery, status
        except Exception as error:
            return [], [], f"Error: {error}"

    with gr.Blocks(title="Stage 3 Reference-Free MVTec Anomaly Generator") as demo:
        gr.Markdown(
            "# Stage 3 Reference-Free MVTec Anomaly Generator\n"
            "Choose a loaded MVTec category, upload a good/normal image, and "
            "generate type-conditioned anomalies with exact core masks. "
            "No reference anomaly image is required."
        )
        gr.Markdown(
            "Each category uses its own frozen normal checkpoint, Stage 3 adapter, "
            "and training-only empirical mask bank. Severity is continuous over "
            "0.20–1.00. The optional two-mask toggle unions two distinct bank "
            "masks of the selected type; the exact binary core mask is shown "
            "below. Category switching also switches the valid anomaly-type list."
        )

        category = gr.Dropdown(
            choices=categories,
            value=default_category,
            label="MVTec category",
        )
        normal_image = gr.Image(
            label="Good / normal input image",
            type="numpy",
            height=360,
        )

        with gr.Row():
            defect_type = gr.Dropdown(
                choices=generators[default_category].type_names,
                value=generators[default_category].type_names[0],
                label="Anomaly type",
            )
            severity = gr.Slider(
                minimum=0.20,
                maximum=1.0,
                value=DEFAULT_SEVERITY,
                step=0.01,
                label="Severity",
            )
            seed_start = gr.Number(
                value=101,
                precision=0,
                label="Starting seed",
            )

        with gr.Row():
            use_multiple_masks = gr.Checkbox(
                value=False,
                label="Use two same-type mask components",
                info=(
                    "When enabled, union two distinct training-only masks "
                    "from the selected defect-type bank when two are available."
                ),
            )

        with gr.Row():
            number_of_outputs = gr.Slider(
                minimum=1,
                maximum=6,
                value=3,
                step=1,
                label="Number of outputs",
            )
            inference_steps = gr.Slider(
                minimum=10,
                maximum=50,
                value=30,
                step=5,
                label="DDIM inference steps",
            )

        generate_button = gr.Button(
            "Generate reference-free anomalies",
            variant="primary",
        )
        status = gr.Textbox(label="Status", interactive=False)

        with gr.Row():
            image_gallery = gr.Gallery(
                label="Generated anomalous images",
                columns=3,
                height="auto",
                object_fit="contain",
            )
            mask_gallery = gr.Gallery(
                label="Exact generated core masks",
                columns=3,
                height="auto",
                object_fit="contain",
            )

        category.change(
            fn=update_category,
            inputs=category,
            outputs=defect_type,
        )
        generate_button.click(
            fn=generate_callback,
            inputs=[
                category,
                normal_image,
                defect_type,
                severity,
                use_multiple_masks,
                seed_start,
                number_of_outputs,
                inference_steps,
            ],
            outputs=[image_gallery, mask_gallery, status],
        )

    return demo

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the Stage 3 reference-free bottle/capsule anomaly UI."
    )
    parser.add_argument(
        "--category",
        choices=["auto", "bottle", "capsule"],
        default="auto",
        help="Load one category, or auto-load every category found under --weights-root.",
    )
    parser.add_argument(
        "--weights-root",
        default=None,
        help=(
            "Folder containing category subfolders (bottle/ and capsule/) or "
            "one category's three Stage 3 checkpoint files."
        ),
    )
    parser.add_argument(
        "--normal-checkpoint",
        default=None,
        help="Frozen normal diffusion checkpoint for a single selected category.",
    )
    parser.add_argument(
        "--adapter-checkpoint",
        default=None,
        help=(
            "Stage 3 adapter checkpoint (type_conditioned_adapter_v3.pt; "
            "the deployed v2 filename is also accepted)."
        ),
    )
    parser.add_argument(
        "--mask-bank",
        default=None,
        help=(
            "Stage 3 mask bank (generalized_empirical_mask_bank_v3.pt; "
            "the deployed empirical_mask_bank.pt filename is also accepted)."
        ),
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Inference device. Default: CUDA when available.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Use 0.0.0.0 for LAN access.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Local web port.",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Ask Gradio for a temporary public share URL.",
    )
    parser.add_argument(
        "--show-paths",
        action="store_true",
        help="Print resolved checkpoint paths.",
    )
    return parser.parse_args()


def main():
    global DEVICE, USE_AMP
    args = parse_args()
    DEVICE = choose_device(args.device)
    USE_AMP = DEVICE.type == "cuda"

    if args.category == "auto" and any([
        args.normal_checkpoint,
        args.adapter_checkpoint,
        args.mask_bank,
    ]):
        raise ValueError(
            "Explicit checkpoint paths require --category bottle or --category capsule; "
            "use --weights-root for automatic multi-category loading."
        )

    categories_to_load = (
        ["bottle", "capsule"] if args.category == "auto"
        else [args.category]
    )
    generators = {}
    failures = {}
    for category in categories_to_load:
        try:
            normal_path, adapter_path, bank_path = resolve_paths(args, category)
        except FileNotFoundError as error:
            failures[category] = str(error)
            continue
        if args.show_paths:
            print(f"[{category}] normal checkpoint:", normal_path)
            print(f"[{category}] Stage 3 adapter checkpoint:", adapter_path)
            print(f"[{category}] Stage 3 mask bank:", bank_path)
        print(f"Loading {category} Stage 3 reference-free generator...")
        generator = V2Generator(
            normal_checkpoint=normal_path,
            adapter_checkpoint=adapter_path,
            mask_bank_path=bank_path,
        )
        generator.category = category
        generators[category] = generator
        print(f"[{category}] loaded anomaly types:", generator.type_names)

    if not generators:
        details = "\n".join(
            f"{category}: {message}"
            for category, message in failures.items()
        )
        raise FileNotFoundError(
            "No category weights could be loaded.\n" + details
        )
    if failures:
        print("Skipped categories without a complete weight bundle:")
        for category, message in failures.items():
            print(f"  {category}: {message}")

    print("Device:", DEVICE)
    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    demo = create_app(generators)
    demo.queue(max_size=4)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
