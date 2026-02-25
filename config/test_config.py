from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class TestConfig:
    # Data paths
    data_root: str = "./data/sailor"
    save_path: str = './sailor_eval_17'
    model_checkpoint: str = "./ckpt/tadiff_811.ckpt"

    # Patient IDs to test
    patient_ids: List[str] = field(default_factory=lambda: ['sub-17'])

    # ---------- Model parameters (must match the trained checkpoint) ----------
    # If testing with YOUR trained model, use paper values:
    model_channels: int = 64          # was 32 (debug), paper = 64
    num_heads: int = 4                # was 1 (debug), paper = 4
    num_res_blocks: int = 2           # was 1 (debug), paper = 2
    channel_mult: tuple = (1, 2, 4, 8)  # paper = [64,128,256,512]
    attention_resolutions: list = field(default_factory=lambda: [8, 4])
    image_size: int = 192
    in_channels: int = 13
    out_channels: int = 7
    num_classes: int = 81

    # ---------- Test parameters ----------
    # Paper uses DDPM with T=600 for testing (Section III-C)
    diffusion_steps: int = 600
    target_session_idx: int = 3       # predict the 4th (last) session
    num_samples: int = 5              # ensemble of 5 predictions
    min_tumor_size: int = 20          # skip slices with fewer tumor voxels
    top_k_slices: int = 3

    # Data keys (must match preprocessed .npy file naming)
    npz_keys: List[str] = field(default_factory=lambda: ['image', 'label', 'days', 'treatment'])

    # Visualization settings
    colors: dict = field(default_factory=lambda: {
        0: (0, 0, 0),      # background, black
        1: (0, 255, 0),    # class 1, green / growth
        2: (0, 0, 255),    # class 2, blue / shrinkage
        3: (255, 0, 0),    # class 3, red / stable tumor
    })

    # Thresholds
    mask_threshold: float = 0.49
    dice_thresholds: List[float] = field(default_factory=lambda: [0.25, 0.5, 0.75])
