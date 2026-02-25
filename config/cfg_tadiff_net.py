from munch import DefaultMunch

# -----------------------------------------------
# model config
network = 'TaDiff_Net'
data_pool = ['sailor']

# UPDATE THIS PATH to your local preprocessed data directory
# On TACC ls6: /work/11343/rohitk59/ls6/TaDiff_Baseline/TaDiff-Net_SAILOR/data/sailor/sailor_processed
data_dir = {'sailor': '/work/11343/rohitk59/ls6/TaDiff_Baseline/TaDiff-Net_SAILOR/data/sailor/sailor_processed'}

image_size = 192
in_channels = 13
out_channels = 7
num_intv_time = 3

# ---------- PAPER VALUES (Table I, Section III-A) ----------
# The original repo had debug values (model_channels=32, channel_mult=(1,2,3,4)).
# Below are the correct values from the published paper:
#   "channel widths [64, 128, 256, 512]" → model_channels=64, channel_mult=(1,2,4,8)
model_channels = 64            # was 32 (debug), paper says 64
num_res_blocks = 2
channel_mult = (1, 2, 4, 8)   # was (1,2,3,4), paper says [64,128,256,512]
attention_resolutions = [8, 4]
num_heads = 4
num_classes = 81  # treat_code

max_T = 1000 # diffusion steps (paper: T=1000 for training)
ddpm_schedule = 'linear' # 'linear', 'cosine', 'log'

# -----------------------------------------------
# optimizer, lr, loss, train config
# ---------- PAPER VALUES (Section III-B) ----------
opt = 'adamw' # adam, adamw, sgd, adan

lr = 2.5e-4              # was 5e-3 (debug), paper says 2.5e-4
max_epochs = 0            # set to 0 to use max_steps instead (paper trains by iterations)
max_steps = 5000000       # was 60000 (debug), paper says 5M iterations
weight_decay = 3e-5
lrdecay_cosine = True
lr_gamma = 0.585
warmup_steps = 1000       # was 100 (debug), paper says 1000

loss_type = 'mse'         # paper uses MSE for diffusion loss (Eq. 16)
aux_loss_w = 0.01         # ADDED: lambda in joint loss (Eq. 16), referenced in tadiff_model.py line 209

batch_size = 1             # per-GPU batch size for 3D volumes (each yields sw_batch 2D slices)
sw_batch = 16              # number of 2D slices sampled per volume
num_workers = 8

grad_clip = 1.5
accumulate_grad_batches = 2  # was 4 (debug), paper says 2. Effective batch = batch_size * n_gpu * accum
n_repeat_tr = 10   # simulate larger train dataset by repeating it
n_repeat_val = 5   # simulate larger val data by repeating

cache_rate = 0.  # cache rate for MONAI CacheDataset (0=no cache, 1=all in memory)


# -----------------------------------------------
# I/O, system and log config for trainer (e.g. lightning)
wandb_entity = "qhliu"    # UPDATE to your own wandb entity (or will use TensorBoard fallback)
logdir = '/work/11343/rohitk59/ls6/TaDiff_Baseline/tadiff_ckpts'
log_interval = 1
seed = 114514

# UPDATE these based on your hardware:
gpu_devices = '0'          # was '0, 1' — set based on available GPUs
gpu_strategy = "auto"      # was "ddp" — use "auto" for single GPU, "ddp" for multi-GPU
gpu_accelerator = "gpu"
precision = 32             # 16-mixed, 32

val_interval_epoch = 10
resume_from_ckpt = False
ckpt_best_or_last = None   # if not None, will load ckpt for val or resume training
ckpt_save_top_k = 3
ckpt_save_last = True
ckpt_monitor = "val_loss"
ckpt_filename = "ckpt-{epoch}-{step}-{val_loss:.6f}"
ckpt_mode = "min"

do_train_only = False
do_test_only = False

# -----------------------------------------------
# patient split for train/val (SAILOR dataset, 25 valid patients)
# The paper used leave-one-out cross-validation.
# Default: use sub-17 as test, rest for train/val
train_patients = [f'sub-{i:02d}' for i in range(1, 28) if i not in [17, 13, 23]]  # exclude test + QC-failed
val_patients = ['sub-17']
# NOTE: sub-13 and sub-23 had QC issues in the original dataset

# -----------------------------------------------
# -----------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and
               isinstance(v, (int, float, bool, str, list, tuple, dict))]
config = {k: globals()[k] for k in config_keys}
config = DefaultMunch.fromDict(config)

# # print config for debug
# for key, value in config.items():
#     print(key, value)
