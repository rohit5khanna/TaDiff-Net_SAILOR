# Time-Dependent Lambda Schedule Experiment

## Overview

This document describes the **time-dependent lambda schedule** experiment for TaDiff-Net. The experiment modifies the joint loss function (Equation 16 in the paper) to use a dynamically weighted segmentation loss instead of a fixed weight.

**Branch**: `feature/lambda-schedule`

## Motivation

### Original Paper (Fixed Lambda)

The TaDiff paper uses a fixed segmentation loss weight:

$$L = \|(\mathbf{x}_t - \hat{\epsilon})\|^2 + \lambda \cdot L_{seg}$$

where $\lambda = 0.01$ is constant across all timesteps $t \in [1, T]$.

### Problem with Fixed Lambda

At different noise levels, the quality of segmentation signal varies:

1. **High noise (t close to T)**: The image is heavily corrupted by noise. The segmentation masks are also noisy and unreliable. Weighting the segmentation loss equally at this stage can:
   - Lead to noisy gradient updates
   - Cause training instability
   - Distract the model from learning the denoising task

2. **Low noise (t close to 0)**: The image is nearly clean with strong segmentation signal. The model should focus on accurate segmentation since:
   - Image reconstruction is trivial at low noise
   - Segmentation masks are clear and meaningful
   - Accurate masks improve tumor tracking

### Our Solution: Time-Dependent Lambda

We propose:

$$\lambda(t) = \lambda_0 \cdot \bar{\alpha}_t^k$$

where:
- $\lambda_0 = 0.01$ (base weight from paper)
- $\bar{\alpha}_t$ is the cumulative product of alphas at timestep $t$ (SNR proxy)
- $k = 2$ (exponent controlling schedule shape)

**Effect**: Segmentation loss is weighted less at high noise (where signal is weak) and more at low noise (where signal is strong).

## Mathematical Formulation

### Original Loss (Fixed Lambda)

$$L_{total} = \mathbb{E}_{t,\epsilon,\mathbf{x}_0} \left[ \|\epsilon - \hat{\epsilon}_\theta(x_t, t, \mathbf{c})\|^2 + 0.01 \cdot L_{seg} \right]$$

### Proposed Loss (Time-Dependent Lambda)

$$L_{total} = \mathbb{E}_{t,\epsilon,\mathbf{x}_0} \left[ \|\epsilon - \hat{\epsilon}_\theta(x_t, t, \mathbf{c})\|^2 + \lambda(t) \cdot L_{seg} \right]$$

where

$$\lambda(t) = 0.01 \cdot \bar{\alpha}_t^2$$

### Interpretation

- $\bar{\alpha}_t = \prod_{i=1}^{t} \alpha_i$ (signal-to-noise ratio proxy)
- $\bar{\alpha}_t \in [0, 1]$, starting high and decreasing toward 0 as $t$ increases
- $\bar{\alpha}_t^2$ decreases faster than $\bar{\alpha}_t$, making the schedule more aggressive

**Schedule behavior**:
- At $t = 1$ (low noise): $\bar{\alpha}_1 \approx 0.99 \Rightarrow \lambda(1) \approx 0.0099$ (close to original)
- At $t = T/2$ (mid noise): $\bar{\alpha}_{T/2} \approx 0.5 \Rightarrow \lambda(T/2) \approx 0.0025$ (4× lower)
- At $t = T$ (high noise): $\bar{\alpha}_T \approx 0.01 \Rightarrow \lambda(T) \approx 0.0001$ (100× lower)

## Implementation Details

### Code Changes

**File**: `src/tadiff_model.py`, lines 209-226

```python
# Compute time-dependent lambda: lambda(t) = lambda_0 * alphabar_t^k
k = 2.0  # exponent controlling how much lambda varies with time
lambda_t = self.cfg.aux_loss_w * (alphabar_t ** k)  # shape: (b,)

# Apply time-dependent weighting to dice loss
weighted_dice_loss = torch.mean(dice_loss) * lambda_t.view(-1, 1, 1, 1).mean()

loss = loss1 + weighted_dice_loss
```

### Key Parameters

- **`aux_loss_w`** (in config): Base segmentation loss weight ($\lambda_0 = 0.01$)
- **`k`** (hardcoded in code): Exponent for schedule ($k = 2$)

To adjust the schedule aggressiveness:
- Increase `k` → more aggressive schedule (larger difference between high/low noise)
- Decrease `k` → more conservative schedule (closer to fixed lambda)
- `k = 1` → $\lambda(t) = \lambda_0 \cdot \bar{\alpha}_t$ (linear schedule)
- `k = 0` → $\lambda(t) = \lambda_0$ (back to original fixed weight)

## Alternative Approaches Considered

### 1. **Cosine Schedule** (Not Selected)

$$\lambda(t) = \lambda_0 \cdot \cos\left(\frac{\pi \bar{\alpha}_t}{2}\right)^2$$

**Pros**:
- Smooth, differentiable schedule
- Used in other diffusion models (e.g., EDM)
- Matches curriculum learning literature

**Cons**:
- More complex
- Harder to interpret
- No clear advantage over power schedule for this task

**Decision**: Power schedule is simpler and more intuitive.

### 2. **Linear Schedule** ($k=1$)

$$\lambda(t) = \lambda_0 \cdot \bar{\alpha}_t$$

**Pros**:
- Simplest non-fixed schedule
- Linear relationship between SNR and loss weight

**Cons**:
- Less aggressive scaling
- May not sufficiently reduce noise impact at high $t$

**Decision**: We chose $k=2$ (quadratic) as a middle ground.

### 3. **Step Function / Threshold Schedule** (Not Selected)

$$\lambda(t) = \begin{cases}
\lambda_0 & \text{if } \bar{\alpha}_t > 0.5 \\
\lambda_0 / 10 & \text{otherwise}
\end{cases}$$

**Pros**:
- Drastic reduction at high noise
- Easy to implement

**Cons**:
- Discontinuous gradient
- Sharp transition can cause training instability
- Hyperparameter sensitivity ($0.5$ threshold)

**Decision**: Smooth power schedule avoids these issues.

### 4. **Learnable Schedule** (Future Work)

Instead of a fixed formula, learn $\lambda(t)$ during training using a neural network:

$$\lambda(t) = \text{NN}(t, \bar{\alpha}_t)$$

**Pros**:
- Maximum flexibility
- Model learns optimal schedule

**Cons**:
- Adds complexity and hyperparameters
- Harder to interpret
- Risk of overfitting to training distribution

**Decision**: Save for future work after validating the fixed schedule.

## Expected Outcomes

### Hypothesis

We expect the time-dependent lambda schedule to improve:

1. **Training Stability**: Reduced noise in gradient updates at high noise levels
2. **Segmentation Accuracy**: Better masks at low noise where signal is clear
3. **Convergence Speed**: Faster convergence due to curriculum-like weighting
4. **Generalization**: Better test-time segmentation performance

### Potential Risks

1. **Over-weighting Segmentation**: If $k$ is too large, segmentation loss dominates at low noise, hurting image generation
2. **Under-weighting Segmentation**: If $k$ is too small, schedule has negligible effect
3. **Training Divergence**: Aggressive schedule could cause instability if not tuned correctly

## Experimental Setup

### Baseline (Main Branch)

- **Config**: `config/cfg_tadiff_net.py` (unchanged)
- **Loss**: Fixed $\lambda = 0.01$
- **WandB Run**: `main` branch logs

### Lambda Schedule (This Branch)

- **Config**: Same as baseline
- **Loss**: Time-dependent $\lambda(t) = 0.01 \cdot \bar{\alpha}_t^2$
- **WandB Run**: `feature/lambda-schedule` logs

### Comparison Metrics

We'll compare via WandB dashboard:

| Metric | Description |
|--------|-------------|
| `train_loss` | Total training loss |
| `train_mse` | Diffusion (image) loss component |
| `train_dice` | Segmentation loss component |
| `val_loss` | Validation total loss |
| `val_dice` | Validation Dice score |
| `val_mse` | Validation image MSE |

### WandB Tracking

All runs automatically log to: https://wandb.ai/rkhanna5/TaDiff_Net

Use WandB's parallel coordinates or scatter plots to:
- Compare loss curves across runs
- Analyze convergence speed
- Identify potential instability

## Usage

### Training with Lambda Schedule

On TACC:

```bash
cd /work/11343/rohitk59/ls6/TaDiff_Baseline/TaDiff-Net_SAILOR

# Checkout the lambda schedule branch
git fetch origin
git checkout feature/lambda-schedule

# Activate venv
source tadiff/bin/activate

# Run training
sbatch train_tacc.sh
```

### Toggling Back to Fixed Lambda

To compare both approaches, switch branches:

```bash
# Back to fixed lambda
git checkout main

# Back to time-dependent lambda
git checkout feature/lambda-schedule
```

## Future Directions

1. **Hyperparameter Tuning**: Experiment with different $k$ values (1.0, 1.5, 2.0, 2.5, 3.0)
2. **Adaptive Schedule**: Learn $k$ during training
3. **Segmentation-Specific Loss**: Replace Dice with focal loss or Lovasz loss
4. **Multi-Task Weighting**: Dynamically weight different tasks (image generation vs segmentation)
5. **Comparison with Other Papers**: Compare to lambda schedules in recent diffusion papers

## References

### TaDiff Paper

- Liu, S., et al. "TaDiff: Time-aware Diffusion Model for Longitudinal 3D Medical Image Synthesis." (2023)
- Original fixed lambda: $\lambda = 0.01$ (Equation 16)

### Related Work on Curriculum Learning & Loss Weighting

- Bengio et al. (2009). "Curriculum Learning" — Motivates easier tasks first
- Graves et al. (2017). "Automated Curriculum Learning for Reinforcement Learning"
- OhSawa et al. (2021). "Curriculum Learning as a Source Selection Strategy"

### Diffusion Model References

- Nichol & Dhariwal (2021). "Improved Denoising Diffusion Probabilistic Models"
- Karras et al. (2022). "Elucidating the Design Space of Diffusion-Based Generative Models (EDM)"

## Contact & Questions

If you have questions or suggestions about this experiment, please reach out!

---

**Last Updated**: February 2025
**Experiment Status**: Active
**Branch**: `feature/lambda-schedule`
