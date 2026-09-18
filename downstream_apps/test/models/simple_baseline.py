"""
A simple linear regression model to be used as a baseline for flare forecasting.
"""

import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F


def destandardize_channels(batch: dict, channel_order: list, scalers: dict) -> dict:
    """Return a new batch dict with 'ts' moved from normalized space to signum-log space.

    This undoes the per-channel z-score ONLY. The signum-log compression applied by the
    dataset is deliberately left in place, so the result is
    ``sign(x*s) * log1p(|x*s|)`` — not raw DN/Gauss. Values spanning many orders of
    magnitude make poor features for a single linear layer, so log space is what the
    baseline wants.

    If you need true physical units (plotting, a physical-space loss), use
    ``HelioNetCDFDataset.inverse_transform_data()`` instead, which undoes both stages.
    See the "THE THREE SPACES" block in ``workshop_infrastructure/datasets/helio.py``.

    Args:
        batch: Batch dict containing at minimum a 'ts' key with shape (B, C, T, H, W).
        channel_order: Channel names in the same order as the C dimension of 'ts'.
        scalers: Dict mapping channel name -> scaler with an inverse_transform method.

    Returns:
        A new batch dict with 'ts' replaced by the de-standardized (signum-log) tensor.
    """
    x = batch["ts"].clone()
    with torch.no_grad():
        for i, channel in enumerate(channel_order):
            x[:, i, ...] = scalers[channel].inverse_transform(x[:, i, ...])
    return {**batch, "ts": x}


class RegressionFlareModel(nn.Module):
    def __init__(self, input_dim: int):
        """
        Initializes the RegressionFlareModel.

        Args:
            input_dim (int): The size of the input vector after channel and time dimensions are flattened.

        Note:
            This model expects 'ts' in the batch dict to already be in **signum-log** space
            (channel z-scores undone, log compression retained). Use
            destandardize_channels() to pre-process normalized SDO inputs before passing
            them here (e.g., via the preprocess_fn argument of FlareLightningModule).
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: dict) -> torch.Tensor:
        """
        Performs a forward pass through the model.

        Args:
            x (dict): Batch dict with 'ts' of shape (B, C, T, H, W) in signum-log space.

        B - Batch size
        C - Channels
        T - Time steps
        H - Height
        W - Width
        """
        x = x["ts"]

        # Collapse input stack spatially and take absolute value for strictly positive flare fluxes
        x = x.abs().mean(dim=[3, 4])

        # Rearrange in preparation for linear layer
        x = rearrange(x, "b c t -> b (c t)")

        return self.linear(x)


class ConstantProbabilityModel(nn.Module):
    """Trainable baseline: a single learned logit, independent of the input.

    Learns the class base rate (the wave / no-wave prior) and nothing else — the
    simplest possible baseline against which any input-dependent model should be judged.
    """

    def __init__(self):
        super().__init__()
        self.logit = nn.Parameter(torch.zeros(1))

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Args:
            batch (dict): Batch dict; only used to read the batch size from
                ``batch["forecast"]`` (the target), since the model itself ignores
                the input entirely.
        """
        batch_size = batch["forecast"].shape[0]
        return self.logit.expand(batch_size, 1)

def mean_abs_pooled_running_difference(
    ts_signum_log: torch.Tensor,
    channel_index: int,
    pool_kernel: int,
    ) -> torch.Tensor:
    """Collapse a running-difference image into one scalar per sample.

    ``mean(|pool_K(now - prev)|)``: mean-pool the running difference by ``pool_kernel``,
    take the absolute value, then average over the whole frame. Pooling first is the
    point — a wave front is a coherent, low-amplitude brightening spread over many
    pixels, so a 32x32 block mean adds it up while averaging pixel noise down.

    **The order is not interchangeable.** Mean pooling composes (pooling a kernel-8 map
    by a further 4 equals pooling the original by 32), which is what lets
    ``extract_baseline_features.py`` cache one resolution and serve a whole kernel sweep.
    ``abs()`` does not: ``mean(|pool8(d)|) != mean(|pool32(d)|)``, because the absolute
    value stops opposite-signed blocks from cancelling. Any code deriving this scalar
    from a cached map must therefore re-pool to ``pool_kernel`` *first*, then ``abs``,
    then the spatial mean.

    Args:
        ts_signum_log: ``(B, C, T, H, W)`` in **signum-log** space (channel z-score
            undone, log compression retained) — see ``destandardize_channels()``.
        channel_index: Index of the channel to difference within the C dimension.
        pool_kernel: Mean-pool kernel size (and stride).

    Returns:
        ``(B, 1)`` — one scalar per sample, shaped ready for ``nn.Linear(1, 1)``.
    """
    x = ts_signum_log[:, channel_index]                                   # (B, T, H, W)
    diff = x[:, -1] - x[:, 0]                                             # (B, H, W)
    pooled = F.avg_pool2d(diff.unsqueeze(1), kernel_size=pool_kernel)     # (B, 1, S, S)
    return pooled.abs().mean(dim=(-2, -1))                                # (B, 1)


def pooled_running_difference_stats(
    ts_signum_log: torch.Tensor,
    channel_index: int,
    pool_kernel: int,
) -> torch.Tensor:
    """Collapse a running-difference image into a 5-number summary per sample.

    Same first two steps as ``mean_abs_pooled_running_difference()`` — mean-pool the
    running difference by ``pool_kernel``, THEN take the absolute value (see that
    function's docstring for why the order matters: it lets incoherent pixel noise
    partially cancel during pooling while a coherent brightening survives it). Instead
    of collapsing the resulting |pooled diff| map to a single spatial mean, this keeps
    five statistics of its distribution across the (S, S) grid of blocks:

        [p98, p95, p90, mean, std]

    The percentiles are sensitive to a small number of strongly-brightened blocks (a
    localized wave front against a quiet background); mean and std describe the overall
    level and spread. Together they give the linear layer more to work with than a
    single scalar — at the cost of turning ``nn.Linear(1, 1)`` into ``nn.Linear(5, 1)``
    (6 trainable parameters instead of 2).

    Args:
        ts_signum_log: ``(B, C, T, H, W)`` in signum-log space (channel z-score undone,
            log compression retained) — see ``destandardize_channels()``.
        channel_index: Index of the channel to difference within the C dimension.
        pool_kernel: Mean-pool kernel size (and stride), applied before the absolute
            value.

    Returns:
        ``(B, 5)`` tensor: ``[p98, p95, p90, mean, std]`` of ``|pool_K(now - prev)|``,
        each computed over the spatial (S, S) grid of pooled blocks for that sample.
    """
    x = ts_signum_log[:, channel_index]                                   # (B, T, H, W)
    diff = x[:, -1] - x[:, 0]                                             # (B, H, W)
    pooled = F.avg_pool2d(diff.unsqueeze(1), kernel_size=pool_kernel)     # (B, 1, S, S)
    pooled_abs = pooled.abs().flatten(start_dim=1)                       # (B, S*S)

    q = torch.tensor([0.98, 0.95, 0.90], device=pooled_abs.device, dtype=pooled_abs.dtype)
    p98, p95, p90 = torch.quantile(pooled_abs, q, dim=1)                 # each (B,)
    mean = pooled_abs.mean(dim=1)                                        # (B,)
    std = pooled_abs.std(dim=1)                                          # (B,)

    return torch.stack([p98, p95, p90, mean, std], dim=1)                # (B, 5)

class RunningDifferenceLogisticModel(nn.Module):
    """Trainable baseline: logistic regression on ONE scalar per sample.

    The scalar is ``mean(|32x32 mean pool of AIA193(now) - AIA193(prev)|)`` in signum-log
    space — see ``mean_abs_pooled_running_difference()``. A single ``nn.Linear(1, 1)``
    turns it into a logit; ``BCEWithLogitsLoss`` supplies the sigmoid.

    Two parameters, which is the whole idea. The previous version fed the entire 128x128
    pooled map into ``nn.Linear(16384, 1)`` and memorized its 384 training samples
    (train loss -> 0.0065, validation AUROC frozen at 0.4). The pooled *image* is a good
    picture of a wave; it is a poor design matrix at this sample size.

    Because the readout is monotone in a single feature, the model is a threshold detector:
    the learned weight sets the sign and the scale, the bias sets the threshold, and
    ranking metrics (AUROC) are a fixed property of the feature rather than something
    training can improve. Only the loss and the calibration are learned.

    Note:
        Expects 'ts' with shape (B, C, T=2, H, W), where T is ordered [t_prev, t_now] —
        ``HelioNetCDFDataset._get_index_data`` always places the offset-0 ("now") frame
        last, so ``x[:, -1] - x[:, 0]`` is the forward-in-time difference.

        Expects 'ts' in **signum-log** space (channel z-scores undone, log compression
        retained), same convention as ``RegressionFlareModel``. Use
        ``destandardize_channels()`` to get there from what the dataset returns (e.g. via
        the ``preprocess_fn`` argument of ``WaveLightningModule``).

        The feature is small — a mean of absolute log-space differences, order 0.1 — while
        the logit needs a weight of order 10-100 to separate the classes. Adam moves a
        parameter by roughly ``lr`` per step, so the ``learning_rate: 0.01`` that suited a
        16384-dimensional input is too small here by a couple of orders of magnitude.
        This affects the loss and the threshold only; AUROC is unchanged either way.
    """

    def __init__(self, channel_index: int, pool_kernel: int = 32):
        """
        Args:
            channel_index (int): Index of AIA193 within the C dimension of 'ts'. Note this
                differs per config — 3 in the 13-channel configs, 0 when a caller narrows
                ``data.channels`` to ``["aia193"]`` — so pass
                ``cfg.data.channels.index("aia193")`` rather than a literal.
            pool_kernel (int): Mean-pool kernel size (and stride) applied before the
                absolute value. Defaults to 32, the scale at which the wave front is
                visible by eye in the task 1.1 figures.
        """
        super().__init__()
        self.channel_index = channel_index
        self.pool_kernel = pool_kernel
        # 1 -> 1: two parameters, a slope and a threshold. img_size is irrelevant now,
        # because the readout width no longer depends on the image resolution.
        self.linear = nn.Linear(5, 1)

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Args:
            batch (dict): Batch dict with 'ts' of shape (B, C, T=2, H, W) in signum-log space.

        Returns:
            ``(B, 1)`` logits.
        """
        #feature = mean_abs_pooled_running_difference(
        #    batch["ts"], self.channel_index, self.pool_kernel
        #)
        features = pooled_running_difference_stats(
            batch["ts"], self.channel_index, self.pool_kernel
        )
        return self.linear(features)