"""Shared signed Laplacian for network inputs and texture routing."""
import torch.nn.functional as F


def signed_laplacian(image):
    """Return NCHW signed responses with the same units as RGB/grayscale input.

    Center +1, eight neighbors -1/8; replicate padding. Channels remain
    separate. No absolute value, clipping, rounding or extra normalization.
    """
    if image.ndim != 4 or image.shape[1] not in (1, 3):
        raise ValueError(f"Expected N1HW or N3HW image, got shape {tuple(image.shape)}")
    if not image.is_floating_point():
        image = image.float()
    channels = image.shape[1]
    kernel = image.new_tensor(
        [[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]]
    ).view(1, 1, 3, 3).repeat(channels, 1, 1, 1) / 8
    return F.conv2d(F.pad(image, (1, 1, 1, 1), mode="replicate"),
                    kernel, groups=channels)
