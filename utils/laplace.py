from laplace_ops import signed_laplacian


def rgb_to_gray(image):
    """Convert an NCHW [0, 255] RGB image to conventional grayscale values."""
    if image.ndim != 4:
        raise ValueError(f"Expected NCHW image, got shape {tuple(image.shape)}")
    if image.shape[1] == 1:
        return image
    if image.shape[1] != 3:
        raise ValueError(f"Expected 1 or 3 channels, got {image.shape[1]}")
    coefficients = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return (image * coefficients).sum(dim=1, keepdim=True)


def rgb_to_studio_y(image):
    """Convert an NCHW [0, 255] RGB image to the studio-range Y channel."""
    if image.ndim != 4:
        raise ValueError(f"Expected NCHW image, got shape {tuple(image.shape)}")
    if image.shape[1] == 1:
        return image
    if image.shape[1] != 3:
        raise ValueError(f"Expected 1 or 3 channels, got {image.shape[1]}")
    coefficients = image.new_tensor([65.481, 128.553, 24.966]).view(1, 3, 1, 1)
    return (image * coefficients).sum(dim=1, keepdim=True) / 255.0 + 16.0


def laplacian_map(image):
    """Weighted absolute channel-wise Laplacian for NCHW RGB/grayscale [0, 255].

    RGB: 0.299*abs(L(R)) + 0.587*abs(L(G)) + 0.114*abs(L(B)).
    Grayscale: abs(L(Y)). Returns N1HW, using replicate boundaries and
    the normalized eight-neighbor kernel. No intermediate rounding/clipping.
    """
    return rgb_to_gray(signed_laplacian(image).abs())
