import numpy as np
import torch
from PIL import Image, ImageOps


MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def _canvas_to_grayscale(canvas_image: np.ndarray) -> Image.Image:
    image = Image.fromarray(canvas_image.astype("uint8"), mode="RGBA")
    background = Image.new("RGBA", image.size, (0, 0, 0, 255))
    image = Image.alpha_composite(background, image).convert("L")
    return image


def _ensure_mnist_colors(image: Image.Image) -> Image.Image:
    image = ImageOps.autocontrast(image)
    pixels = np.asarray(image)
    border = np.concatenate(
        [
            pixels[0, :],
            pixels[-1, :],
            pixels[:, 0],
            pixels[:, -1],
        ]
    )

    # If the surrounding background is bright, invert to MNIST style:
    # black background and white foreground.
    if np.median(border) > 127:
        return ImageOps.invert(image)
    return image


def _crop_digit(image: Image.Image) -> Image.Image:
    pixels = np.asarray(image)
    ys, xs = np.where(pixels > 24)
    if len(xs) == 0 or len(ys) == 0:
        return Image.new("L", (28, 28), 0)

    left, right = xs.min(), xs.max()
    top, bottom = ys.min(), ys.max()
    return image.crop((left, top, right + 1, bottom + 1))


def _resize_and_center(image: Image.Image) -> Image.Image:
    image = _crop_digit(image)
    if image.getbbox() is None:
        return Image.new("L", (28, 28), 0)

    image = ImageOps.autocontrast(image)
    width, height = image.size
    scale = 21.0 / max(width, height)
    new_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    image = image.resize(new_size, Image.Resampling.LANCZOS)

    canvas = Image.new("L", (28, 28), 0)
    left = (28 - new_size[0]) // 2
    top = (28 - new_size[1]) // 2
    canvas.paste(image, (left, top))

    pixels = np.asarray(canvas).astype(np.float32)
    total_intensity = pixels.sum()
    if total_intensity <= 0:
        return canvas

    y_indices, x_indices = np.indices(pixels.shape)
    center_x = (x_indices * pixels).sum() / total_intensity
    center_y = (y_indices * pixels).sum() / total_intensity
    shift_x = int(round(13.5 - center_x))
    shift_y = int(round(13.5 - center_y))

    shifted = Image.new("L", (28, 28), 0)
    shifted.paste(canvas, (shift_x, shift_y))
    return shifted


def preprocess_canvas(canvas_image: np.ndarray) -> tuple[torch.Tensor, Image.Image]:
    """
    Convert a drawable-canvas RGBA array to a normalized MNIST tensor.

    Returns:
        tensor: shape [1, 1, 28, 28]
        preview: PIL image in MNIST colors for display
    """
    image = _canvas_to_grayscale(canvas_image)
    image = _ensure_mnist_colors(image)
    image = _resize_and_center(image)
    image = ImageOps.autocontrast(image)

    array = np.asarray(image).astype(np.float32) / 255.0
    normalized = (array - MNIST_MEAN) / MNIST_STD
    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
    return tensor, image


@torch.no_grad()
def predict_digit(model: torch.nn.Module, image_tensor: torch.Tensor, device: torch.device):
    model.eval()
    image_tensor = image_tensor.to(device)
    logits = model(image_tensor)
    probabilities = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
    predicted_digit = int(probabilities.argmax())
    return predicted_digit, probabilities
