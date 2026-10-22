
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# Depthwise convolution:
# Apply a separate spatial filter to each channel.
def depthwise_convolution(image, kernel):
    height, width, channels = image.shape
    kh, kw = kernel.shape

    pad_h = kh // 2
    pad_w = kw // 2

    padded = np.pad(
        image,
        ((pad_h, pad_h), (pad_w, pad_w), (0, 0)),
        mode="reflect"
    )

    output = np.zeros_like(image, dtype=np.float32)

    for c in range(channels):
        for i in range(height):
            for j in range(width):
                region = padded[i:i + kh, j:j + kw, c]
                output[i, j, c] = np.sum(region * kernel)

    return output


# Pointwise convolution:
# Use a 1x1 filter to mix information across channels.
def pointwise_convolution(image, kernel):
    height, width, channels = image.shape

    if kernel.shape[0] != channels:
        raise ValueError("Kernel input channels must match image.")

    # kernel shape: (input_channels, output_channels)
    output = np.matmul(image, kernel)

    return output


def normalize_image(image):
    # Convert results into a visible 0-255 image.
    minimum = image.min()
    maximum = image.max()

    if maximum > minimum:
        image = (image - minimum) / (maximum - minimum)
        image = image * 255
    else:
        image = np.zeros_like(image)

    return image.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--flag",
        choices=["depthwise", "pointwise"],
        required=True
    )
    parser.add_argument(
        "--image",
        default="input.jpg"
    )
    parser.add_argument(
        "--output",
        default=None
    )
    args = parser.parse_args()

    # Load image as RGB and resize for faster processing.
    image = Image.open(args.image).convert("RGB")
    image = image.resize((128, 128))
    image_array = np.asarray(image, dtype=np.float32)

    if args.flag == "depthwise":
        # A 3x3 edge detection filter
        kernel = np.array([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ], dtype=np.float32)

        result = depthwise_convolution(image_array, kernel)

    else:
        # 1x1 pointwise filter, one output channel.
        # Convert RGB channels into a grayscale output.
        kernel = np.array([
            [0.299],
            [0.587],
            [0.114]
        ], dtype=np.float32)

        result = pointwise_convolution(image_array, kernel)

    result = normalize_image(result)

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    output_file = args.output or f"results/{args.flag}.png"

    if result.shape[2] == 1:
        result_image = Image.fromarray(result[:, :, 0], mode="L")
    else:
        result_image = Image.fromarray(result, mode="RGB")

    result_image.save(output_file)

    # Display input and output images
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    if result.shape[2] == 1:
        plt.imshow(result[:, :, 0], cmap="gray", vmin=0, vmax=255)
    else:
        plt.imshow(result)

    plt.title(f"{args.flag.capitalize()} Convolution")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(
        f"results/{args.flag}_comparison.png",
        dpi=150
    )
    plt.show()

    print(f"Mode: {args.flag}")
    print(f"Input shape: {image_array.shape}")
    print(f"Kernel shape: {kernel.shape}")
    print(f"Output shape: {result.shape}")
    print(f"Saved output to: {output_file}")


if __name__ == "__main__":
    main()
