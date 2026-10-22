
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt



# 1. Depthwise Convolution
# Apply a spatial filter to each image channel separately


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



# 2. Pointwise Convolution
# Use 1x1 filters to combine information across channels


def pointwise_convolution(image, kernel):
    height, width, channels = image.shape

    if kernel.shape[0] != channels:
        raise ValueError(
            "Kernel input channels must match image channels."
        )

    # Kernel shape: (input_channels, output_channels)
    output = np.matmul(image, kernel)

    return output



# 3. Normalize Image
# Convert convolution results into visible 0-255 values


def normalize_image(image):
    minimum = image.min()
    maximum = image.max()

    if maximum > minimum:
        image = (image - minimum) / (maximum - minimum)
        image = image * 255
    else:
        image = np.zeros_like(image)

    return image.astype(np.uint8)



# 4. Main Program


def main():


    # Step 1: Read command-line arguments

    parser = argparse.ArgumentParser(
        description="Depthwise and Pointwise Convolution"
    )

    parser.add_argument(
        "--flag",
        choices=["depthwise", "pointwise"],
        required=True,
        help="Choose convolution type"
    )

    parser.add_argument(
        "--image",
        default="input.jpg",
        help="Path to input image"
    )


    parser.add_argument(
        "--output",
        default=None,
        help="Path to save output image"
    )

    parser.add_argument(
        "--kernel",
        type=str,
        default=None,
        help="Path to an external .npy kernel file"
    )

    args = parser.parse_args()


    # Step 2: Load input image


    image_path = Path(args.image)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Input image not found: {image_path}"
        )

    image = Image.open(image_path).convert("RGB")
    image = image.resize((128, 128))

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    print("\n========== INPUT ==========")
    print(f"Image: {args.image}")
    print(f"Input shape: {image_array.shape}")


    # Step 3: Load external or default kernel


    if args.flag == "depthwise":

        if args.kernel:
            kernel = np.load(
                args.kernel,
                allow_pickle=False
            ).astype(np.float32)

        else:
            # Default 3x3 edge detection filter
            kernel = np.array([
                [-1, -1, -1],
                [-1,  8, -1],
                [-1, -1, -1]
            ], dtype=np.float32)

        # Validate depthwise kernel
        if (
            kernel.ndim != 2
            or any(size % 2 == 0 for size in kernel.shape)
        ):
            raise ValueError(
                "Depthwise kernel must be a 2D array "
                "with odd dimensions."
            )

        print("\nUsing Depthwise Convolution")

        result = depthwise_convolution(
            image_array,
            kernel
        )

    else:

        if args.kernel:
            kernel = np.load(
                args.kernel,
                allow_pickle=False
            ).astype(np.float32)

        else:
            # Default RGB-to-grayscale filter
            kernel = np.array([
                [0.299],
                [0.587],
                [0.114]
            ], dtype=np.float32)

        # Validate pointwise kernel
        if (
            kernel.ndim != 2
            or kernel.shape[0] != image_array.shape[2]
            or kernel.shape[1] not in (1, 3)
        ):
            raise ValueError(
                "Pointwise kernel must have shape "
                "(3, 1) or (3, 3)."
            )

        print("\nUsing Pointwise Convolution")

        result = pointwise_convolution(
            image_array,
            kernel
        )


    # Step 4: Process output image


    if args.flag == "depthwise":
        # Normalize signed edge responses for visualization.
        result = normalize_image(result)
    else:
        # Preserve the RGB/grayscale scale.
        result = np.clip(
            result, 0, 255
        ).astype(np.uint8)


    # Step 5: Create output directory


    output_dir = Path("results")
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Determine kernel name
    if args.kernel:
        kernel_name = Path(args.kernel).stem
    else:
        kernel_name = "default"

    # Determine output filename
    output_file = (
        Path(args.output)
        if args.output
        else output_dir / f"{args.flag}_{kernel_name}.png"
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    # Step 6: Save output image


    if result.shape[2] == 1:
        result_image = Image.fromarray(
            result[:, :, 0]
        )
    else:
        result_image = Image.fromarray(result)

    result_image.save(output_file)


    # Step 7: Generate comparison image


    plt.figure(figsize=(10, 4))

    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    # Convoluted image
    plt.subplot(1, 2, 2)

    if result.shape[2] == 1:
        plt.imshow(
            result[:, :, 0],
            cmap="gray",
            vmin=0,
            vmax=255
        )
    else:
        plt.imshow(result)

    plt.title(
        f"{args.flag.capitalize()} Convolution"
    )
    plt.axis("off")

    plt.tight_layout()

    comparison_file = (
        output_dir
        / f"{args.flag}_{kernel_name}_comparison.png"
    )

    plt.savefig(
        comparison_file,
        dpi=150
    )

    plt.show()
    plt.close()


    # Step 8: Print results


    print("\n========== RESULTS ==========")
    print(f"Mode: {args.flag}")
    print(f"Kernel shape: {kernel.shape}")
    print(f"Output shape: {result.shape}")
    print(f"Output image: {output_file}")
    print(f"Comparison image: {comparison_file}")

    print("\nConvolution completed successfully!")



# 5. Run Program


if __name__ == "__main__":
    main()
