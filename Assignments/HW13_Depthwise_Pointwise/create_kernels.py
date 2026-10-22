
import numpy as np

# Depthwise: 3x3 Blur Kernel
blur_kernel = np.ones((3, 3), dtype=np.float32) / 9
np.save("blur_kernel.npy", blur_kernel)

# Depthwise: 3x3 Edge Detection Kernel
edge_kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
], dtype=np.float32)
np.save("edge_kernel.npy", edge_kernel)

# Pointwise: RGB to Grayscale Kernel
gray_kernel = np.array([
    [0.299],
    [0.587],
    [0.114]
], dtype=np.float32)
np.save("gray_kernel.npy", gray_kernel)

# Pointwise: Color Mixing Kernel
color_kernel = np.array([
    [0.5, 0.5, 0.0],
    [0.0, 0.5, 0.5],
    [0.5, 0.0, 0.5]
], dtype=np.float32)
np.save("color_kernel.npy", color_kernel)

print("Four kernel files created successfully!")
