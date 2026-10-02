#!/usr/bin/env python3
import os
import random
import math

INPUT_DIR = "input"
COUNT = 120
WIDTH = 256
HEIGHT = 256

os.makedirs(INPUT_DIR, exist_ok=True)

def write_ppm(path, width, height, seed):
    random.seed(seed)
    with open(path, "wb") as f:
        f.write(f"P6\n{width} {height}\n255\n".encode())

        cx = random.randint(width // 4, 3 * width // 4)
        cy = random.randint(height // 4, 3 * height // 4)
        radius = random.randint(30, 90)

        for y in range(height):
            row = bytearray()
            for x in range(width):
                # Gradient background
                r = int(255 * x / (width - 1))
                g = int(255 * y / (height - 1))
                b = int(255 * ((x + y) % width) / (width - 1))

                # Circle
                dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                if dist < radius:
                    r, g, b = 255, 255, 255

                # Repeating stripes
                if ((x + seed) // 20) % 2 == 0:
                    b = min(255, b + 35)

                # Small deterministic noise
                noise = random.randint(-8, 8)
                r = max(0, min(255, r + noise))
                g = max(0, min(255, g + noise))
                b = max(0, min(255, b + noise))

                row.extend((r, g, b))
            f.write(row)

for i in range(COUNT):
    path = os.path.join(INPUT_DIR, f"image_{i+1:03d}.ppm")
    if not os.path.exists(path):
        write_ppm(path, WIDTH, HEIGHT, i + 12345)

print(f"Dataset ready: {COUNT} images of {WIDTH}x{HEIGHT} pixels in {INPUT_DIR}/")
