import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from audio.mfcc import extract_mfcc
audio = np.load("tests/mic_audio.npy")

mfcc = extract_mfcc(audio)

print("MFCC shape:", mfcc.shape)
print("MFCC dtype:", mfcc.dtype)

plt.figure(figsize=(12, 5))

plt.imshow(
    mfcc.T,
    origin="lower",
    aspect="auto"
)

plt.colorbar(label="MFCC Value")
plt.title("MFCC Features")
plt.xlabel("Time Frame")
plt.ylabel("MFCC Coefficient")

plt.tight_layout()
plt.show()