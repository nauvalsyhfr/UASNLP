import numpy as np

# Simulasi scores array
scores = np.array([0.0, 0.1, 0.0, 0.2, 0.0, 0.3, 0.0])
print(f"Scores: {scores}")
print(f"Scores > 0: {np.sum(scores > 0)}")

# Test np.where
idxs = np.where(scores > 0)[0]
print(f"Indices where scores > 0: {idxs}")
print(f"Number of indices: {len(idxs)}")

# Sort by score
idxs_sorted = idxs[scores[idxs].argsort()[::-1]]
print(f"Sorted indices: {idxs_sorted}")
print(f"Sorted scores: {scores[idxs_sorted]}")
