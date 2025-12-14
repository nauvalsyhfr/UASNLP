import numpy as np
import os

print("Checking embeddings files...")
print("=" * 70)

# Check embeddings
emb_path = "doc_embeddings_minilm.npy"
if os.path.exists(emb_path):
    embeddings = np.load(emb_path)
    print(f"✓ Embeddings loaded")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Expected: (3137, 384)")
    print(f"  Match: {embeddings.shape[0] == 3137}")
else:
    print(f"✗ Embeddings not found: {emb_path}")

# Check TF-IDF
from scipy.sparse import load_npz
import joblib

tfidf_matrix_path = "tfidf_matrix.npz"
tfidf_vec_path = "tfidf_vectorizer.joblib"

if os.path.exists(tfidf_matrix_path):
    tfidf_matrix = load_npz(tfidf_matrix_path)
    print(f"\n✓ TF-IDF matrix loaded")
    print(f"  Shape: {tfidf_matrix.shape}")
    print(f"  Expected: (3137, ...)")
    print(f"  Match: {tfidf_matrix.shape[0] == 3137}")
else:
    print(f"\n✗ TF-IDF matrix not found")

if os.path.exists(tfidf_vec_path):
    vectorizer = joblib.load(tfidf_vec_path)
    print(f"\n✓ TF-IDF vectorizer loaded")
    print(f"  Features: {len(vectorizer.get_feature_names_out())}")
else:
    print(f"\n✗ TF-IDF vectorizer not found")