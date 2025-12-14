# generate_embeddings.py
"""
Generate pre-computed embeddings untuk MedicIR
Run sekali saja, hasil disimpan di file .npy
"""

from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import os

print("=" * 70)
print("Generating Pre-computed Embeddings for MedicIR")
print("=" * 70)

# 1. Load model
print("\n1. Loading MiniLM model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("✓ Model loaded")

# 2. Load dataset
print("\n2. Loading dataset...")
data_path = "final_clean_data_20112024_halodoc_based.csv"
df = pd.read_csv(data_path)
print(f"✓ Loaded {len(df)} documents")

# 3. Create combined text
print("\n3. Creating combined text...")
cols_to_combine = ['nama', 'deskripsi', 'indikasi_umum', 'komposisi']
valid_cols = [c for c in cols_to_combine if c in df.columns]

df['combined_text'] = (
    df[valid_cols]
    .fillna('')
    .agg(' '.join, axis=1)
    .str.lower()
)
print(f"✓ Combined {len(valid_cols)} columns")

# 4. Generate embeddings
print("\n4. Generating embeddings (this may take a few minutes)...")
print("   Please wait...")

embeddings = model.encode(
    df['combined_text'].tolist(),
    convert_to_numpy=True,
    show_progress_bar=True,
    batch_size=32  # Adjust based on your RAM
)

print(f"✓ Generated embeddings shape: {embeddings.shape}")

# 5. Save embeddings
print("\n5. Saving embeddings...")
output_path = "doc_embeddings_minilm.npy"
np.save(output_path, embeddings)
print(f"✓ Saved to: {output_path}")

# 6. Verify
print("\n6. Verifying saved file...")
loaded = np.load(output_path)
print(f"✓ Verification successful: {loaded.shape}")

print("\n" + "=" * 70)
print("✅ EMBEDDINGS GENERATION COMPLETE!")
print("=" * 70)
print(f"\nFile saved: {os.path.abspath(output_path)}")
print(f"File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
print("\nNow restart your MedicIR server to use pre-computed embeddings.")
print("This will make search MUCH FASTER!")
print("=" * 70)