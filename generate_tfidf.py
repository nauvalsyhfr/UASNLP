# generate_tfidf.py
"""
Generate pre-computed TF-IDF matrix untuk MedicIR
Run sekali saja, hasil disimpan di file .npz dan .joblib
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import joblib
from scipy.sparse import save_npz
import os

print("=" * 70)
print("Generating Pre-computed TF-IDF Matrix for MedicIR")
print("=" * 70)

# 1. Load dataset
print("\n1. Loading dataset...")
data_path = "final_clean_data_20112024_halodoc_based.csv"
df = pd.read_csv(data_path)
print(f"✓ Loaded {len(df)} documents")

# 2. Create combined text
print("\n2. Creating combined text...")
cols_to_combine = ['nama', 'deskripsi', 'indikasi_umum', 'komposisi']
valid_cols = [c for c in cols_to_combine if c in df.columns]

df['combined_text'] = (
    df[valid_cols]
    .fillna('')
    .agg(' '.join, axis=1)
    .str.lower()
)
print(f"✓ Combined {len(valid_cols)} columns")

# 3. Create TF-IDF vectorizer
print("\n3. Creating TF-IDF vectorizer...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2
)

# 4. Fit and transform
print("\n4. Fitting TF-IDF (this may take a minute)...")
tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_text'])
print(f"✓ TF-IDF matrix shape: {tfidf_matrix.shape}")

# 5. Save vectorizer
print("\n5. Saving TF-IDF vectorizer...")
vectorizer_path = "tfidf_vectorizer.joblib"
joblib.dump(tfidf_vectorizer, vectorizer_path)
print(f"✓ Saved vectorizer to: {vectorizer_path}")

# 6. Save matrix
print("\n6. Saving TF-IDF matrix...")
matrix_path = "tfidf_matrix.npz"
save_npz(matrix_path, tfidf_matrix)
print(f"✓ Saved matrix to: {matrix_path}")

print("\n" + "=" * 70)
print("✅ TF-IDF GENERATION COMPLETE!")
print("=" * 70)
print(f"\nFiles saved:")
print(f"  - {os.path.abspath(vectorizer_path)} ({os.path.getsize(vectorizer_path) / 1024:.2f} KB)")
print(f"  - {os.path.abspath(matrix_path)} ({os.path.getsize(matrix_path) / (1024*1024):.2f} MB)")
print("\nNow restart your MedicIR server to use pre-computed TF-IDF.")
print("This will make server startup MUCH FASTER!")
print("=" * 70)