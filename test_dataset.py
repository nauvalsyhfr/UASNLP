import pandas as pd
import os

print("Checking dataset...")
print("=" * 70)

# Check file exists
csv_path = "final_clean_data_20112024_halodoc_based.csv"
print(f"CSV exists: {os.path.exists(csv_path)}")
print(f"CSV size: {os.path.getsize(csv_path) if os.path.exists(csv_path) else 'N/A'} bytes")

# Load dataset
try:
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded: {len(df)} documents")
    print(f"✓ Columns: {list(df.columns)}")
    
    # Create combined text
    cols_to_combine = ['nama', 'deskripsi', 'indikasi_umum', 'komposisi']
    valid_cols = [c for c in cols_to_combine if c in df.columns]
    print(f"✓ Valid columns for search: {valid_cols}")
    
    df['combined_text'] = (
        df[valid_cols]
        .fillna('')
        .agg(' '.join, axis=1)
        .str.lower()
    )
    
    print(f"\nSample combined text:")
    print(df['combined_text'].iloc[0])
    
    # Check for "paracetamol"
    matches = df[df['combined_text'].str.contains('paracetamol', case=False, na=False)]
    print(f"\n✓ Documents containing 'paracetamol': {len(matches)}")
    if len(matches) > 0:
        print(f"  Example: {matches.iloc[0]['nama']}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()