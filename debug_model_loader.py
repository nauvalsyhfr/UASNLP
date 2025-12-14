import sys
sys.path.insert(0, '/d:/Project')

from app.model_loader import search_tfidf_docs

# Test direct call
query = "obat diare"
print("Testing search_tfidf_docs...")
print()

# Test with top_k=None
result_all = search_tfidf_docs(query, top_k=None)
print(f"Results with top_k=None: {len(result_all)}")
print(f"First 3 results:")
for i, r in enumerate(result_all[:3]):
    print(f"  {i+1}. {r['nama']}")

# Test with top_k=10
result_10 = search_tfidf_docs(query, top_k=10)
print(f"\nResults with top_k=10: {len(result_10)}")
