import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "app"))

# pastikan root project terbaca
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.model_loader import (
    _bm25_ranked,
    _dense_ranked,
    _tfidf_ranked,
    hybrid_rrf_search,
    available_methods,
    _make_doc_object,
)

print("=== MedicIR DEBUG SEARCH TEST ===\n")

query = "obat diare"
print(f"Query: {query}\n")

# ------------------------------------------------------
# 1. Test BM25
# ------------------------------------------------------
print(">>> Testing BM25...")
bm25_results = _bm25_ranked(query, top_k=10)

if len(bm25_results) == 0:
    print("BM25 NOT available or no results.\n")
else:
    print(f"BM25 returned {len(bm25_results)} results.")
    for idx, score in bm25_results[:3]:
        doc = _make_doc_object(idx, score)
        print(f"- {doc['nama']} (score={score:.4f})")
    print()

# ------------------------------------------------------
# 2. Test Dense (SBERT)
# ------------------------------------------------------
print(">>> Testing Semantic / Dense Search...")
dense_results = _dense_ranked(query, top_k=10)

if len(dense_results) == 0:
    print("Semantic SBERT NOT available.\n")
else:
    print(f"Dense search returned {len(dense_results)} results.")
    for idx, score in dense_results[:3]:
        doc = _make_doc_object(idx, score)
        print(f"- {doc['nama']} (score={score:.4f})")
    print()

# ------------------------------------------------------
# 3. Test TF-IDF (Optional)
# ------------------------------------------------------
print(">>> Testing TF-IDF...")
tfidf_results = _tfidf_ranked(query, top_k=10)

if len(tfidf_results) == 0:
    print("TF-IDF NOT available (this is OK).\n")
else:
    print(f"TF-IDF returned {len(tfidf_results)} results.")
    for idx, score in tfidf_results[:3]:
        doc = _make_doc_object(idx, score)
        print(f"- {doc['nama']} (score={score:.4f})")
    print()

# ------------------------------------------------------
# 4. Test Hybrid RRF
# ------------------------------------------------------
print(">>> Testing Hybrid RRF...")
rrf_results = hybrid_rrf_search(query, top_k=10)

if len(rrf_results) == 0:
    print("Hybrid RRF NOT available.\n")
else:
    print(f"Hybrid RRF returned {len(rrf_results)} results.")
    for doc in rrf_results[:3]:
        print(f"- {doc['nama']} (rrf_score={doc['score']:.4f})")
    print()

print("=== DEBUG FINISHED ===")
