#!/usr/bin/env python
"""
Direct test of search functions to diagnose why search returns 0 results
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("DIRECT SEARCH FUNCTION TEST")
print("=" * 70)

# Import search functions
from app.model_loader_hybrid import (
    smart_search,
    search_minilm,
    search_tfidf,
    df_docs,
    sbert_model,
    doc_embeddings,
    tfidf_vectorizer,
    tfidf_matrix
)

print("\n1. Checking loaded resources...")
print(f"   df_docs: {type(df_docs)} - {len(df_docs) if df_docs is not None else 'None'} rows")
print(f"   sbert_model: {type(sbert_model)} - {'Loaded' if sbert_model is not None else 'None'}")
print(f"   doc_embeddings: {doc_embeddings.shape if doc_embeddings is not None else 'None'}")
print(f"   tfidf_vectorizer: {'Loaded' if tfidf_vectorizer is not None else 'None'}")
print(f"   tfidf_matrix: {tfidf_matrix.shape if tfidf_matrix is not None else 'None'}")

# Test 1: TF-IDF Search
print("\n" + "=" * 70)
print("TEST 1: TF-IDF Search for 'paracetamol'")
print("=" * 70)

try:
    results_tfidf = search_tfidf("paracetamol", top_k=5)
    print(f"✓ TF-IDF returned {len(results_tfidf)} results")
    
    if results_tfidf:
        print("\nTop 3 TF-IDF results:")
        for i, r in enumerate(results_tfidf[:3]):
            print(f"\n{i+1}. {r.get('nama', 'N/A')}")
            print(f"   Score: {r.get('score', 0):.4f}")
            print(f"   Tipe: {r.get('tipe', 'N/A')}")
    else:
        print("✗ No results returned from TF-IDF!")
        
except Exception as e:
    print(f"✗ TF-IDF search failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: MiniLM Search
print("\n" + "=" * 70)
print("TEST 2: MiniLM Search for 'paracetamol'")
print("=" * 70)

try:
    results_minilm = search_minilm("paracetamol", top_k=5)
    print(f"✓ MiniLM returned {len(results_minilm)} results")
    
    if results_minilm:
        print("\nTop 3 MiniLM results:")
        for i, r in enumerate(results_minilm[:3]):
            print(f"\n{i+1}. {r.get('nama', 'N/A')}")
            print(f"   Score: {r.get('score', 0):.4f}")
            print(f"   Tipe: {r.get('tipe', 'N/A')}")
    else:
        print("✗ No results returned from MiniLM!")
        
except Exception as e:
    print(f"✗ MiniLM search failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Hybrid Search
print("\n" + "=" * 70)
print("TEST 3: Hybrid (smart_search) for 'paracetamol'")
print("=" * 70)

try:
    results_hybrid, method_used = smart_search("paracetamol", top_k=5, method="hybrid")
    print(f"✓ Hybrid search used method: {method_used}")
    print(f"✓ Hybrid returned {len(results_hybrid)} results")
    
    if results_hybrid:
        print("\nTop 3 Hybrid results:")
        for i, r in enumerate(results_hybrid[:3]):
            print(f"\n{i+1}. {r.get('nama', 'N/A')}")
            print(f"   Score: {r.get('score', 0):.4f}")
            print(f"   Method: {r.get('method', 'N/A')}")
            print(f"   Tipe: {r.get('tipe', 'N/A')}")
    else:
        print("✗ No results returned from Hybrid!")
        
except Exception as e:
    print(f"✗ Hybrid search failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Validation
print("\n" + "=" * 70)
print("TEST 4: Validation Check")
print("=" * 70)

try:
    from app.model_loader_hybrid import (
        validate_search_results,
        is_medical_query,
        check_results_relevance
    )
    
    # Check if query is medical
    is_med = is_medical_query("paracetamol")
    print(f"is_medical_query('paracetamol'): {is_med}")
    
    # Check results relevance (if we have results)
    if results_hybrid:
        has_relevant = check_results_relevance(results_hybrid, min_score=0.15)
        print(f"check_results_relevance(results): {has_relevant}")
        
        # Full validation
        validation = validate_search_results("paracetamol", results_hybrid)
        print(f"\nValidation result:")
        print(f"  is_valid: {validation.get('is_valid')}")
        print(f"  reason: {validation.get('reason')}")
        print(f"  message: {validation.get('message')}")
        print(f"  filtered_results count: {len(validation.get('filtered_results', []))}")
        
        if validation.get('filtered_results'):
            print(f"\n  First filtered result:")
            r = validation['filtered_results'][0]
            print(f"    {r.get('nama')} - Score: {r.get('score', 0):.4f}")
    
except Exception as e:
    print(f"✗ Validation test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Direct dataset search
print("\n" + "=" * 70)
print("TEST 5: Direct Dataset Search")
print("=" * 70)

if df_docs is not None:
    matches = df_docs[df_docs['nama'].str.contains('paracetamol', case=False, na=False)]
    print(f"Direct search in df_docs['nama']: {len(matches)} matches")
    
    if len(matches) > 0:
        print("\nFirst 5 matches:")
        for idx, row in matches.head(5).iterrows():
            print(f"  - {row['nama']} ({row.get('tipe', 'N/A')})")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)