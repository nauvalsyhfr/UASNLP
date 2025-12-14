# app/model_loader_hybrid.py
"""
MedicIR Model Loader - MiniLM + TF-IDF Hybrid Version
Optimized for Indonesian language medical search with MiniLM
"""

import sys
import os
import joblib
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# PATHS CONFIGURATION
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Data paths
DATA_PATH = os.path.join(PROJECT_ROOT, "final_clean_data_20112024_halodoc_based.csv")

# Model paths - UPDATED TO MINILM
MODEL_DIR = os.path.join(PROJECT_ROOT, "model", "semantic_model_export")
MINILM_MODEL_PATH = os.path.join(MODEL_DIR, "minilm_model")  # Sesuaikan dengan struktur folder Anda

# Pre-computed embeddings and matrices
EMBEDDINGS_PATH = os.path.join(PROJECT_ROOT, "doc_embeddings_minilm.npy")
TFIDF_MATRIX_PATH = os.path.join(PROJECT_ROOT, "tfidf_matrix.npz")
TFIDF_VECTORIZER_PATH = os.path.join(PROJECT_ROOT, "tfidf_vectorizer.joblib")
ID_TO_IDX_PATH = os.path.join(PROJECT_ROOT, "id_to_idx.json")

# ============================================================================
# GLOBAL VARIABLES
# ============================================================================

df_docs = None
sbert_model = None
doc_embeddings = None
tfidf_vectorizer = None
tfidf_matrix = None
id_to_idx = None

# ============================================================================
# LOAD MINILM MODEL
# ============================================================================

def load_minilm_model():
    """Load MiniLM model for semantic search"""
    global sbert_model, doc_embeddings
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Try loading from local path first
        if os.path.exists(MINILM_MODEL_PATH):
            logger.info(f"Loading MiniLM model from: {MINILM_MODEL_PATH}")
            sbert_model = SentenceTransformer(MINILM_MODEL_PATH)
            logger.info("✓ MiniLM model loaded from local path")
        else:
            # Fallback: download from HuggingFace
            logger.info("Local model not found, downloading MiniLM from HuggingFace...")
            sbert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            logger.info("✓ MiniLM model downloaded and loaded")
        
        # Load pre-computed embeddings if available
        if os.path.exists(EMBEDDINGS_PATH):
            doc_embeddings = np.load(EMBEDDINGS_PATH)
            logger.info(f"✓ Loaded pre-computed embeddings: {doc_embeddings.shape}")
        else:
            logger.warning(f"Pre-computed embeddings not found at: {EMBEDDINGS_PATH}")
            logger.warning("Will compute embeddings on-the-fly (slower)")
            doc_embeddings = None
        
        return sbert_model
        
    except ImportError:
        logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
        return None
    except Exception as e:
        logger.error(f"Error loading MiniLM: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# RESOURCE LOADING
# ============================================================================

def load_resources():
    """Load all data and models on startup"""
    global df_docs, tfidf_vectorizer, tfidf_matrix, id_to_idx
    
    logger.info("=" * 70)
    logger.info("MedicIR - MiniLM + TF-IDF Hybrid System")
    logger.info("=" * 70)
    
    # 1. Load CSV Data
    try:
        if os.path.exists(DATA_PATH):
            df_docs = pd.read_csv(DATA_PATH)
            
            # Create combined text
            cols_to_combine = ['nama', 'deskripsi', 'indikasi_umum', 'komposisi']
            valid_cols = [c for c in cols_to_combine if c in df_docs.columns]
            
            df_docs['combined_text'] = (
                df_docs[valid_cols]
                .fillna('')
                .agg(' '.join, axis=1)
                .str.lower()
            )
            
            logger.info(f"✓ Loaded Dataset: {len(df_docs)} documents")
        else:
            logger.error(f"✗ Data file not found: {DATA_PATH}")
            df_docs = pd.DataFrame()
    except Exception as e:
        logger.error(f"✗ Error loading CSV: {e}")
        df_docs = pd.DataFrame()
    
    # 2. Load MiniLM Model
    try:
        model_loaded = load_minilm_model()
        if model_loaded:
            logger.info("✓ MiniLM ready for semantic search")
        else:
            logger.warning("✗ MiniLM not loaded")
    except Exception as e:
        logger.error(f"✗ Error loading MiniLM: {e}")
    
    # 3. Load TF-IDF (pre-computed or create new)
    try:
        if os.path.exists(TFIDF_VECTORIZER_PATH) and os.path.exists(TFIDF_MATRIX_PATH):
            # Load pre-computed TF-IDF
            tfidf_vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
            from scipy.sparse import load_npz
            tfidf_matrix = load_npz(TFIDF_MATRIX_PATH)
            logger.info(f"✓ Loaded pre-computed TF-IDF: {tfidf_matrix.shape}")
        else:
            # Create new TF-IDF
            if df_docs is not None and not df_docs.empty:
                logger.info("Creating new TF-IDF vectorizer...")
                tfidf_vectorizer = TfidfVectorizer(
                    max_features=5000,
                    ngram_range=(1, 2),
                    min_df=2
                )
                tfidf_matrix = tfidf_vectorizer.fit_transform(df_docs['combined_text'])
                logger.info(f"✓ Created TF-IDF: {tfidf_matrix.shape}")
    except Exception as e:
        logger.error(f"✗ Error with TF-IDF: {e}")
    
    # 4. Load ID mapping (optional)
    try:
        if os.path.exists(ID_TO_IDX_PATH):
            import json
            with open(ID_TO_IDX_PATH, 'r') as f:
                id_to_idx = json.load(f)
            logger.info(f"✓ Loaded ID mapping: {len(id_to_idx)} entries")
    except Exception as e:
        logger.warning(f"ID mapping not loaded: {e}")
    
    logger.info("=" * 70)
    logger.info("System Ready: MiniLM + TF-IDF Hybrid")
    logger.info("=" * 70)

# ============================================================================
# SEARCH IMPLEMENTATIONS
# ============================================================================

def _format_results(indices, scores, method_name: str) -> List[Dict]:
    """Format search results with sanitization"""
    results = []
    if df_docs is None or df_docs.empty:
        return results
    
    for idx, score in zip(indices, scores):
        if idx < len(df_docs):
            record = df_docs.iloc[idx].to_dict()
            
            # Clean NaN values
            clean_record = {
                k: (v if pd.notna(v) else "") 
                for k, v in record.items()
            }
            
            # Add metadata
            clean_record['score'] = float(score) if not np.isnan(score) else 0.0
            clean_record['method'] = method_name
            clean_record['doc_id'] = int(idx)
            
            results.append(clean_record)
    
    return results


def search_minilm(query: str, top_k: int = 10) -> List[Dict]:
    """
    Semantic search using MiniLM
    Fast and efficient for medical queries
    """
    if sbert_model is None:
        logger.warning("MiniLM not available, using TF-IDF fallback")
        return search_tfidf(query, top_k)
    
    try:
        # Encode query using MiniLM
        query_embedding = sbert_model.encode([query], convert_to_numpy=True)
        
        # If we have pre-computed embeddings, use them
        if doc_embeddings is not None:
            cosine_scores = cosine_similarity(query_embedding, doc_embeddings).flatten()
        else:
            # Compute embeddings on-the-fly (slower)
            logger.warning("Computing document embeddings on-the-fly...")
            if df_docs is None or df_docs.empty:
                return []
            doc_texts = df_docs['combined_text'].tolist()
            doc_embs = sbert_model.encode(doc_texts, convert_to_numpy=True, show_progress_bar=True)
            cosine_scores = cosine_similarity(query_embedding, doc_embs).flatten()
        
        # Get top-k
        top_indices = cosine_scores.argsort()[::-1][:top_k]
        top_scores = cosine_scores[top_indices]
        
        return _format_results(top_indices, top_scores, 'minilm')
    
    except Exception as e:
        logger.error(f"MiniLM search error: {e}")
        import traceback
        traceback.print_exc()
        return search_tfidf(query, top_k)


def search_tfidf(query: str, top_k: int = 10) -> List[Dict]:
    """TF-IDF based search"""
    if tfidf_vectorizer is None or tfidf_matrix is None:
        logger.warning("TF-IDF not available")
        return []
    
    try:
        query_vec = tfidf_vectorizer.transform([query.lower()])
        cosine_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
        
        # Get top-k indices
        top_indices = cosine_scores.argsort()[::-1][:top_k]
        top_scores = cosine_scores[top_indices]
        
        # Filter zero scores
        valid_mask = top_scores > 0
        return _format_results(
            top_indices[valid_mask], 
            top_scores[valid_mask], 
            'tfidf'
        )
    
    except Exception as e:
        logger.error(f"TF-IDF search error: {e}")
        return []


def hybrid_minilm_tfidf(query: str, top_k: int = 10, 
                        minilm_weight: float = 0.7, 
                        tfidf_weight: float = 0.3) -> List[Dict]:
    """
    Hybrid search combining MiniLM + TF-IDF using RRF
    This is the BEST method for Indonesian medical search
    
    Args:
        query: Search query
        top_k: Number of results to return
        minilm_weight: Weight for MiniLM (default 0.7 - semantic understanding)
        tfidf_weight: Weight for TF-IDF (default 0.3 - keyword matching)
    """
    # Get results from both methods
    search_k = top_k * 3  # Get more candidates for better fusion
    
    minilm_results = search_minilm(query, search_k)
    tfidf_results = search_tfidf(query, search_k)
    
    # RRF (Reciprocal Rank Fusion) with weights
    k = 60  # RRF constant
    rrf_scores = {}
    doc_data = {}
    
    # Process MiniLM results (semantic - higher weight)
    for rank, doc in enumerate(minilm_results):
        doc_id = doc.get('doc_id') or doc.get('nama')
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = 0
            doc_data[doc_id] = doc
        
        # MiniLM weighted score
        rrf_scores[doc_id] += minilm_weight / (k + rank + 1)
    
    # Process TF-IDF results (keyword - lower weight)
    for rank, doc in enumerate(tfidf_results):
        doc_id = doc.get('doc_id') or doc.get('nama')
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = 0
            doc_data[doc_id] = doc
        
        # TF-IDF weighted score
        rrf_scores[doc_id] += tfidf_weight / (k + rank + 1)
    
    # Sort by RRF score
    sorted_docs = sorted(
        rrf_scores.items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:top_k]
    
    # Format final results
    final_results = []
    for doc_id, rrf_score in sorted_docs:
        doc = doc_data[doc_id].copy()
        doc['score'] = rrf_score
        doc['method'] = 'hybrid_minilm_tfidf'
        final_results.append(doc)
    
    return final_results


# ============================================================================
# MAIN SEARCH FUNCTION
# ============================================================================

def smart_search(query: str, top_k: int = 20, method: str = None) -> Tuple[List[Dict], str]:
    """
    Smart search router
    Default: Hybrid MiniLM + TF-IDF (BEST for medical search)
    
    Methods:
        - 'hybrid' or 'auto': Hybrid MiniLM + TF-IDF (RECOMMENDED)
        - 'minilm' or 'semantic': MiniLM semantic search
        - 'tfidf': TF-IDF keyword search
    """
    if method is None or method == 'auto' or method == 'hybrid':
        return hybrid_minilm_tfidf(query, top_k), "hybrid_minilm_tfidf"
    elif method == 'minilm' or method == 'semantic':
        return search_minilm(query, top_k), "minilm"
    elif method == 'tfidf':
        return search_tfidf(query, top_k), "tfidf"
    else:
        # Default to hybrid
        logger.warning(f"Unknown method '{method}', using hybrid")
        return hybrid_minilm_tfidf(query, top_k), "hybrid_minilm_tfidf"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_available_methods() -> Dict[str, bool]:
    """Get status of all available search methods"""
    return {
        'minilm': sbert_model is not None,
        'tfidf': tfidf_vectorizer is not None,
        'hybrid': (sbert_model is not None and tfidf_vectorizer is not None),
        'current_model': 'all-MiniLM-L6-v2',
    }


def detect_query_type(query: str) -> str:
    """Detect if query is single-hop or multi-hop"""
    multi_hop_indicators = [
        ' dan ', ' atau ', ' dengan ', ' untuk ',
        ' yang ', ' serta ', ' juga '
    ]
    
    query_lower = query.lower()
    
    if any(indicator in query_lower for indicator in multi_hop_indicators):
        return "multi-hop"
    
    if len(query.split()) > 5:
        return "multi-hop"
    
    return "single-hop"


def calculate_adaptive_topk(query: str, per_page: int, page: int, max_topk: int = 1000) -> int:
    """Calculate adaptive top-k based on query complexity"""
    base_k = (page + 1) * per_page
    query_type = detect_query_type(query)
    multiplier = 2 if query_type == "multi-hop" else 3
    adaptive_k = base_k * multiplier
    return min(adaptive_k, max_topk)


def sanitize_value(value, key=None):
    """Sanitize single value"""
    if value is None:
        return None if key != 'score' else 0.0
    
    if isinstance(value, (np.floating, float)):
        if np.isnan(value) or np.isinf(value):
            return 0.0
        return float(value)
    
    if isinstance(value, (np.integer, int)):
        return int(value)
    
    if isinstance(value, str):
        return ' '.join(value.split())
    
    if isinstance(value, list):
        return [sanitize_value(v, key) for v in value]
    
    if isinstance(value, dict):
        return {k: sanitize_value(v, k) for k, v in value.items()}
    
    return value


def sanitize_search_results(results: List[Dict]) -> List[Dict]:
    """Sanitize all search results"""
    return [
        {k: sanitize_value(v, k) for k, v in doc.items()}
        for doc in results
    ]


# ============================================================================
# QUERY VALIDATION & RELEVANCE CHECKING
# ============================================================================

def is_medical_query(query: str) -> bool:
    """
    Check if query is related to medical/pharmaceutical topics
    Returns True if query contains medical keywords
    """
    query_lower = query.lower()
    
    # Medical keywords (Indonesian)
    medical_keywords = [
        # Generic medical terms
        'obat', 'medicine', 'medication', 'drug', 'vitamin', 'suplemen',
        'tablet', 'kapsul', 'sirup', 'salep', 'cream', 'drops', 'injeksi',
        
        # Symptoms & conditions
        'sakit', 'nyeri', 'demam', 'batuk', 'pilek', 'flu', 'diare',
        'mual', 'pusing', 'asma', 'diabetes', 'hipertensi', 'kolesterol',
        'alergi', 'infeksi', 'radang', 'luka', 'gatal', 'jerawat',
        
        # Body parts (medical context)
        'kepala', 'perut', 'lambung', 'mata', 'telinga', 'hidung',
        'tenggorokan', 'kulit', 'jantung', 'paru',
        
        # Medical actions
        'mengobati', 'menyembuhkan', 'mengurangi', 'mencegah', 'terapi',
        'pengobatan', 'perawatan', 'dosis', 'resep',
        
        # Common drug types
        'antibiotik', 'analgesik', 'antihistamin', 'antasida', 'parasetamol',
        'ibuprofen', 'aspirin', 'amoxicillin', 'paracetamol'
    ]
    
    # Check if any medical keyword exists in query
    for keyword in medical_keywords:
        if keyword in query_lower:
            return True
    
    return False


def check_results_relevance(results: List[Dict], min_score: float = 0.01) -> bool:  # 0.3 → 0.15
    """
    Check if search results are relevant enough
    Returns False if all scores are below threshold
    """
    if not results:
        return False
    
    # Check if at least one result has score >= min_score
    relevant_results = [r for r in results if r.get('score', 0) >= min_score]
    
    return len(relevant_results) > 0


def validate_search_results(query: str, results: List[Dict]) -> Dict[str, any]:
    """
    Validate search results and provide recommendation
    FIXED: More lenient for medical queries
    """
    # Check 1: Is query medically relevant?
    is_medical = is_medical_query(query)
    
    # Check 2: Are results relevant?
    has_relevant_results = check_results_relevance(results, min_score=0.15)
    
    # PRIORITY: If query is medical, be MORE LENIENT
    if is_medical:
        # If medical query and we have ANY results, show them!
        if results and len(results) > 0:
            # Filter only extremely low scores (< 0.10)
            filtered = [r for r in results if r.get('score', 0) >= 0.005]
            
            if filtered:
                return {
                    'is_valid': True,
                    'reason': 'valid',
                    'message': 'Hasil ditemukan',
                    'filtered_results': filtered
                }
        
        # Medical query but NO results at all
        return {
            'is_valid': False,
            'reason': 'no_relevant_results',
            'message': 'Obat tidak ditemukan dalam database',
            'filtered_results': []
        }
    
    # Non-medical query
    if not has_relevant_results:
        return {
            'is_valid': False,
            'reason': 'non_medical_query',
            'message': 'Query tidak berhubungan dengan obat atau kesehatan',
            'filtered_results': []
        }
    
    # Non-medical but has high-confidence results
    filtered = [r for r in results if r.get('score', 0) >= 0.30]
    
    if not filtered:
        return {
            'is_valid': False,
            'reason': 'low_confidence',
            'message': 'Obat tidak ditemukan',
            'filtered_results': []
        }
    
    return {
        'is_valid': True,
        'reason': 'valid',
        'message': 'Hasil ditemukan',
        'filtered_results': filtered
    }


# ============================================================================
# INITIALIZATION
# ============================================================================

# Load resources when module is imported
load_resources()

# Log available methods
logger.info("\n✓ MiniLM + TF-IDF Hybrid System Ready")
logger.info("✓ Optimized for Medical Information Retrieval")
for method, available in get_available_methods().items():
    status = "✓" if available else "✗"
    logger.info(f"  {status} {method}")