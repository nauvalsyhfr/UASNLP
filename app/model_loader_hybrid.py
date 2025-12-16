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


def hybrid_minilm_tfidf(query: str, top_k: int = 20, minilm_weight: float = 0.7, tfidf_weight: float = 0.3) -> List[Dict]:
    """
    FIXED: Hybrid search combining MiniLM semantic + TF-IDF keyword
    Using IMPROVED scoring that maintains score range 0.3-0.9
    """
    # Get results from both methods
    search_k = min(top_k * 3, 100)
    
    minilm_results = search_minilm(query, search_k)
    tfidf_results = search_tfidf(query, search_k)
    
    # Create score dictionaries
    minilm_scores = {doc.get('doc_id', doc.get('nama')): doc.get('score', 0) for doc in minilm_results}
    tfidf_scores = {doc.get('doc_id', doc.get('nama')): doc.get('score', 0) for doc in tfidf_results}
    
    # Collect all unique documents
    all_doc_ids = set(minilm_scores.keys()) | set(tfidf_scores.keys())
    doc_data = {}
    
    for doc in minilm_results + tfidf_results:
        doc_id = doc.get('doc_id', doc.get('nama'))
        if doc_id not in doc_data:
            doc_data[doc_id] = doc
    
    # Calculate HYBRID scores with PROPER WEIGHTING
    hybrid_scores = {}
    for doc_id in all_doc_ids:
        minilm_score = minilm_scores.get(doc_id, 0)
        tfidf_score = tfidf_scores.get(doc_id, 0)
        
        # FIXED: Weighted average (maintains 0.3-0.9 range)
        hybrid_score = (minilm_weight * minilm_score) + (tfidf_weight * tfidf_score)
        
        hybrid_scores[doc_id] = hybrid_score
    
    # Sort by hybrid score
    sorted_docs = sorted(
        hybrid_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]
    
    # Format final results
    final_results = []
    for doc_id, score in sorted_docs:
        if doc_id in doc_data:
            doc = doc_data[doc_id].copy()
            doc['score'] = float(score)
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

MEDICAL_KEYWORDS = {
    'abajos', 'abbotic', 'abdomen', 'abdominal', 'abnormal', 'aborsi', 'abortus', 'abrasi', 'abrus', 'abses',
    'acarbose', 'ace', 'acepress', 'acetate', 'acetin', 'acetonide', 'acetyl', 'acetylcysteine', 'acetylsalicylic', 'aciclovir',
    'acid', 'acidum', 'acifar', 'acitral', 'aclam', 'aclonac', 'acne', 'actapin', 'actavis', 'actifed',
    'activated', 'actos', 'actosmet', 'actrapid', 'acyclovir', 'adalat', 'adanya', 'adapalene', 'adecco', 'adekuat',
    'adem', 'adenoma', 'adesilentikular', 'adnexitis', 'adrenoceptor', 'adretor', 'adult', 'adults', 'aerius', 'aerob',
    'aerofagia', 'aerosol', 'aeruginosa', 'afiat', 'afidol', 'afiflex', 'afitson', 'aflucaps', 'aftosa', 'afucid',
    'agen', 'agent', 'aggravan', 'agregasi', 'agromed', 'aids', 'ailin', 'aini', 'ains', 'air',
    'akibat', 'akilen', 'akilostomiasis', 'akne', 'aknil', 'akrosia', 'aktif', 'akut', 'alami', 'alaxan',
    'albicans', 'albiguraa', 'albiotin', 'albisam', 'albuforce', 'alcet', 'alco', 'aldisa', 'alegi', 'alegysal',
    'alerfed', 'alergi', 'alergia', 'alergik', 'alergika', 'alergine', 'alerhis', 'alermax', 'alernitis', 'alerzin',
    'alfabio', 'alii', 'alkohol', 'allercyl', 'allerin', 'alleron', 'alletrol', 'allium', 'allohex', 'allopurinol',
    'alloris', 'alluric', 'allylestrenol', 'almacon', 'alodan', 'aloe', 'alofar', 'alopesia', 'alopros', 'alovell',
    'alphahydroxy', 'alphaketoisoleucin', 'alphaketoleucine', 'alphaketophenylalanine', 'alphaketovaline', 'alphamol', 'alstonia', 'aluminium', 'aluss', 'alvita',
    'alxil', 'alyxia', 'alzheimer', 'amadiab', 'amaryl', 'amarylm', 'ambeien', 'ambeven', 'ambonia', 'ambroxol',
    'amcor', 'amebiasis', 'amenore', 'aminophylline', 'aminoral', 'amiodarone', 'amlocor', 'amlodipin', 'amlodipine', 'amlogal',
    'ammeltz', 'ammonium', 'amobiotic', 'amomi', 'amomum', 'amosterra', 'amoxicillin', 'amoxsan', 'ampicillin', 'amubiasis',
    'amvar', 'amylmetacresol', 'amylum', 'anacetine', 'anadex', 'anadium', 'anaerob', 'anafen', 'anak', 'anak-anak',
    'anakanak', 'anakonidin', 'analgesik', 'analpim', 'anastan', 'anaton', 'anbacim', 'anbiolid', 'ancefa', 'ancefaf',
    'ancla', 'andonex', 'androgenetika', 'andrographidis', 'andrographis', 'anemia', 'anerocid', 'anestesi', 'aneurisma', 'anfix',
    'anflat', 'angelica', 'angelicae', 'anggur', 'angin', 'angina', 'angintriz', 'angioedema', 'angioten', 'anhydrate',
    'anisi', 'ankilosa', 'ankilosing', 'ankylosing', 'anoral', 'anoreksia', 'anovulasi', 'anpiride', 'ansietas', 'antagonis',
    'antangin', 'antara', 'antasida', 'anterior', 'anti', 'antiangina', 'antibiotik', 'antidia', 'antihemoroid', 'antihipertensi',
    'antimo', 'antioksidan', 'antiplat', 'antitussive', 'antiza', 'antrain', 'antraks', 'anus', 'anuva', 'anvomer',
    'apel', 'apetic', 'apidra', 'apii', 'apium', 'apolar', 'apolipoprotein', 'appleberry', 'apricot', 'aprovel',
    'aptor', 'aqueous', 'arava', 'arcamox', 'arcapec', 'arcerin', 'arcoxia', 'ardium', 'area', 'argesid',
    'aritmia', 'armacort', 'aroma', 'aromaterapi', 'aromatherapy', 'aromatics', 'arsinal', 'arsitam', 'artalgia', 'artepid',
    'arteri', 'arthritis', 'artikular', 'artoflam', 'artrilox', 'artritis', 'artrodar', 'asam', 'asap', 'ascardia',
    'ascariasis', 'ascaris', 'ascorbic', 'asem', 'asi', 'asiatica', 'asimor', 'asites', 'askamex', 'asma',
    'asmatik', 'asmef', 'aspergillosis', 'aspilets', 'aspirasi', 'aspirin', 'astar', 'astharol', 'asthenia', 'asthenof',
    'astherin', 'astika', 'asvex', 'atagip', 'ataroc', 'atas', 'atenolol', 'aterosklerosis', 'atevir', 'atheletes',
    'atmacid', 'atofar', 'atoni', 'atopik', 'atorvastatin', 'atp', 'atrial', 'atritis', 'atrium', 'attapulgite',
    'augentonic', 'aun', 'aurantifolia', 'aurantium', 'aureus', 'autoimun', 'avamys', 'avelox', 'avesco', 'avocel',
    'awal', 'axaprofen', 'axetil', 'ayam', 'aza', 'azelaic', 'azithromycin', 'azmacon', 'azomax', 'azomep',
    'aztrin', 'baby', 'bacbutol', 'bacitracin', 'bactesyn', 'bactoderm', 'bactoprim', 'bacula', 'badak', 'badan',
    'bagian', 'bahan', 'bahu', 'baik', 'baiyao', 'bakar', 'bakteremia', 'bakteri', 'bakterial', 'bakterialis',
    'balance', 'baljitot', 'balm', 'balpirik', 'balsam', 'balsamifera', 'balsem', 'ban', 'bantif', 'baquinor',
    'barbae', 'barbers', 'barley', 'bartolium', 'basiler', 'batsaria', 'batu', 'batuk', 'bawah', 'baycutenn',
    'bayi', 'bbeta', 'bdm', 'bear', 'bebas', 'bebio', 'becantex', 'bedah', 'befixim', 'bejo',
    'bekerja', 'belanda', 'belas', 'benadryl', 'beneuron', 'bengkak', 'benign', 'benjolan', 'benoson', 'benosong',
    'benostan', 'benoxuric', 'bentuk', 'benzoat', 'benzocaine', 'benzoicum', 'benzolac', 'benzolaccl', 'benzoyl', 'beprin',
    'berair', 'beras', 'berat', 'berbagai', 'berdahak', 'berdarah', 'berea', 'berenang', 'berfungsi', 'berhasil',
    'berhubungan', 'beriberi', 'berkaitan', 'berlebih', 'berlebihan', 'berlico', 'berlifed', 'berlubang', 'bermanfaat', 'berodual',
    'berotec', 'berry', 'bersama', 'bersamaan', 'bersin', 'bersin-bersin', 'bersol', 'beruang', 'berulang', 'berusia',
    'besar', 'besi', 'besilate', 'best', 'bestalin', 'besylate', 'betacylic', 'betadine', 'betahistine', 'betaloc',
    'betamethasone', 'betamox', 'betaone', 'betaserc', 'betasin', 'betason', 'bevalex', 'biang', 'biatron', 'bibir',
    'bicara', 'bicorsan', 'bicrolid', 'bidicef', 'bidium', 'biduran', 'bifido', 'bifonazole', 'biji', 'bikmil',
    'bila', 'biloba', 'bilon', 'bimacyl', 'bimaflox', 'bimastan', 'binotal', 'bintamox', 'bintang', 'bio',
    'biocream', 'biocurkem', 'biodasin', 'bioderm', 'biogastron', 'biogesic', 'biolergy', 'biolife', 'biolincom', 'biomex',
    'bioplacenton', 'bioprexum', 'bioprost', 'bioquin', 'biosan', 'bioskin', 'biostrath', 'biothicol', 'bipro', 'bisacodyl',
    'biscor', 'bisolvon', 'bisoprolol', 'bisovell', 'bisul', 'bisulfate', 'blackcurrant', 'blastomikosis', 'blecidex', 'blefaritis',
    'blefarokonjungtivitis', 'blesifen', 'blocand', 'blocker', 'blocking', 'blopress', 'blorec', 'bloved', 'blu', 'bluberry',
    'blue', 'blumea', 'bodrex', 'bodrexin', 'bokashi', 'bola', 'bonviva', 'boost', 'booster', 'boraks',
    'borraginoln', 'borraginols', 'boss', 'bowel', 'braito', 'bralifex', 'bravoderm', 'bravodermn', 'brazine', 'brd',
    'breathy', 'breezhaler', 'bricasma', 'briclot', 'brilinta', 'brm', 'brochifar', 'brocon', 'bromedcyl', 'bromhexine',
    'bromide', 'bromifar', 'bromika', 'brommer', 'bromokriptin', 'bronchitin', 'bronchitis', 'broncho', 'bronchosal', 'bronchosan',
    'bronchovaxom', 'bronkhial', 'bronkhitis', 'bronkhospasme', 'bronkhospastik', 'bronkhus', 'bronkial', 'bronkiektasis', 'bronkiolitis', 'bronkipect',
    'bronkitis', 'bronkodilator', 'bronkokonstriksi', 'bronkopasma', 'bronkopneumonia', 'bronkopulmonal', 'bronkopulmoner', 'bronkospasme', 'bronkris', 'bronkritis',
    'bronkus', 'broxfion', 'brucellosis', 'buah', 'buang', 'bubble', 'bubblegum', 'budenofalk', 'budesonide', 'bufabron',
    'bufacaryl', 'bufacomb', 'bufacort', 'bufaloide', 'bufamoxy', 'bufantacid', 'bufect', 'bukan', 'bulan', 'bulbus',
    'bumi', 'bunga', 'buntu', 'burmanii', 'burmannii', 'bursitis', 'burung', 'buscopan', 'buta', 'butylbromide',
    'cabe', 'cacar', 'cacing', 'caduet', 'caffeine', 'cair', 'cairan', 'cajuput', 'cajuputi', 'caladine',
    'calcareous', 'calcitriol', 'calcitum', 'calcium', 'calcusol', 'caliberi', 'callusol', 'calorex', 'calortusin', 'calsivas',
    'cambuk', 'camellia', 'cameloc', 'campak', 'camphor', 'camphora', 'campuran', 'candefar', 'candepress', 'canderin',
    'candesartan', 'candida', 'candidiasis', 'candipar', 'candistin', 'candotens', 'canedrylskin', 'canespor', 'canesten', 'canicol',
    'cap', 'capcaisin', 'capek', 'capitis', 'cappuccino', 'caprazol', 'capsaicin', 'capsicum', 'capsinat', 'capsule',
    'capsules', 'captopril', 'capucchino', 'capung', 'carbonate', 'card', 'cardamomum', 'cardicap', 'cardio', 'cardipin',
    'cardisan', 'cardismo', 'care', 'carmed', 'carpiaton', 'carpine', 'carvedilol', 'carvilol', 'cassiae', 'cataflam',
    'catarlent', 'cavicur', 'cayla', 'cazetin', 'cedera', 'cedocard', 'cefabiotic', 'cefacef', 'cefadroxil', 'cefarox',
    'cefat', 'cefila', 'cefixime', 'cefspan', 'cefuroxime', 'celebrex', 'celecoxib', 'celery', 'celestik', 'celocid',
    'cendo', 'cendrid', 'cenfresh', 'cengkeh', 'centabio', 'centella', 'cepat', 'ceptik', 'cerafix', 'cerebral',
    'cerini', 'cerna', 'cetapred', 'ceteme', 'cetinal', 'cetirgi', 'cetirizine', 'cetrin', 'cetymin', 'cetzin',
    'cfc', 'chai', 'chang', 'channa', 'cheng', 'cherry', 'chest', 'chi', 'child', 'children',
    'chinensis', 'chlamydia', 'chloramfecort', 'chloramfecorth', 'chloramphenicol', 'chlorexol', 'chloride', 'chlorofit', 'chlorophyll', 'chlorphenamine',
    'cholecalciferol', 'cholespar', 'cholestat', 'cholestor', 'cholvastin', 'chrysantemum', 'cialis', 'ciastar', 'ciflon', 'ciflos',
    'cilexetil', 'cilostazol', 'cimetidine', 'cina', 'cindala', 'cinnamomi', 'cinnamomum', 'cinnarizine', 'cinogenta', 'cinolon',
    'cinolonn', 'ciprec', 'ciprofloxacin', 'citaz', 'citomol', 'citoprim', 'citrate', 'citri', 'citrifolia', 'citronella',
    'citrus', 'clabat', 'clamixin', 'claneksi', 'clarithromycin', 'claritin', 'clast', 'clavamox', 'clavulanate', 'clidacor',
    'clinbercin', 'clindamycin', 'clinidac', 'clinium', 'clinjos', 'clinmas', 'clinovir', 'clobesan', 'clobetasol', 'cloderm',
    'clofion', 'clogin', 'clomifene', 'clonaderm', 'clopidogrel', 'clotaire', 'clotrimazole', 'clovebalsem', 'clovertil', 'coamoxiclav',
    'coaprovel', 'coated', 'cobazim', 'cocidioidomycosis', 'codela', 'codiovan', 'cohistan', 'coirvell', 'coklat', 'colchicine',
    'colcitine', 'cold', 'colergis', 'colfin', 'coli', 'colidium', 'colipred', 'colistine', 'collerin', 'colme',
    'colsancetine', 'colubriformis', 'combantrin', 'combi', 'combicitrine', 'combivent', 'comdipin', 'common', 'compacti', 'complex',
    'compositum', 'comtusi', 'conal', 'concor', 'conmycin', 'constantia', 'constipen', 'constuloz', 'contagiosum', 'contrexyn',
    'conucol', 'cool', 'cooling', 'coolmint', 'copal', 'coparcetin', 'copidrel', 'copper', 'coralan', 'cordarone',
    'cordeson', 'coredryl', 'corhinza', 'corifam', 'corocyd', 'coroflox', 'corporis', 'corsaneuron', 'corsatrocin', 'cortamine',
    'cortex', 'corthon', 'costil', 'cotrim', 'cotrimoxazole', 'cougar', 'cough', 'counterpain', 'coveram', 'covid-19',
    'coxiron', 'cozaar', 'cpg', 'cream', 'crestor', 'crohn', 'cromophtal', 'cronitin', 'cruris', 'cryptal',
    'crystal', 'cuaca', 'cupanol', 'curcuma', 'curcumae', 'custodiol', 'cyanocobalamin', 'cyclogynon', 'cyclon', 'cycloprogynova',
    'cydifar', 'cygest', 'cylowam', 'cyproheptadine', 'cysteine', 'dacin', 'dahak', 'dahuricae', 'dakriosistitis', 'daktarin',
    'dalacin', 'damaben', 'daneuron', 'danoclav', 'dapyrin', 'darah', 'darahnya', 'daratin', 'darsi', 'darvon',
    'daryanttulle', 'daryazinc', 'datan', 'dates', 'daun', 'day', 'daya', 'dazolin', 'dcl', 'dcool',
    'debostin', 'debu', 'decadryl', 'decamox', 'decolgen', 'decolsin', 'decrilip', 'deculin', 'deep', 'defisiensi',
    'defisit', 'deflamat', 'deformans', 'degeneratif', 'degirol', 'degrium', 'dehaf', 'dehidrasi', 'dehista', 'dekubitus',
    'delirium', 'demacolin', 'demam', 'demensia', 'denomix', 'dependent', 'depresi', 'dequalinium', 'dermabiotik', 'dermacoid',
    'dermaral', 'dermasolon', 'dermatitis', 'dermatofit', 'dermatofita', 'dermatografisme', 'dermatologi', 'dermatologik', 'dermatomikosis', 'dermatomiositis',
    'dermatop', 'dermatosis', 'dermovate', 'dermovel', 'desloratadine', 'deslotine', 'desolex', 'desolexn', 'desonide', 'desoximetasone',
    'destavell', 'destloratadine', 'detrusitol', 'devosix', 'dewasa', 'dewi', 'dex', 'dexacap', 'dexaflox', 'dexamethasone',
    'dexanta', 'dexaton', 'dexchlorpheniramine', 'dexclosan', 'dexketoprofen', 'dexosyn', 'dextaco', 'dextamine', 'dexteem', 'dextral',
    'dextran', 'dextromethorphan', 'dextrose', 'dextrosin', 'dexyclav', 'dexycol', 'dexyl', 'dexymox', 'diabe', 'diabemed',
    'diabetes', 'diabetik', 'diabit', 'diachol', 'diadasa', 'diafac', 'diaform', 'diaformin', 'diagit', 'diakibatkan',
    'dialance', 'diamicron', 'diane', 'diapers', 'diapet', 'diare', 'diastrix', 'diatabs', 'diatasi', 'diaversa',
    'dibekacin', 'diberikan', 'dibetik', 'diclofenac', 'dicloflam', 'dicom', 'dicoren', 'diet', 'diethylamine', 'dietin',
    'diflam', 'diflucan', 'difteri', 'digenta', 'digest', 'digigit', 'digoxin', 'digunakan', 'dihasilkan', 'dihydrate',
    'dihydrochloride', 'diindikasikan', 'dikendalikan', 'diketahui', 'diklovit', 'dikombinasikan', 'dikonfirmasi', 'dikontrol', 'dilmen', 'diltiazem',
    'dimana', 'dimenhydrinate', 'dingin', 'dini', 'dinitrate', 'dionicol', 'diosmin', 'diovan', 'dioxide', 'diphenhydramine',
    'diprogenta', 'dipropionate', 'diprosalic', 'diprosone', 'diprosta', 'dipsamol', 'diquas', 'discoid', 'disease', 'disebabkan',
    'disebakan', 'disekitar', 'disentri', 'disertai', 'disflatyl', 'disforik', 'disfungsi', 'disfungsional', 'diskoid', 'diskus',
    'dislipidemia', 'dismenore', 'dismenorea', 'dispense', 'dispepsia', 'dispesia', 'dispnea', 'disudrinped', 'disulfide', 'ditandai',
    'diterapi', 'ditoleransi', 'diuretik', 'divask', 'divoltar', 'dizine', 'dmp', 'dobrizol', 'doen', 'dogesic',
    'dohixat', 'dokter', 'dolo', 'dolofenf', 'dololicobion', 'doloneurobion', 'dolos', 'dom', 'dome', 'domedon',
    'domeran', 'domestica', 'domesticae', 'dometa', 'dominal', 'domperidone', 'doodle', 'dopamet', 'dorbigot', 'dormi',
    'dose', 'dosis', 'dosivec', 'dothrocyn', 'douche', 'down', 'doxicor', 'doxycycline', 'dph', 'dragon',
    'dramamine', 'dried', 'drops', 'drovax', 'droxal', 'droxefa', 'drug', 'dry', 'dryfresh', 'dryl',
    'dulcolactol', 'dulcolax', 'dumin', 'dumocycline', 'duo', 'duodenal', 'duodenitis', 'duodenum', 'duotrav', 'duphalac',
    'duramycin', 'durocort', 'dynalax', 'ear', 'echinacea', 'echinaceae', 'echinatur', 'eclid', 'econazine', 'edema',
    'edorisan', 'edotin', 'efek', 'efektif', 'efisol', 'efisolc', 'eflagen', 'eflin', 'egoji', 'ejakulasi',
    'ekaliptus', 'eksantem', 'eksantema', 'eksaserbasi', 'eksik', 'eksim', 'ekspektoran', 'eksterna', 'eksternal', 'ekstra',
    'ekstrak', 'ekstremitas', 'ektima', 'ekzema', 'elidel', 'eliquis', 'ellison', 'elocon', 'elomox', 'elopro',
    'elox', 'elroid', 'emboli', 'emeran', 'emerten', 'emfisema', 'empagliflozin', 'empedu', 'empiema', 'emte',
    'emturnas', 'emulsion', 'enakur', 'enatin', 'enbatic', 'encok', 'endogen', 'endokarditis', 'endokrin', 'endometriosis',
    'endometritis', 'endoskopi', 'enema', 'enkasari', 'ensefalitis', 'ensefalopati', 'enteric', 'enterobacter', 'enterobiasis', 'enterokolitis',
    'entrostop', 'enystin', 'enzyplex', 'eperisone', 'epexol', 'ephedrine', 'epidermophyton', 'epididimo', 'epididimo-orkitis', 'epididimoorkitis',
    'epigastralgia', 'epigastrium', 'episan', 'episkleritis', 'epo', 'eprinoc', 'epsonal', 'erabutol', 'eradikasi', 'eraphage',
    'erdosteine', 'ereksi', 'ericaf', 'ericfil', 'eritema', 'eritematosus', 'eritrasma', 'eritritis', 'erla', 'erlaflu',
    'erlagin', 'erlamol', 'erlamoxy', 'erlamycetin', 'erlapect', 'erosi', 'erosif', 'erpepsa', 'erphadrox', 'erphaflam',
    'erphaflu', 'erphakaf', 'erphalanz', 'erphamazol', 'erphamol', 'erphasal', 'erphatrim', 'ersoprinosine', 'ersylan', 'ervask',
    'ervy', 'erymed', 'erysanbe', 'erythrin', 'erythromycin', 'esemag', 'esensial', 'esepuluh', 'esinol', 'esofageal',
    'esofagitis', 'esofagus', 'esoferr', 'esomeprazole', 'essensial', 'estesia', 'esthero', 'estin', 'estradiol', 'estrogen',
    'esvat', 'etabiotic', 'etadexta', 'etafenin', 'etaflox', 'etamox', 'etamoxul', 'etawalin', 'ethambutol', 'ethinylestradiol',
    'ethyl', 'etoricoxib', 'etorix', 'etorvel', 'euca', 'eucalyptus', 'eugenol', 'euphyllin', 'eurycoma', 'evalen',
    'evening', 'evothyl', 'exaflam', 'exaserbasi', 'exforge', 'exovon', 'expectorant', 'extra', 'extract', 'extractum',
    'eye', 'eyefresh', 'eyelotion', 'eyes', 'ezelin', 'ezetimibe', 'ezetrol', 'ezol', 'factors', 'faecalis',
    'faktu', 'falciparum', 'falergi', 'falikulitis', 'famili', 'familial', 'famocid', 'famotidine', 'fapivell', 'farbion',
    'farbivent', 'fargetix', 'fargoxin', 'faringitis', 'faringotonsilitis', 'farizol', 'farlev', 'farmabes', 'farmacrol', 'farmadol',
    'farmadral', 'farmakologik', 'farmalat', 'farmasal', 'farmoten', 'farnirex', 'farnormin', 'farpain', 'farsifen', 'farsiretic',
    'farsix', 'farsorbid', 'farsycol', 'fartolin', 'fasgesic', 'fasgo', 'fasidol', 'fasiprim', 'fasolon', 'fast',
    'fastor', 'fatibact', 'fatigue', 'favikal', 'faxiden', 'fds', 'fdt', 'febogrel', 'febrinex', 'feburic',
    'febuxostat', 'feldco', 'femaplex', 'feminax', 'feminine', 'femisic', 'fenamin', 'fenaren', 'fendex', 'fenetrasi',
    'feng', 'fenicol', 'fenofibrate', 'fenoflex', 'fenosup', 'fenris', 'fermol', 'ferrous', 'fertilive', 'fertin',
    'festaric', 'fetik', 'fever', 'fexazol', 'fexofenadine', 'fiber', 'fibesco', 'fibramed', 'fibrilasi', 'fibroid',
    'fibropect', 'figure', 'fimestan', 'fionvask', 'fisik', 'fistula', 'fisura', 'fit', 'fitajoint', 'fitensi',
    'fitocare', 'fitrazinc', 'fixacep', 'fixatic', 'fixiphar', 'flagyl', 'flagystatin', 'flam', 'flamar', 'flamergi',
    'flamigra', 'flasicox', 'flavour', 'flaxseed', 'flebitis', 'fleet', 'flek', 'flexamine', 'fleximuv', 'flexpen',
    'flixonase', 'floksid', 'flos', 'floxa', 'floxacap', 'floxifar', 'floxigra', 'flu', 'flucadex', 'fluconazole',
    'flucoral', 'fludrocortisone', 'fluimucil', 'flumin', 'flunarizine', 'flunax', 'fluocinolone', 'fluocortn', 'fluomizin', 'fluorometholone',
    'flush', 'flutamol', 'flutamolp', 'flutias', 'fluticasone', 'flutopc', 'flutrop', 'flutter', 'fluvir', 'fluxar',
    'fluza', 'fluzep', 'fol', 'folat', 'folavit', 'folic', 'folikulitis', 'folium', 'follicore', 'fonylin',
    'foot', 'forasma', 'forbetes', 'forcanox', 'force', 'forderm', 'fordica', 'fordin', 'forelax', 'forifek',
    'formoterol', 'formula', 'formyco', 'forotic', 'forres', 'forsendi', 'forte', 'forten', 'fortibi', 'fortusin',
    'forxiga', 'fosen', 'fosicol', 'four', 'fraktur', 'free', 'frego', 'frekuensi', 'freshcare', 'freshliving',
    'freshter', 'frigout', 'friladar', 'frost', 'fructus', 'fruit', 'fruity', 'fucilex', 'fuco', 'fukricin',
    'fuladic', 'fumarate', 'fung', 'fungares', 'fungasolss', 'fungi', 'fungiderm', 'fungistop', 'fungitrazol', 'fungoral',
    'fungsi', 'fungsional', 'funtas', 'furoate', 'furosemide', 'furunculosis', 'furunkulosis', 'fusaltrax', 'fusidasol', 'fusidic',
    'fusipar', 'fuson', 'fuzide', 'gaforin', 'gagal', 'gajah', 'galdom', 'galflux', 'galpain', 'galtaren',
    'galvus', 'galvusmet', 'gamat', 'gandapura', 'gangguan', 'gangrene', 'garabiotic', 'garam', 'garcia', 'garcinia',
    'garexin', 'gargle', 'garlic', 'garlicia', 'garlite', 'gasela', 'gaster', 'gastin', 'gastran', 'gastric',
    'gastridin', 'gastrinal', 'gastritis', 'gastro', 'gastroduodenitis', 'gastroenteritis', 'gastroesophageal', 'gastrointestinal', 'gastroparesis', 'gastrucid',
    'gatal', 'gatal-gatal', 'gatal2', 'gavistal', 'gejala', 'gejala-gejala', 'gejala2', 'gel', 'gelang', 'geliga',
    'gemfibrozil', 'gemuk', 'genalsik', 'genalten', 'gencef', 'gendang', 'genicol', 'genital', 'genito-urinaria', 'genitourinari',
    'genoclom', 'genoint', 'genolon', 'gensia', 'genta', 'gentacid', 'gentalex', 'gentamicin', 'gentasolon', 'gentasonn',
    'geranium', 'gerd', 'gerdilium', 'gestamag', 'ghenshen', 'giardiasis', 'giflox', 'gigi', 'gigitan', 'ginekologi',
    'gingivitis', 'gingkan', 'ginifar', 'ginjal', 'ginkgo', 'ginseng', 'girabloc', 'gitas', 'glabra', 'glamarol',
    'glaopen', 'glaoplus', 'glargine', 'glaukoma', 'glauseta', 'gliabetes', 'gliaride', 'glibenclamide', 'gliclazide', 'glicolock',
    'glikamel', 'glikemik', 'glikos', 'glikosida', 'glimefion', 'glimepiride', 'glimetic', 'gliserin', 'glopac', 'glosofaringeal',
    'glubose', 'glucobay', 'glucodex', 'gluconate', 'glucophage', 'glucored', 'glucoryl', 'glucotika', 'glucotrol', 'glucovel',
    'gludepatic', 'glufor', 'glukolos', 'glukosa', 'glumin', 'gluvas', 'gluvocel', 'glycerol', 'glycyrrhiza', 'glyxambi',
    'goflex', 'golden', 'gom', 'gondopuro', 'gonokokal', 'gonore', 'gonorrhoea', 'good', 'gosok', 'gosokpijaturut',
    'gosokpijaturutminyak', 'gout', 'govazol', 'gpu', 'gracivask', 'gradiab', 'grafacef', 'grafachlor', 'grafadon', 'grafalin',
    'grafix', 'gragenta', 'grahabion', 'gralixa', 'gram', 'gramax', 'grameta', 'granopi', 'grantusif', 'granul',
    'granula', 'granule', 'granus', 'grape', 'graperide', 'graprima', 'graseric', 'grass', 'gratheos', 'gratizin',
    'gravask', 'gravastin', 'graveolens', 'graveolentis', 'gravidarum', 'gravis', 'graxine', 'green', 'gricin', 'griseofulvin',
    'grivin', 'guaifenesin', 'guanistrep', 'gula', 'gum', 'gurah', 'gusi', 'gynoxa', 'habbatussauda', 'habitulasi',
    'haemocaine', 'haid', 'halfilyn', 'halus', 'hamil', 'han', 'hangat', 'hanpian', 'hansaplast', 'happy',
    'hapsen', 'harnal', 'harvest', 'hati', 'hau', 'hawaiian', 'hcl', 'head', 'health', 'heartburn',
    'hecobac', 'hedera', 'helicobacter', 'helix', 'helixim', 'heltiskin', 'hematoma', 'hemaviton', 'hemihydrate', 'hemobion',
    'hemoroa', 'hemorogard', 'hemoroid', 'hepagusan', 'heparin', 'hepatik', 'hepatitis', 'hepatobiliar', 'hepatobilier', 'heplav',
    'heptahydrate', 'heptasan', 'herba', 'herbacold', 'herbacure', 'herbadrink', 'herbakof', 'herbal', 'herbalax', 'herbamuno',
    'herbana', 'herbapain', 'herbatia', 'herbavomitz', 'herbesser', 'herbivi', 'herclov', 'hercum', 'hernia', 'herniasi',
    'herocyn', 'herpes', 'herpetic', 'hervis', 'hesmin', 'hesroid', 'heterozigot', 'hexadol', 'hexavask', 'hexon',
    'hialid', 'hiatus', 'hicholfen', 'hico', 'hid', 'hidradenitis', 'hidung', 'hijau', 'hikmah', 'himalaya',
    'hingga', 'hiopar', 'hiperaldosteronisme', 'hiperammonemia', 'hiperasiditas', 'hiperemesis', 'hiperglikemia', 'hiperkalemia', 'hiperkeratosis', 'hiperkolesterolemia',
    'hiperlipidemia', 'hipermotilitas', 'hiperparatiroidisme', 'hiperpigmentasi', 'hipersekresi', 'hipersensitif', 'hipertensi', 'hipertrigliserida', 'hipertrigliseridemia', 'hiperurisemia',
    'hipoalbuminemia', 'hipogalaksia', 'hipogalaktia', 'hipokalemia', 'hipomagnesemia', 'hipoparatiroid', 'hipoparatiroidisme', 'hipotensi', 'hipovolemik', 'hirokids',
    'hirsutisme', 'hislorex', 'histapan', 'histidine', 'histigo', 'histoplasmosis', 'histrine', 'hitam', 'holicos', 'holidon',
    'holizinc', 'homoclomin', 'homocystinuria', 'homozigot', 'honey', 'hong', 'hordeolum', 'hormon', 'hot', 'hotin',
    'hsa', 'huang', 'hufabethamin', 'hufadine', 'hufadon', 'hufafural', 'hufagripp', 'hufallerzine', 'hufamag', 'hufanoxil',
    'hufaprofen', 'hufatidine', 'hufralgin', 'humalog', 'hustab', 'hustadin', 'hyaloph', 'hyalub', 'hyaluronate', 'hyclate',
    'hydrobromide', 'hydrochloride', 'hydrochlorothiazide', 'hydrocort', 'hydrocortisone', 'hydrogen', 'hydroquinone', 'hydrotalcite', 'hydroxide', 'hygiene',
    'hyoscine', 'hyperchol', 'hyperil', 'hyperplasia', 'hypertonia', 'hypofil', 'hypromellose', 'hyric', 'hytroz', 'iberet',
    'ibp', 'ibuprofen', 'ichtyol', 'iddm', 'ideal', 'idiopati', 'idiopatik', 'ifarsyl', 'ifen', 'ifitamol',
    'iflacort', 'ika', 'ikacetamol', 'ikaderm', 'ikadryl', 'ikagen', 'ikamicetin', 'ikan', 'iktiosis', 'ileus',
    'imboost', 'imodium', 'imosa', 'imperfecta', 'impetigo', 'impotensi', 'improvox', 'imreg', 'imun', 'imunex',
    'imunisasi', 'imunodefisiensi', 'imunomodulator', 'inadryl', 'inamid', 'inbacef', 'incetyl', 'incidal', 'inclarin', 'inclovir',
    'indalctn', 'indanox', 'indapamide', 'indikasi', 'indobion', 'indomag', 'indra', 'inerson', 'infalgin', 'infark',
    'infatrim', 'infeksi', 'infeld', 'infertilitas', 'inflamasi', 'inflamation', 'inflammation', 'influensa', 'influenza', 'influenzae',
    'ingat', 'inggris', 'inguinale', 'inha', 'inhalasi', 'inhaler', 'inhibitor', 'inhipump', 'inhitril', 'injeksi',
    'inkontinensia', 'inkurin', 'inmatrol', 'innovair', 'inolin', 'inoxin', 'inpepsa', 'insaar', 'insomnia', 'insto',
    'insufflations', 'insufisiensi', 'insulin', 'interbi', 'intercon', 'interdoxin', 'interflox', 'interhistin', 'intermiten', 'intermoxil',
    'internolol', 'interpec', 'interpect', 'interpril', 'interquin', 'intertrigo', 'intervask', 'intervertebra', 'interzinc', 'interzol',
    'intestinal', 'intifen', 'intoleransi', 'intra', 'intra-abdomen', 'intrizin', 'intunal', 'intunalf', 'inversyn', 'invomit',
    'inza', 'inzana', 'iodide', 'ipratropium', 'irama', 'irbesartan', 'irbosyd', 'iremax', 'iridocyclitis', 'iritabilitas',
    'iritasi', 'iritis', 'irritable', 'irtan', 'irvask', 'irvell', 'ischialgia', 'isivas', 'iskemia', 'iskemik',
    'iskhemik', 'isoniazid', 'isoprinosine', 'isoptin', 'isorbid', 'isoric', 'isosorbide', 'isotic', 'isprinol', 'itamol',
    'itch', 'itchy', 'itrabat', 'itraconazole', 'itzol', 'ivy', 'jadied', 'jagat', 'jahe', 'jahem',
    'jambu', 'jamsi', 'jamu', 'jamur', 'jangka', 'jantung', 'janumet', 'januvia', 'japonica', 'jardiance',
    'jari', 'jaringan', 'jasmine', 'jati', 'jenis', 'jerawat', 'jeruk', 'jesscool', 'jinak', 'jirovecii',
    'joint', 'jointfit', 'joyo', 'jrg', 'junior', 'juvenile', 'kadar', 'kadas', 'kadiflam', 'kadiogenik',
    'kaditic', 'kaflam', 'kahiyang', 'kakeksia', 'kaki', 'kaku', 'kal', 'kalbion', 'kalbutol', 'kalcinol',
    'kalitake', 'kalmoxilin', 'kaloba', 'kalpanax', 'kalpanaxk', 'kalpepsa', 'kalquest', 'kalsium', 'kaltrofen', 'kamal',
    'kambuhnya', 'kamolas', 'kandida', 'kandidemia', 'kandidiasis', 'kandidosis', 'kandistatin', 'kandung', 'kang', 'kanker',
    'kaocitin', 'kaolana', 'kaolin', 'kaotin', 'kapak', 'kapalan', 'kapitis', 'kaplet', 'kapsida', 'kapsul',
    'karbunkel', 'kardiomiopati', 'kardiovaskular', 'karena', 'karpos', 'kary', 'kasa', 'katarak', 'kattuk', 'kayu',
    'keadaan', 'kebiruan', 'kebutuhan', 'kecemasan', 'kecil', 'kecuali', 'kegagalan', 'keganasan', 'kehamilan', 'kehangatan',
    'kehilangan', 'kejadian', 'kejang', 'kejibeling', 'kekebalan', 'kekeringan', 'kekurangan', 'kelahiran', 'kelainan', 'kelamin',
    'kelebihan', 'kelembaban', 'kelenjar', 'kelumpuhan', 'kemasan', 'kematian', 'kembang', 'kembung', 'kemerahan', 'kemih',
    'kemiri', 'kemoterapi', 'kenacort', 'kencing', 'kencur', 'kendaraan', 'kendaron', 'kenis', 'kental', 'kepala',
    'keputihan', 'keracunan', 'keratinisasi', 'keratitis', 'keratokonjungtivitis', 'keratosis', 'kering', 'keringat', 'kerongkongan', 'kerusakan',
    'kesehatan', 'keseimbangan', 'keseleo', 'kesemutan', 'kesulitan', 'ketajaman', 'ketesse', 'ketidaknyamanan', 'ketoconazole', 'ketombe',
    'ketomed', 'ketoprofen', 'ketricin', 'khronik', 'khronis', 'khusus', 'khususnya', 'kid', 'kids', 'kifluzol',
    'kifovir', 'kimoxil', 'kinderen', 'kiri', 'kista', 'kit', 'kita', 'klaudikasio', 'klebsiella', 'klimakterik',
    'klin', 'klinis', 'klinset', 'kliran', 'kloasma', 'kloderma', 'klorfeson', 'klorokuin', 'klotaren', 'kocok',
    'koksidioidomikosis', 'kolagen', 'kolera', 'kolesterol', 'kolik', 'kolkatriol', 'kolopatis', 'kolton', 'kolumna', 'koma',
    'kombiglyze', 'kombinasi', 'kombinasiitraconazole', 'kombinasikombinasi', 'kombinasilansoprazole', 'kombinasiomeprazole', 'komedo', 'komix', 'komplikasi', 'kompolax',
    'komuniti', 'kondisi', 'kondisi2', 'kongestif', 'konicare', 'konidin', 'konigen', 'konilife', 'konjungtivitis', 'konsentrasi',
    'konstipasi', 'konstraksi', 'kontagiosa', 'kontak', 'kontrasepsi', 'kontrol', 'konvermex', 'konvulsi', 'koreng', 'kornea',
    'koroner', 'korporis', 'kortikosteroid', 'koyo', 'kram', 'kremi', 'krim', 'kriptokokal', 'kriptokokosis', 'kriptosporidiosis',
    'kronik', 'kronis', 'kruris', 'kuat', 'kucing', 'kudis', 'kuku', 'kuldon', 'kulit', 'kuman',
    'kumatnya', 'kumis', 'kuning', 'kunir', 'kunyit', 'kuo', 'kupukupu', 'kuramping', 'kurang', 'kurap',
    'kutil', 'kutilos', 'kutu', 'kutus', 'kwikpen', 'labialis', 'labirin', 'labirinitis', 'lacoldin', 'lacoma',
    'lacophen', 'lacosib', 'lactose', 'lactulax', 'lactulose', 'ladenum', 'ladyfem', 'lafalos', 'laflanac', 'lagas',
    'lagesil', 'lain', 'lainnya', 'laktafit', 'lama', 'lambliasis', 'lambucid', 'lambung', 'lametic', 'lamofer',
    'lanakeloid', 'lanakeloide', 'lanamol', 'lanareuma', 'lancid', 'lando', 'lanfix', 'lang', 'lanos', 'lanosan',
    'lanpepsa', 'lanpracid', 'lansoprazole', 'lantus', 'lanvell', 'lapibion', 'lapicef', 'lapifed', 'lapiflox', 'lapigim',
    'lapimox', 'lapimuc', 'lapisan', 'lapisiv', 'lapisivt', 'lapistan', 'lapiva', 'lapraz', 'laprosin', 'laproton',
    'laringitis', 'laringotrakeobronkitis', 'larutan', 'lasal', 'lasegar', 'laserin', 'lasgan', 'lasix', 'lasmalin', 'latanoprost',
    'latibet', 'latipress', 'lauramox', 'lavender', 'laxadilac', 'laxadine', 'laxana', 'laxarec', 'laxatab', 'laxing',
    'laxsanbe', 'laz', 'lcisin', 'leaf', 'lebam', 'lebih', 'lecet', 'leci', 'lefos', 'lega',
    'leher', 'lelah', 'lelap', 'lemak', 'lemon', 'lemongrass', 'lendir', 'leng', 'lensa', 'lentikular',
    'leomoxyl', 'lepra', 'leptospirosis', 'lergio', 'lerzin', 'lesi', 'lester', 'lesvatin', 'levemir', 'levertran',
    'levit', 'levitra', 'levocetirizine', 'levocin', 'levodopa', 'levodropropizine', 'levofloxacin', 'levopront', 'levores', 'levotusin',
    'levovid', 'lexa', 'lexacorton', 'lexacrol', 'lexadium', 'lexadon', 'lexagin', 'lexahist', 'lexapram', 'lexicam',
    'lexigo', 'lexipron', 'lexmodine', 'lfalergi', 'lfx', 'librofed', 'lichen', 'licidal', 'licogenta', 'licosolon',
    'lidah', 'lidocaine', 'lidose', 'lifezar', 'liken', 'likhen', 'lilo', 'limfogranuloma', 'limoxin', 'linchopar',
    'lincocin', 'lincomycin', 'lindacyn', 'ling', 'lingkungan', 'lingzhi', 'lini', 'liniment', 'linogra', 'lintropsin',
    'linu', 'lion', 'lioresal', 'lipanthyl', 'liparin', 'lipepsa', 'lipitor', 'lipivast', 'liposin', 'lipres',
    'liquid', 'liquiritiae', 'lisinopril', 'litoxa', 'litrol', 'liv', 'liver', 'livola', 'livractin', 'lixiana',
    'lizor', 'llax', 'lock', 'lodecon', 'lodia', 'lodipas', 'lodoz', 'logg', 'lokal', 'lokev',
    'longatin', 'longer', 'longifolia', 'lopamid', 'loperamide', 'lopiten', 'loprolol', 'loracor', 'loran', 'loratadine',
    'loremid', 'lorihis', 'lorinid', 'lorson', 'losartan', 'lostacef', 'lotasbat', 'lotharson', 'lotion', 'lotyn',
    'lovask', 'loximei', 'lozenges', 'luar', 'lubago', 'lubricen', 'luka', 'lumbar', 'lumbricoides', 'lumiquin',
    'lunak', 'lung', 'luohanguo', 'lupus', 'lutut', 'lvit', 'lymphogranuloma', 'lymphomagranuloma', 'lysine', 'lyteers',
    'lzinc', 'maag', 'maagfit', 'mabuk', 'maca', 'madeca', 'madia', 'madu', 'magalat', 'maganol',
    'magasida', 'magnesium', 'magtral', 'maharani', 'maintate', 'makan', 'makanan', 'maksilaris', 'maksimal', 'malaria',
    'maleate', 'malnutrisi', 'maltofer', 'mamabear', 'man', 'mancur', 'manfaat', 'mangga', 'manggis', 'mangostana',
    'manifestasi', 'manis', 'manjakani', 'manus', 'martha', 'mas', 'masalah', 'masela', 'mastatin', 'mastin',
    'mastositiosis', 'masuk', 'mat', 'mata', 'matahari', 'matrovir', 'maupun', 'max', 'maxcef', 'maximus',
    'maxlis', 'maxpro', 'maxstan', 'mbm', 'mebhydrolin', 'mebo', 'meccaderma', 'mecox', 'medi', 'media',
    'mediabetea', 'mediamer', 'medicated', 'medication', 'medichlor', 'medicine', 'medicort', 'mediflex', 'medigrel', 'mediquin',
    'medistein', 'medium', 'mediven', 'medscab', 'mefenamat', 'mefenamic', 'mefentan', 'mefinal', 'mefix', 'meflam',
    'meflosin', 'mefoz', 'mefurosan', 'megaloblastik', 'megatic', 'megnesium', 'meiact', 'meibomitis', 'meiji', 'meixam',
    'melahirkan', 'melancarkan', 'melasma', 'melegakan', 'melembabkan', 'melindungi', 'melitus', 'mellitus', 'melocid', 'melon',
    'meloxicam', 'meloxin', 'memadai', 'memar', 'membantu', 'memberi', 'memberikan', 'membersihkan', 'membran', 'membutuhkan',
    'memelihara', 'memenuhi', 'memori', 'memperbaiki', 'mempercepat', 'memperlancar', 'memperoleh', 'memproduksi', 'memucil', 'menahun',
    'menambah', 'menangani', 'mencegah', 'mendadak', 'mengalami', 'mengancam', 'mengandung', 'mengatasi', 'mengeluarkan', 'mengencerkan',
    'mengeras', 'mengganggu', 'menghambat', 'menghangatkan', 'menghasilkan', 'menghilangkan', 'menghindari', 'mengobati', 'mengontrol', 'mengurangi',
    'meniere', 'menin', 'meningeal', 'meningitis', 'meningkatkan', 'meningkatnya', 'menjaga', 'menjalani', 'menopause', 'menses',
    'mensipox', 'menstimulasi', 'menstruasi', 'mental', 'menthol', 'menular', 'menurunkan', 'menyebabkan', 'menyegarkan', 'menyembuhkan',
    'menyertai', 'menyusui', 'mepromaag', 'meprotrin', 'meptin', 'mera', 'merah', 'mercotin', 'meredakan', 'meriang',
    'merimac', 'meringankan', 'merislon', 'merit', 'merron', 'mersibion', 'mertigo', 'mertus', 'merupakan', 'mesilate',
    'mesone', 'mestamox', 'mestinon', 'metabolik', 'metabolisme', 'metamizole', 'meteorisme', 'metformin', 'metham', 'methionine',
    'methisoprinol', 'methopi', 'methyl', 'metilev', 'metoclopramide', 'metphar', 'metrix', 'metronidazole', 'mevilox', 'mexon',
    'mexpharm', 'mextril', 'mexylin', 'mezatrin', 'mialgia', 'micardis', 'miconazole', 'microgest', 'microlax', 'microlut',
    'microtina', 'mielopati', 'mierin', 'mig', 'migra', 'migrain', 'migranal', 'migren', 'mikosis', 'mikroangiopati',
    'mikroorganisme', 'mild', 'miliaria', 'milk', 'millsen', 'milmor', 'milorin', 'mineral', 'minggu', 'mini',
    'miniaspi', 'minol', 'minor', 'minosep', 'minoxidil', 'mint', 'minyak', 'miokard', 'miom', 'miozidine',
    'mipi', 'mirabilis', 'miracloven', 'miradryl', 'mirapect', 'mirasic', 'miratadin', 'miravon', 'misalnya', 'mite',
    'miu', 'mix', 'mixadin', 'mixagrip', 'mixalgin', 'mixsaga', 'mixtard', 'mixture', 'mjs', 'mobiflex',
    'mocha', 'modalim', 'mofacort', 'mofulex', 'moisderm', 'mokbios', 'moladerm', 'molagit', 'molakrim', 'molaneuron',
    'molapect', 'molasic', 'molavir', 'molazol', 'molcin', 'molexflu', 'molluscum', 'moloco', 'molozcap', 'mom',
    'momet', 'mometasone', 'monarin', 'monecto', 'monell', 'monohydrate', 'mononitrate', 'monoterapi', 'montelukast', 'monuril',
    'moretic', 'morinda', 'morning', 'motaderm', 'moteson', 'motilium', 'mouthwash', 'movicox', 'movix', 'moxalas',
    'moxam', 'moxibat', 'moxic', 'moxifloxacin', 'moxigra', 'mual', 'mucera', 'mucohexin', 'mucopan', 'mucopect',
    'mucos', 'mucosta', 'mucotein', 'mucoxol', 'mudah', 'mukokutan', 'mukolitik', 'mukosa', 'mukosal', 'mukostasis',
    'mukovisidosis', 'mukus', 'mules', 'multiple', 'mulut', 'muncul', 'muntah', 'muntah-muntah', 'mupirocin', 'mups',
    'muricata', 'muscular', 'musiman', 'muskuloskeletal', 'mxn', 'myasthenia', 'mycetine', 'mycoral', 'mycorine', 'mycos',
    'mycospor', 'mycostatin', 'mycoz', 'mydriatil', 'mylanta', 'myllacid', 'myoman', 'myonal', 'myonep', 'myopia',
    'myores', 'myori', 'myositis', 'mytaderm', 'nafas', 'nafsu', 'nairet', 'naletal', 'nalgestan', 'nalitik',
    'nanotech', 'napa', 'napadisilate', 'napas', 'naphazoline', 'naprex', 'narfoz', 'nariz', 'nasacort', 'nasafed',
    'nasal', 'nasonex', 'natacen', 'natexam', 'natrilix', 'natrium', 'natrol', 'natural', 'naturale', 'naturals',
    'nature', 'natures', 'naturest', 'naturexp', 'naturic', 'naturprost', 'nazovel', 'nebacetin', 'nebilet', 'nebules',
    'neciblok', 'neem', 'nefropati', 'nefrotik', 'negatal', 'negatif', 'neisseria', 'nekreotika', 'nelicort', 'nellco',
    'neo', 'neodulax', 'neolyson', 'neomisin', 'neomycin', 'neoplasma', 'neosanmag', 'neosinol', 'neosma', 'neotibi',
    'neozep', 'nephrolit', 'nerd', 'nerilon', 'nerva', 'nervosa', 'nestacort', 'neuralgia', 'neuralgin', 'neuraxon',
    'neuritis', 'neuro', 'neurobat', 'neurobion', 'neurodermatitis', 'neurodex', 'neurofenac', 'neurohax', 'neurologi', 'neurologik',
    'neuromed', 'neuromuskuler', 'neuroparalisis', 'neuropati', 'neuropyronv', 'neurosanbe', 'neurotropik', 'neurovit', 'nevodio', 'nevox',
    'nevradin', 'new', 'newspar', 'nexium', 'niacef', 'nichodryl', 'nichofed', 'nichomycin', 'nicotinamide', 'niddm',
    'nifedin', 'nifedipine', 'niften', 'nifudiar', 'nifural', 'nifuroxazide', 'night', 'nikolam', 'nilacelin', 'nilacol',
    'nimotop', 'nipe', 'nipis', 'nisa', 'nisagon', 'nistagmus', 'nistatin', 'nistrol', 'nitacur', 'nitral',
    'nitrat', 'nitrate', 'nitrokaf', 'nixaven', 'nizol', 'nizoral', 'nocandis', 'nocid', 'noda', 'nodrof',
    'nogaze', 'nolipo', 'nomika', 'noncort', 'nong', 'nongonokokus', 'noni', 'nonik', 'noperten', 'nopril',
    'norages', 'norit', 'norizec', 'normal', 'normetec', 'norpid', 'norsec', 'norvask', 'noscapine', 'nosib',
    'nosirax', 'nosokomial', 'nostren', 'notritis', 'novabiotic', 'novadiar', 'novagesic', 'novagyl', 'novales', 'novamag',
    'novamox', 'novapyron', 'novatusin', 'novax', 'novaxifen', 'novexib', 'novomix', 'novorapid', 'noza', 'nubrex',
    'nucef', 'nucral', 'nufacort', 'nufapreg', 'nufaprim', 'nukleus', 'nurutenz', 'nutralix', 'nutrimax', 'nuvopec',
    'nuzartan', 'nyaman', 'nyamuk', 'nyeri', 'nymiko', 'nystatin', 'nystin', 'nytex', 'oaw', 'obana',
    'obasa', 'obat', 'obat2', 'obh', 'obimin', 'obp', 'obstruksi', 'obstruktif', 'ocidermn', 'ocuflam',
    'ocufresh', 'ocular', 'oculenta', 'ocuson', 'odanostin', 'odf', 'odr', 'odt', 'oestrogel', 'officinale',
    'officinalis', 'ofloxacin', 'ogb', 'oil', 'ointment', 'oksalat', 'oksovell', 'okular', 'okuler', 'olah',
    'olahraga', 'olbas', 'oles', 'oleum', 'olie', 'oligomenorea', 'oligospermia', 'olmetec', 'omecidal', 'omecough',
    'omed', 'omedom', 'omedrinat', 'omegavit', 'omegdiar', 'omegesic', 'omegrip', 'omegtamine', 'omekur', 'omemox',
    'omeneuron', 'omeprazole', 'omeric', 'omz', 'onbrez', 'ondane', 'ondansetron', 'ondavar', 'ondavell', 'onetic',
    'onkomikosis', 'oparin', 'operasi', 'ophtalmia', 'opicef', 'opiclam', 'opidiar', 'opilax', 'opimox', 'opistan',
    'opivask', 'opthil', 'optibet', 'opticom', 'optiflu', 'optik', 'optixitrol', 'oral', 'oralinu', 'oralit',
    'orang', 'orange', 'orezinc', 'organ', 'organisme', 'original', 'orinox', 'orkitis', 'orofaringeal', 'oros',
    'oroxin', 'orphen', 'orsaderm', 'orthosiphon', 'ortopedik', 'orvast', 'oscal', 'oseltamivir', 'oskadon', 'oskadryl',
    'ossopan', 'ossoral', 'ostarin', 'ostelox', 'osteo', 'osteoarthritis', 'osteoartritis', 'osteodistrofi', 'osteofar', 'osteogenesis',
    'osteomielitis', 'osteonate', 'osteoporosis', 'ostovel', 'ostriol', 'otak', 'otilon', 'otitis', 'otolin', 'otopain',
    'otorinolaringologi', 'otot', 'otozambon', 'ottogenta', 'ottopan', 'ottoprim', 'ovarium', 'oviskin', 'oviskinn', 'ovula',
    'ovulasi', 'oxfezin', 'oxide', 'oxoferin', 'oxomemazine', 'oxopect', 'oxopi', 'oxoril', 'oxtin', 'oxy',
    'ozen', 'ozid', 'pacdin', 'pacego', 'pacetik', 'paediatric', 'paget', 'pagoda', 'pahit', 'pain',
    'painkila', 'pajanan', 'pala', 'palenox', 'pallidum', 'palmetto', 'palmitate', 'palsy', 'pamoate', 'pamol',
    'panadol', 'panas', 'panasin', 'paniculata', 'panjang', 'pankreas', 'pankreatitis', 'pantoprazole', 'pantopump', 'pantothenate',
    'pantozol', 'panu', 'papua', 'papula', 'paracana', 'paracetamol', 'paracetine', 'paraco', 'parafen', 'parakoksidioidomikosis',
    'paralisis', 'paralitik', 'param', 'paramex', 'paramol', 'paramorina', 'paranervion', 'paratenza', 'paratifoid', 'paratifus',
    'paratusin', 'parau', 'parazon', 'parcok', 'parestesia', 'pariet', 'parkinson', 'paroksismal', 'paronikia', 'parotitis',
    'paroxismal', 'partum', 'paru', 'paru-paru', 'pasak', 'pasca', 'pasien', 'paska', 'pasquam', 'passion',
    'pasta', 'pastiles', 'patch', 'patogenik', 'patologis', 'paxlovid', 'payudara', 'pecah', 'pectin', 'pectorin',
    'pectoris', 'pectum', 'pedab', 'pedang', 'pedialyte', 'pediatric', 'pedis', 'peditox', 'pegal', 'pegal-pegal',
    'pehadoxin', 'pehamoxil', 'pehavask', 'pei', 'peipa', 'peka', 'pektoris', 'pelancar', 'pelangsing', 'pelargonium',
    'pelega', 'pelengkap', 'pellet', 'pellets', 'pelvis', 'pemakaian', 'pembedahan', 'pembengkakan', 'pemberian', 'pembuluh',
    'pemeliharaan', 'pemeriksaan', 'penambahan', 'penanganan', 'penatalaksanaan', 'pencabutan', 'pencegahan', 'pencernaan', 'pendaharan', 'pendarahan',
    'pendek', 'pendengaran', 'penderita', 'pendinginan', 'penfill', 'pengaturan', 'pengencer', 'pengendapan', 'pengganti', 'penggunaan',
    'penghambat', 'penghasil', 'penglihatan', 'pengobatan', 'pengosongan', 'pengurang', 'pengurangan', 'pening', 'peningkatan', 'penisilinase',
    'penmox', 'penta', 'penuh', 'penunjang', 'penurunan', 'penyakit', 'penyakit-penyakit', 'penyakit2', 'penyebab', 'penyebabnya',
    'penyegar', 'penyekat', 'penyembuhan', 'penyempitan', 'penyerapan', 'penyumbatan', 'peppermint', 'peptik', 'peptikum', 'peptovell',
    'pepzol', 'peradangan', 'perasaan', 'perawatan', 'perdarahan', 'pereda', 'peredaran', 'perenial', 'periadenitis', 'periartritis',
    'periartropati', 'pericarpium', 'perifas', 'perifer', 'perih', 'perindopril', 'periodontal', 'periodontitis', 'peritonitis', 'perjalanan',
    'permethrin', 'permukaan', 'permyo', 'pernafasan', 'pernapasan', 'pernisiosa', 'peroxide', 'persalinan', 'persendian', 'persiapan',
    'persisten', 'pertumbuhan', 'pertusis', 'perut', 'pessary', 'phaproxin', 'pharmafix', 'pharmalene', 'pharmaxil', 'pharolit',
    'phenerica', 'phenobiotic', 'phenol', 'phenylephrine', 'phenylpropanolamine', 'phenzacol', 'phlebitis', 'phosphate', 'phospho', 'phosphosoda',
    'photoaging', 'phylari', 'pibaksin', 'pida', 'pidea', 'pielitis', 'pielonefritis', 'pien', 'pil', 'pilaris',
    'pilek', 'pills', 'pilorus', 'pimacolin', 'pimag', 'pimtrakol', 'pin', 'pine', 'pinfetil', 'pinggang',
    'pinggul', 'pioderma', 'pioglitazone', 'pionix', 'piparlu', 'piralen', 'pirocam', 'pirofel', 'pirosis', 'pirotop',
    'piroxicam', 'pitiriasis', 'placenta', 'placta', 'pladogrel', 'plak', 'plantacid', 'planus', 'plaster', 'plasterrd',
    'platogrix', 'plavix', 'pletaal', 'plineuritis', 'plossa', 'plus', 'pluss', 'pneumococci', 'pneumocystis', 'pneumokokus',
    'pneumokoniosis', 'pneumonia', 'pneumoniae', 'point', 'polacel', 'polar', 'poldan', 'poliartritis', 'polidemisin', 'polifrisin',
    'polineuritis', 'polineuropati', 'poliomielitis', 'polip', 'polisistik', 'polisitemia', 'polofar', 'polycrol', 'polydex', 'polygran',
    'polymyxin', 'polynel', 'polypred', 'polysilane', 'poncodryl', 'pondex', 'ponflu', 'ponstelax', 'popok', 'portal',
    'positif', 'posop', 'post', 'potaflam', 'potassium', 'povidoneiodine', 'powder', 'ppok', 'prabetic', 'pradaxa',
    'pralax', 'pratifar', 'pravastatin', 'pravinat', 'praxion', 'prazina', 'prazotec', 'preabor', 'precatorius', 'prednisolone',
    'pregnolin', 'pregtenol', 'pregvomit', 'premaston', 'prematur', 'premenstrual', 'premium', 'prenamia', 'presentasi', 'press',
    'prestrenol', 'pria', 'pride', 'prima', 'primadex', 'primadol', 'prime', 'primer', 'primpen', 'primperan',
    'primrose', 'primunox', 'prinol', 'prinzmetal', 'pritagesic', 'pritamox', 'pritanol', 'pro', 'proast', 'probaflek',
    'probiomag', 'probiotin', 'procaterol', 'proceles', 'procold', 'procolic', 'prodermis', 'produksi', 'produktif', 'profat',
    'profen', 'profenal', 'profenid', 'profertil', 'profibrat', 'profilaksis', 'profilas', 'profim', 'progesteron', 'progesterone',
    'progeston', 'progyl', 'proktitis', 'prolaps', 'prolecin', 'prolic', 'prolinu', 'prolipid', 'promag', 'prome',
    'promedex', 'promeno', 'promethazine', 'promixin', 'promuxol', 'pronalges', 'pronam', 'pronicy', 'pronovir', 'propepsa',
    'propionate', 'propolplus', 'propranolol', 'propyphenazone', 'propyretic', 'prorenal', 'proris', 'prose', 'prosedur', 'proses',
    'prosic', 'prosinal', 'prosogan', 'proson', 'prostakur', 'prostalon', 'prostanac', 'prostat', 'prostatic', 'prostatitis',
    'prostresa', 'protagenta', 'protection', 'protein', 'proterol', 'proteus', 'protifed', 'protocort', 'protozoa', 'prouric',
    'provagin', 'provamed', 'prove', 'provera', 'provomer', 'provula', 'proxime', 'proxona', 'prurigo', 'pruritis',
    'pruritus', 'pseudoephedrine', 'pseudohipoparatiroidisme', 'psikologis', 'psitakosis', 'psoriasis', 'pulmicort', 'pulmonal', 'pulmoner', 'pulposus',
    'pumpitor', 'pundak', 'punggung', 'pure', 'purekids', 'puricemia', 'purpura', 'purpurea', 'pusaka', 'pusing',
    'pustula', 'pustularis', 'putih', 'puyer', 'pxm', 'pyderma', 'pyfakof', 'pylori', 'pylorus', 'pyrantel',
    'pyrathiazine', 'pyravit', 'pyrazinamide', 'pyrex', 'pyrexin', 'pyricef', 'pyricin', 'pyridol', 'pyridoxine', 'pyridryl',
    'qcef', 'quantidex', 'quidex', 'quill', 'quinobiotic', 'rachitis', 'racun', 'radang', 'radiasi', 'radikal',
    'radioterapi', 'radix', 'raga', 'ragamasalah', 'rahim', 'rakitis', 'rambut', 'ramipril', 'ramolit', 'ran',
    'rancus', 'ranexa', 'rangka', 'ranicho', 'ranitidine', 'ranivel', 'rantin', 'rapet', 'rapihaler', 'rasa',
    'rasilez', 'ratinal', 'raven', 'raynaud', 'reaksi', 'rebal', 'rebamax', 'rebamid', 'rebamipide', 'recansa',
    'rechol', 'reco', 'recustein', 'red', 'redacid', 'redusec', 'refaquin', 'refluks', 'reflux', 'refrakter',
    'refreshing', 'regit', 'reglus', 'regrou', 'regular', 'reguler', 'rehidrasi', 'rejan', 'rektum', 'rekuren',
    'rekurens', 'relaxing', 'relief', 'relieving', 'remabrex', 'remact', 'remago', 'remaja', 'rematik', 'rematoid',
    'removchol', 'renabetic', 'renadinac', 'renalof', 'renapar', 'renapepsa', 'renasistin', 'renator', 'rendah', 'rendapid',
    'rentan', 'repass', 'repens', 'repimide', 'reponsif', 'resida', 'resiko', 'resisten', 'respimat', 'respira',
    'respituss', 'respon', 'respons', 'responsif', 'respules', 'retaphyl', 'retard', 'retensi', 'reticor', 'retinol',
    'retinopathy', 'reucid', 'reumatik', 'reumatoid', 'rexavin', 'rexcof', 'rhelafen', 'rhema', 'rheumacyl', 'rheumason',
    'rheumatoid', 'rhinitis', 'rhinofed', 'rhinorrhea', 'rhinos', 'rhizoma', 'rhodium', 'rhomuz', 'riboflavin', 'ricasid',
    'rickettsia', 'ricovir', 'rif', 'rifampicin', 'rifamtibi', 'rifastar', 'riketsia', 'rinclo', 'rindobion', 'rindoflox',
    'rindopump', 'ringan', 'rinitis', 'rinocet', 'rinofaringitis', 'rinse', 'rinvox', 'risina', 'ristonat', 'ritez',
    'riwayat', 'rodeca', 'rodehond', 'rodemol', 'rohto', 'roksicap', 'roll', 'roller', 'ronazol', 'rongga',
    'rontok', 'rose', 'rostin', 'rosufer', 'rosuvastatin', 'roswin', 'rotaver', 'rovadin', 'rovastar', 'rovator',
    'roverton', 'roxidene', 'roxithromycin', 'rozgra', 'ruam', 'rub', 'rubrum', 'rumin', 'rupafin', 'ruvastin',
    'rydian', 'rytmonorm', 'ryvel', 'saat', 'sachet', 'safe', 'sagalon', 'sagestam', 'sakit', 'salas',
    'salbron', 'salbutamol', 'salbuven', 'salep', 'salfenal', 'salgen', 'salicyl', 'salicylate', 'salicylic', 'salisilat',
    'salmeterol', 'salmonella', 'salonpas', 'salp', 'salpingitis', 'salticin', 'saluran', 'salut', 'sambiloto', 'samconal',
    'samcovask', 'sampai', 'samping', 'samtacid', 'samurat', 'san', 'sanadryl', 'sanaflu', 'sanazet', 'sanazol',
    'sanbe', 'sanbehair', 'sanbeivy', 'sanbenafil', 'sancortmycin', 'sandalwood', 'sanfuro', 'sangobion', 'sanjin', 'sanmag',
    'sanmetidin', 'sanmol', 'sanorine', 'sanpicillin', 'sanprima', 'sansulin', 'santesar', 'santibi', 'santo', 'santopect',
    'sapu', 'sar', 'saraf', 'sarang', 'sari', 'sariaman', 'sariawan', 'sariayu', 'saridon', 'sarmut',
    'sativa', 'sativi', 'sauda', 'saw', 'scabicore', 'scabimite', 'scacid', 'scalp', 'scanaflam', 'scanderma',
    'scanneuron', 'scanovir', 'scantoma', 'scarlet', 'scholaris', 'scopamin', 'scopma', 'scorpio', 'scrub', 'sdigest',
    'sea', 'seahorse', 'sean', 'season', 'seasonal', 'sebabkan', 'sebagai', 'sebelum', 'seboroik', 'secara',
    'sedang', 'sedrofen', 'seed', 'segala', 'segar', 'segmen', 'sehat', 'sehubungan', 'sekresi', 'sekretolitik',
    'seksual', 'sekunder', 'selama', 'selaput', 'selbix', 'selesma', 'selsun', 'selulitis', 'selvim', 'sembelit',
    'semen', 'sementara', 'semprot', 'semua', 'semut', 'sendawa', 'sendi', 'seng', 'senna', 'sensitif',
    'seperti', 'sepsis', 'septic', 'septik', 'septikemia', 'sequest', 'serak', 'serangan', 'serangga', 'serebral',
    'seree', 'sereh', 'seremig', 'serenoa', 'seretide', 'serius', 'serolin', 'serum', 'servikal', 'servisitis',
    'sesak', 'sesden', 'sesquihydrate', 'sesuai', 'sesudah', 'setara', 'setelah', 'shallaki', 'sharox', 'shen',
    'sheuw', 'shigella', 'shock', 'shuang', 'siang', 'sibelium', 'siberid', 'sicca', 'sickness', 'siclidon',
    'sido', 'sidoides', 'sidola', 'sie', 'sifilis', 'sikloplegia', 'sikstop', 'siladex', 'sildenafil', 'silex',
    'silicon', 'silopect', 'siloxan', 'silum', 'silver', 'simarc', 'simclovix', 'simdes', 'simethicone', 'simfed',
    'simflamfas', 'simnos', 'simplek', 'simpleks', 'simplex', 'simprofen', 'simptom', 'simptomatik', 'simryl', 'simtomatik',
    'simtor', 'simucil', 'simvastatin', 'simvasto', 'sinar', 'sinargi', 'sindrom', 'sindroma', 'sinensis', 'sinergi',
    'singulair', 'sinobiotik', 'sinocort', 'sinova', 'sinral', 'sintrol', 'sinusitis', 'sipilis', 'siramid', 'siran',
    'sirdalud', 'sirih', 'sirkulasi', 'sirkumsisi', 'sirosis', 'sirsak', 'sirup', 'sismax', 'sisoprim', 'sistem',
    'sistemik', 'sistitis', 'sistolik', 'sitagliptin', 'sitostatik', 'sitotoksik', 'sitro', 'sitronela', 'situroxime', 'sjogren',
    'skiatika', 'skilone', 'skizon', 'skleritis', 'sklerosis', 'skorbut', 'skotoma', 'snoozzz', 'sobusty', 'soda',
    'sodium', 'sofratulle', 'sohobion', 'solac', 'solaneuron', 'solaxin', 'soldextam', 'solinfec', 'soliqua', 'solpenox',
    'solution', 'somevell', 'sominal', 'soothe', 'soricox', 'sotatic', 'spaf', 'spar', 'spasmal', 'spasme',
    'spasminal', 'spasmolit', 'spastik', 'spastisitas', 'special', 'spedifen', 'spesifik', 'spiramycin', 'spiranter', 'spiritus',
    'spiriva', 'spirola', 'spirolacton', 'spironolactone', 'spirulina', 'splash', 'spondilartritis', 'spondilitis', 'spondylitis', 'sporacid',
    'sporanox', 'sporetik', 'sporotrikosis', 'sport', 'spots', 'spray', 'spyrocon', 'srongpas', 'srp', 'stabil',
    'stadium', 'staforin', 'stamen', 'stamina', 'stamotens', 'staph', 'staphylococci', 'staphylococcus', 'starcef', 'starvit',
    'statcol', 'stator', 'statrol', 'status', 'stavinor', 'steinleventhal', 'stenosis', 'steril', 'steroid', 'stimuno',
    'stinopi', 'stolax', 'stomacain', 'stomach', 'stomatitis', 'stop', 'stopx', 'strabismus', 'strain', 'strath',
    'strawberi', 'strawberry', 'strep', 'strepsils', 'streptococci', 'streptococcus', 'stres', 'striata', 'stroberi', 'strocain',
    'stroke', 'strong', 'struktur', 'stugeron', 'subarachnoid', 'succinate', 'succus', 'sucralbat', 'sucralfate', 'sudut',
    'sugar', 'suggy', 'sujamer', 'sulfadiazine', 'sulfamethoxazole', 'sulfate', 'sulfogaiacol', 'sulfonilurea', 'sulfonylurea', 'sulit',
    'sumagesic', 'sumang', 'sumbatan', 'sumsum', 'super', 'superfisial', 'superfisialis', 'superhoid', 'suplemen', 'suppositoria',
    'supra', 'suprabiotic', 'supramox', 'suprazid', 'surya', 'suspensi', 'suspension', 'susu', 'susut', 'suvesco',
    'svt', 'syamil', 'syifa', 'symbicort', 'sympathetic', 'synalten', 'synarcus', 'syndrome', 'syrup', 'sysmuco',
    'sytin', 'tablet', 'tadalafil', 'tahan', 'tahun', 'tai', 'takikardia', 'talion', 'tamacap', 'tambah',
    'tambahan', 'tambang', 'tamcocin', 'tamezol', 'tamezole', 'tanapress', 'tanda', 'tangan', 'tanpa', 'tantum',
    'tarivid', 'tarka', 'tarsal', 'tasngin', 'tawar', 'tawon', 'taxilan', 'taxime', 'tay', 'tea',
    'tears', 'teaslim', 'teck', 'teens', 'teh', 'tekanan', 'telfast', 'telinga', 'telmisartan', 'telon',
    'telsat', 'tempra', 'temufit', 'tenace', 'tenangin', 'tenazide', 'tendinitis', 'tendovaginitis', 'tenggorokan', 'tenosynovitis',
    'tensicap', 'tensinop', 'tensivask', 'teosal', 'tequinol', 'tera', 'teraf', 'terapi', 'terapinya', 'terasma',
    'teratur', 'terazosin', 'terbakar', 'terbinafine', 'terbuka', 'terbutaline', 'terhadap', 'terinfeksi', 'terkait', 'terkena',
    'terkilir', 'terkomplikasi', 'terkontrol', 'terlambat', 'termagon', 'termasuk', 'termisil', 'termorex', 'terpisah', 'terpukul',
    'tersebut', 'tersiram', 'tersumbat', 'terutama', 'tetanus', 'tetes', 'tetra', 'tetracycline', 'tetrahydrozoline', 'tetrasanbe',
    'tetrin', 'teveten', 'thecort', 'theobron', 'theoclate', 'theophylline', 'theragranm', 'therapain', 'therapy', 'theravask',
    'thiamfilex', 'thiamine', 'thiamphenicol', 'thiamycin', 'thiazide', 'thislacol', 'thomson', 'threonine', 'throat', 'thrombo',
    'thrombogel', 'thrombophob', 'thromboreductin', 'thrombosis', 'thyponisix', 'tiamin', 'tian', 'tiaryt', 'tibigon', 'ticagrelor',
    'ticomag', 'tidur', 'tifestan', 'tifoid', 'tifus', 'tiga', 'tigalin', 'tiger', 'tilaar', 'tilidon',
    'timbul', 'timol', 'timolol', 'timophtal', 'tindakan', 'tinea', 'tingkat', 'tinitus', 'tinja', 'tinnitus',
    'tipe', 'tiriz', 'tirotoksikosis', 'tisacef', 'tismalin', 'titan', 'tizacom', 'tizanidine', 'tjau', 'tjing',
    'tkv', 'tmp', 'tobramycin', 'tobro', 'tobroson', 'toedjoe', 'tofedex', 'toga', 'tokasid', 'toksoplasmosis',
    'tolak', 'tomaag', 'tomit', 'tong', 'tonsilitis', 'top', 'topcillin', 'topcort', 'topgesic', 'topgra',
    'topical', 'topikal', 'topisel', 'topsy', 'torasic', 'total', 'toxaprim', 'tracetate', 'trachomatis', 'trachon',
    'tradisional', 'trajenta', 'trakeabronkitis', 'trakhitis', 'traktus', 'transbroncho', 'transpulmin', 'trauma', 'traumatik', 'travatan',
    'treatment', 'tremens', 'tremenza', 'tremor', 'trental', 'trentin', 'treponema', 'tresiba', 'tresno', 'tretinoin',
    'triamcinolone', 'triamcorta', 'triaminic', 'triatec', 'trichodazol', 'trichomonas', 'trichomoniasis', 'trichophyton', 'trichostatic', 'trichostrongylus',
    'tricker', 'tride', 'trifachlor', 'trifamol', 'trifastan', 'trifed', 'trifedrin', 'trigliserida', 'trihydrate', 'trikomoniasis',
    'trimetazidine', 'trimethoprim', 'trimoxsul', 'triocid', 'triprolidine', 'triptagic', 'trisela', 'trisilicate', 'trizedon', 'troches',
    'trodex', 'trofik', 'trogyl', 'trogystatin', 'trolip', 'tromboflebitis', 'trombosis', 'trombosit', 'trombositemia', 'trometamol',
    'tropica', 'tropidryl', 'tropik', 'tropin', 'tropine', 'trosyd', 'trovensis', 'troviakol', 'trozin', 'truvaz',
    'tryptophan', 'tuberkulosis', 'tuboovarium', 'tubuh', 'tudiab', 'tukak', 'tulang', 'tularemia', 'tuli', 'tummy',
    'tumor', 'tunggal', 'tunjong', 'turbuhaler', 'turpas', 'tuzalos', 'twynsta', 'tylonic', 'type', 'tyran',
    'tyrosine', 'tze', 'udara', 'udv', 'ulcer', 'ulcera', 'ulcerid', 'ulcori', 'ulcron', 'ulcumaag',
    'ulkus', 'ulmifolia', 'ulmo', 'ulsafate', 'ulserasi', 'ulseratif', 'ulsicral', 'ulsidex', 'ultra', 'ultracyn',
    'ultraflu', 'ultraproct', 'ultrasiline', 'ultraxon', 'uluhati', 'unalium', 'uni', 'unigen', 'unisia', 'upbrainina',
    'uperio', 'uplores', 'urafit', 'urat', 'ureaplasma', 'uresix', 'urethral', 'urethritis', 'uretritis', 'uretrosistitis',
    'uretrotrigonitis', 'urfamycin', 'urin', 'urine', 'urispas', 'urixin', 'urogenital', 'urotractin', 'urticaria', 'urtikaria',
    'urut', 'usia', 'usus', 'uterus', 'utrogestan', 'uung', 'uveal', 'uveitis', 'uwuh', 'vaclo',
    'vagina', 'vaginal', 'vaginalis', 'vaginitis', 'vaginosis', 'vagistin', 'vaksinasi', 'valaciclovir', 'valansim', 'valcor',
    'valemia', 'valerate', 'valsartan', 'valtrex', 'valved', 'valvir', 'vaporin', 'vaporub', 'vapril', 'vardipin',
    'varian', 'varicela', 'varises', 'varoc', 'vasacon', 'vasgard', 'vask', 'vaskular', 'vaslone', 'vasokonstriksi',
    'vasomotor', 'vasospasme', 'vasospastik', 'vastigo', 'vaxom', 'vbloc', 'vectrine', 'vedium', 'vegamon', 'vegeta',
    'vein', 'velacom', 'velutine', 'vena', 'venaron', 'venosmil', 'ventolin', 'ventrikel', 'ventrikular', 'vera',
    'verde', 'vergo', 'veridin', 'vermox', 'vernacel', 'versibet', 'versicolor', 'versikolor', 'versilon', 'vertebra',
    'vertigo', 'vertilon', 'vertivom', 'vertizine', 'vesicare', 'vestein', 'vestibular', 'vfresh', 'viadoxin', 'viagra',
    'viajoy', 'viastar', 'vibramox', 'vibranat', 'vicks', 'victoza', 'viflox', 'vildagliptin', 'vildi', 'vincent',
    'vioquin', 'vipalbumin', 'viridis', 'virinol', 'virus', 'visancort', 'vision', 'visto', 'vitacid', 'vitafol',
    'vital', 'vitalitas', 'vitamin', 'vitaquin', 'vitomata', 'vitor', 'vitrolenta', 'vitropic', 'vivace', 'vocefa',
    'vocinti', 'volequin', 'volinol', 'voltadex', 'voltaren', 'volten', 'vomceran', 'vomecho', 'vometa', 'vometron',
    'vomigo', 'vomina', 'vomistop', 'vomitas', 'vomizole', 'vosea', 'vosedon', 'vostrin', 'voxin', 'vroxil',
    'vulgare', 'vulgaris', 'vulvitis', 'vytorin', 'wahid', 'waisan', 'wangi', 'wanita', 'wasir', 'wassing',
    'watermelon', 'wedang', 'wei', 'wellness', 'wet', 'wheat', 'wiaflox', 'widoxil', 'winatin', 'wiros',
    'wonder', 'wong', 'wood', 'woods', 'wybert', 'xalacom', 'xanthorrhiza', 'xanturic', 'xarelto', 'xepabet',
    'xepalium', 'xepamol', 'xepare', 'xgra', 'xiltrop', 'ximex', 'xinafoate', 'xirat', 'xitrol', 'xybat',
    'xylocaine', 'xyzal', 'yafix', 'yao', 'yaridon', 'yasmin', 'yaz', 'yellow', 'yick', 'yokoyoko',
    'yosenob', 'yrins', 'yung', 'yunnan', 'yusimox', 'zalf', 'zambuk', 'zanic', 'zanidip', 'zantifar',
    'zecamex', 'zecaneuron', 'zegren', 'zelface', 'zeliris', 'zelona', 'zemoxil', 'zemyc', 'zendalat', 'zenichlor',
    'zenirex', 'zensoderm', 'zenti', 'zentra', 'zeprazol', 'zeropain', 'zestmag', 'zevask', 'zhang', 'zhou',
    'zibramax', 'zidalev', 'ziloven', 'zinc', 'zincpro', 'zine', 'zingiber', 'zinkid', 'ziplong', 'zircum',
    'zistic', 'zitanid', 'zithrax', 'zithrolic', 'zithromax', 'zitromed', 'zok', 'zolacap', 'zolacort', 'zolagel',
    'zoline', 'zollinger', 'zollinger-ellison', 'zoloral', 'zoralin', 'zoster', 'zoter', 'zovirax', 'zultrop', 'zwitsal',
    'zycin', 'zyloric',
}

# DRUG TYPES FROM DATASET
DRUG_TYPES = {'balsam', 'douche', 'enema', 'gel', 'granula', 'inhaler', 'injeksi', 'jamu', 'kaplet', 'kapsul', 'kasa', 'krim', 'larutan', 'madu', 'minyak', 'odf', 'ovula', 'pasta', 'pessary', 'pil', 'powder', 'roll', 'sachet', 'salep', 'sirup', 'spray', 'suppositoria', 'suspensi', 'tablet', 'tetes', 'vaginal'}

# NON-MEDICAL TERMS (for filtering)
NON_MEDICAL_TERMS = {
    'laptop', 'komputer', 'computer', 'handphone', 'hp', 'gadget', 'elektronik',
    'mobil', 'motor', 'kendaraan', 'car', 'vehicle', 'rumah', 'house', 'home',
    'makanan', 'food', 'minuman', 'drink', 'kue', 'cake', 'masakan', 'cooking',
    'resep masakan', 'recipe', 'furniture', 'mebel', 'pakaian', 'clothing',
    'baju', 'sepatu', 'shoes', 'tas', 'bag', 'perhiasan', 'jewelry',
    'kasih sayang', 'cinta', 'love', 'sayang', 'affection', 'romantis',
    'pacaran', 'dating', 'menikah', 'marriage', 'keluarga bahagia'
}

def is_medical_query(query: str) -> bool:
    """
    FIXED: Check if query is medical-related using global MEDICAL_KEYWORDS
    
    Args:
        query: Search query string
        
    Returns:
        True if medical query, False otherwise
    """
    query_lower = query.lower()
    words = query_lower.split()
    
    # PRIORITY 1: Check NON-medical terms first (immediate rejection)
    for non_medical in NON_MEDICAL_TERMS:
        if non_medical in query_lower:
            return False
    
    # PRIORITY 2: Check drug types (high confidence medical)
    for drug_type in DRUG_TYPES:
        if drug_type in query_lower:
            return True
    
    # PRIORITY 3: Count medical keywords
    medical_count = 0
    total_meaningful_words = 0
    
    stopwords = {
        'dan', 'atau', 'untuk', 'yang', 'dengan', 'pada', 'di', 'ke', 'dari',
        'in', 'on', 'at', 'to', 'for', 'the', 'a', 'an', 'is', 'are', 'ini', 'itu'
    }
    
    for word in words:
        # Skip stopwords and very short words
        if word in stopwords or len(word) < 3:
            continue
            
        total_meaningful_words += 1
        
        # Exact match
        if word in MEDICAL_KEYWORDS:
            medical_count += 1
        # Partial match (for compound words)
        elif len(word) > 4:
            if any(med in word or word in med 
                   for med in MEDICAL_KEYWORDS if len(med) > 4):
                medical_count += 0.5
    
    # No medical keywords found
    if medical_count == 0:
        return False
    
    # For short queries (1-2 meaningful words), need at least 1 medical keyword
    if total_meaningful_words <= 2:
        return medical_count >= 1
    
    # For longer queries, need at least 30% medical keywords
    medical_ratio = medical_count / total_meaningful_words if total_meaningful_words > 0 else 0
    return medical_ratio >= 0.3
    
    # Check if any medical keyword exists in query
    for keyword in medical_keywords:
        if keyword in query_lower:
            return True
    
    return False


def check_results_relevance(results: List[Dict], min_score: float = 0.20) -> bool:
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
    """IMPROVED: Validate with adaptive thresholds"""
    is_medical = is_medical_query(query)
    
    if not results or len(results) == 0:
        if is_medical:
            return {
                'is_valid': False,
                'reason': 'no_relevant_results',
                'message': 'Obat tidak ditemukan dalam database',
                'filtered_results': []
            }
        else:
            return {
                'is_valid': False,
                'reason': 'non_medical_query',
                'message': 'Query tidak berhubungan dengan obat atau kesehatan',
                'filtered_results': []
            }
    
    # ADAPTIVE THRESHOLDS
    if is_medical:
        # Medical: Accept >= 0.20
        top_score = max([r.get('score', 0) for r in results])
        
        if top_score < 0.25:
            filtered = [r for r in results if r.get('score', 0) >= 0.15]
        else:
            filtered = [r for r in results if r.get('score', 0) >= 0.20]
        
        if filtered:
            return {
                'is_valid': True,
                'reason': 'valid',
                'message': 'Hasil ditemukan',
                'filtered_results': filtered
            }
        else:
            return {
                'is_valid': False,
                'reason': 'no_relevant_results',
                'message': 'Obat tidak ditemukan dalam database',
                'filtered_results': []
            }
    else:
        # Non-medical: STRICT >= 0.40
        filtered = [r for r in results if r.get('score', 0) >= 0.40]
        
        if not filtered:
            return {
                'is_valid': False,
                'reason': 'non_medical_query',
                'message': 'Query tidak berhubungan dengan obat atau kesehatan',
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
