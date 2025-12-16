# app/main_hybrid.py
"""
MedicIR FastAPI Server - MiniLM + TF-IDF Hybrid Version
FIXED: Single app instance with lifespan
"""

import sys
import os

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging
from contextlib import asynccontextmanager
from fastapi.responses import FileResponse
from pathlib import Path

# Import MiniLM model loader
try:
    from app.model_loader_hybrid import (
        smart_search,
        sanitize_search_results,
        get_available_methods,
        detect_query_type,
        calculate_adaptive_topk,
        validate_search_results,
    )
except ImportError:
    from model_loader_hybrid import (
        smart_search,
        sanitize_search_results,
        get_available_methods,
        detect_query_type,
        calculate_adaptive_topk,
        validate_search_results,
    )

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# LIFESPAN CONTEXT MANAGER
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("=" * 70)
    logger.info("MedicIR API - MiniLM + TF-IDF Hybrid")
    logger.info("=" * 70)
    
    methods = get_available_methods()
    logger.info("\nAvailable Methods:")
    for method, available in methods.items():
        status = "✓" if available else "✗"
        logger.info(f"  {status} {method}")
    
    logger.info("\n" + "=" * 70)
    logger.info("🌐 WEB APPLICATION ACCESS:")
    logger.info("=" * 70)
    logger.info("\n  ➜ Open in browser: http://localhost:8000")
    logger.info("  ➜ Alternative URL:  http://127.0.0.1:8000")
    logger.info("\n  (DO NOT open index.html file directly!)")
    logger.info("  (Access via browser using URL above)")
    logger.info("\n" + "=" * 70)
    logger.info("📚 API Documentation:")
    logger.info("=" * 70)
    logger.info("\n  ➜ Swagger UI: http://localhost:8000/docs")
    logger.info("  ➜ ReDoc:      http://localhost:8000/redoc")
    logger.info("  ➜ Health:     http://localhost:8000/health")
    logger.info("\n" + "=" * 70)
    
    yield
    
    # Shutdown
    logger.info("MedicIR API Shutting down...")

# ============================================================================
# FASTAPI APP - SINGLE INSTANCE WITH LIFESPAN
# ============================================================================

app = FastAPI(
    title="MedicIR API - MiniLM Hybrid",
    description="Medical Information Retrieval with MiniLM + TF-IDF",
    version="2.0-MiniLM",
    lifespan=lifespan  # ✅ Include lifespan from the start!
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files - FIXED PATH
from pathlib import Path

# Get the correct static path
static_dir = Path(__file__).parent / "static"  # ✅ Karena main_hybrid.py sudah di dalam app/

if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    logger.info(f"✅ Static files mounted from: {static_dir}")
    # List files for debugging
    for f in static_dir.iterdir():
        logger.info(f"   - {f.name}")
else:
    logger.error(f"❌ Static directory NOT FOUND: {static_dir}")

# Serve index.html at root
@app.get("/")
async def read_root():
    index_path = Path(__file__).parent / "index.html"
    return FileResponse(index_path)

# ============================================================================
# MODELS
# ============================================================================

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 20
    method: Optional[str] = "hybrid"
    page: Optional[int] = 0
    per_page: Optional[int] = 20


class SearchResponse(BaseModel):
    query: str
    method_used: str
    total_results: int
    results: List[Dict]
    query_type: str
    page: int
    per_page: int
    has_more: bool


class HealthResponse(BaseModel):
    status: str
    model: str
    available_methods: Dict[str, bool]
    total_documents: int


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", response_class=HTMLResponse, tags=["Root"])
async def root():
    """Serve index.html from project root"""
    try:
        html_path = os.path.join(parent_dir, "index.html")
        
        logger.info(f"Looking for index.html at: {html_path}")
        logger.info(f"File exists: {os.path.exists(html_path)}")
        
        if os.path.exists(html_path):
            logger.info(f"✅ Serving index.html ({os.path.getsize(html_path)} bytes)")
            with open(html_path, 'r', encoding='utf-8') as f:
                return HTMLResponse(content=f.read())
        else:
            logger.warning(f"❌ index.html not found at: {html_path}")
            return JSONResponse({
                "message": "MedicIR API - MiniLM + TF-IDF Hybrid",
                "model": "all-MiniLM-L6-v2",
                "note": "Place index.html in project root to see web interface",
                "expected_path": html_path,
                "api_docs": "/docs",
                "health": "/health",
                "search_example": "/search?query=obat+diare&method=hybrid"
            })
    except Exception as e:
        logger.error(f"Error serving HTML: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint"""
    try:
        from app.model_loader_hybrid import df_docs
    except ImportError:
        from model_loader_hybrid import df_docs
    
    methods = get_available_methods()
    
    return {
        "status": "healthy",
        "model": "MiniLM + TF-IDF Hybrid",
        "available_methods": methods,
        "total_documents": len(df_docs) if df_docs is not None else 0
    }


@app.get("/methods", tags=["System"])
async def get_methods():
    """Get available search methods"""
    methods = get_available_methods()
    
    return {
        "methods": methods,
        "descriptions": {
            "minilm": "MiniLM semantic search (fast and efficient)",
            "tfidf": "TF-IDF vector similarity (keyword-based)",
            "hybrid": "Hybrid MiniLM + TF-IDF (BEST - RECOMMENDED)"
        },
        "default_method": "hybrid",
        "recommendation": "Use 'hybrid' for best medical search results"
    }


@app.get("/search", response_model=SearchResponse, tags=["Search"])
async def search_endpoint(
    query: str = Query(..., description="Search query (Indonesian)", min_length=1),
    method: str = Query("hybrid", description="Search method: hybrid (recommended), minilm, tfidf"),
    top_k: int = Query(20, ge=1, le=100, description="Number of results"),
    page: int = Query(0, ge=0, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Results per page"),
):
    """Main search endpoint with relevance validation"""
    try:
        query_type = detect_query_type(query)
        
        if query_type == "multi-hop":
            adaptive_k = calculate_adaptive_topk(query, per_page, page)
            search_k = min(adaptive_k, top_k * 2)
        else:
            search_k = top_k
        
        results, method_used = smart_search(query, search_k, method)
        results = sanitize_search_results(results)
        
        # VALIDATE RESULTS
        validation = validate_search_results(query, results)
        
        if not validation['is_valid']:
            return {
                "query": query,
                "method_used": method_used,
                "total_results": 0,
                "results": [],
                "query_type": query_type,
                "page": page,
                "per_page": per_page,
                "has_more": False,
                "validation": {
                    "is_valid": False,
                    "reason": validation['reason'],
                    "message": validation['message']
                }
            }
        
        results = validation['filtered_results']
        
        start_idx = page * per_page
        end_idx = start_idx + per_page
        paginated_results = results[start_idx:end_idx]
        has_more = end_idx < len(results)
        
        return {
            "query": query,
            "method_used": method_used,
            "total_results": len(results),
            "results": paginated_results,
            "query_type": query_type,
            "page": page,
            "per_page": per_page,
            "has_more": has_more,
            "validation": {
                "is_valid": True,
                "reason": "valid",
                "message": "Hasil ditemukan"
            }
        }
    
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search_post(request: SearchRequest):
    """POST version of search endpoint"""
    try:
        query_type = detect_query_type(request.query)
        
        if query_type == "multi-hop":
            adaptive_k = calculate_adaptive_topk(request.query, request.per_page, request.page)
            search_k = min(adaptive_k, request.top_k * 2)
        else:
            search_k = request.top_k
        
        results, method_used = smart_search(request.query, search_k, request.method)
        results = sanitize_search_results(results)
        
        start_idx = request.page * request.per_page
        end_idx = start_idx + request.per_page
        paginated_results = results[start_idx:end_idx]
        has_more = end_idx < len(results)
        
        return {
            "query": request.query,
            "method_used": method_used,
            "total_results": len(results),
            "results": paginated_results,
            "query_type": query_type,
            "page": request.page,
            "per_page": request.per_page,
            "has_more": has_more
        }
    
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/doc/{doc_id}", tags=["Documents"])
async def get_document(doc_id: int):
    """Get specific document by ID"""
    try:
        from app.model_loader_hybrid import df_docs
    except ImportError:
        from model_loader_hybrid import df_docs
    
    import pandas as pd
    
    try:
        if df_docs is None or doc_id >= len(df_docs):
            raise HTTPException(status_code=404, detail="Document not found")
        
        doc = df_docs.iloc[doc_id].to_dict()
        clean_doc = {k: (v if not pd.isna(v) else "") for k, v in doc.items()}
        clean_doc['doc_id'] = doc_id
        
        return clean_doc
    
    except Exception as e:
        logger.error(f"Error fetching document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", tags=["System"])
async def get_statistics():
    """Get system statistics"""
    try:
        from app.model_loader_hybrid import df_docs
    except ImportError:
        from model_loader_hybrid import df_docs
    
    if df_docs is None:
        return {"error": "No data loaded"}
    
    stats = {
        "total_documents": len(df_docs),
        "model": "MiniLM + TF-IDF Hybrid",
        "columns": list(df_docs.columns),
        "available_methods": get_available_methods(),
    }
    
    if 'jenis_obat' in df_docs.columns:
        stats['jenis_obat_distribution'] = df_docs['jenis_obat'].value_counts().to_dict()
    
    return stats


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    import traceback
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 70)
    print("🚀 Starting MedicIR Server - MiniLM + TF-IDF Hybrid")
    print("=" * 70)
    print("\n🌐 Server will be available at:")
    print("   → http://localhost:8000")
    print("   → http://127.0.0.1:8000")
    print("\n📚 API Documentation:")
    print("   → http://localhost:8000/docs (Swagger UI)")
    print("   → http://localhost:8000/redoc (ReDoc)")
    print("\n🔍 Quick Test:")
    print("   → http://localhost:8000/health")
    print("   → http://localhost:8000/search?query=obat+diare")
    print("\n⚡ Press CTRL+C to stop the server")
    print("=" * 70 + "\n")
    
    uvicorn.run(
        "app.main_hybrid:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
