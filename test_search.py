# tests/test_search.py
"""
Unit tests for search functions

Run with: pytest tests/test_search.py -v
"""

import pytest
import numpy as np
from typing import List, Dict

# Import functions to test
from app.model_loader_improved import (
    detect_query_type,
    calculate_adaptive_topk,
    sanitize_value,
    sanitize_search_results,
)

# ============================================================================
# QUERY TYPE DETECTION TESTS
# ============================================================================

class TestQueryTypeDetection:
    """Test query type detection"""
    
    def test_simple_single_hop_query(self):
        """Test simple single-hop queries"""
        queries = [
            "obat diare",
            "paracetamol",
            "antibiotik",
            "vitamin C",
        ]
        for query in queries:
            result = detect_query_type(query)
            assert result == "single-hop", f"'{query}' should be single-hop"
    
    def test_multi_hop_queries(self):
        """Test multi-hop queries"""
        queries = [
            "obat untuk diare dan demam",
            "paracetamol atau ibuprofen",
            "antibiotik serta vitamin",
            "obat diare dengan probiotik",
        ]
        for query in queries:
            result = detect_query_type(query)
            assert result == "multi-hop", f"'{query}' should be multi-hop"
    
    def test_edge_cases(self):
        """Test edge cases"""
        # Empty query
        assert detect_query_type("") == "single-hop"
        
        # Very long single term
        assert detect_query_type("supercalifragilisticexpialidocious") == "single-hop"
        
        # Numbers
        assert detect_query_type("123") == "single-hop"
    
    def test_case_insensitivity(self):
        """Test that detection is case-insensitive"""
        queries = [
            ("OBAT DIARE", "single-hop"),
            ("Obat Untuk Diare Dan Demam", "multi-hop"),
            ("PaRaCeTaMoL", "single-hop"),
        ]
        for query, expected in queries:
            result = detect_query_type(query)
            assert result == expected


# ============================================================================
# ADAPTIVE TOP-K TESTS
# ============================================================================

class TestAdaptiveTopK:
    """Test adaptive top-k calculation"""
    
    def test_simple_query(self):
        """Test simple query gets higher top-k"""
        topk = calculate_adaptive_topk("obat", per_page=20, page=0)
        assert topk >= 40  # At least 2x per_page
        assert topk <= 1000  # Within max limit
    
    def test_complex_query(self):
        """Test complex query gets lower top-k"""
        complex_query = "obat untuk mengobati diare yang disertai demam dan mual"
        topk = calculate_adaptive_topk(complex_query, per_page=20, page=0)
        assert topk <= 200  # Limited for complex queries
    
    def test_respects_max_topk(self):
        """Test that max_topk is respected"""
        topk = calculate_adaptive_topk(
            "test",
            per_page=1000,
            page=5,
            max_topk=100
        )
        assert topk <= 100
    
    def test_pagination_increases_topk(self):
        """Test that higher page numbers increase top-k"""
        topk_page0 = calculate_adaptive_topk("test", per_page=10, page=0)
        topk_page5 = calculate_adaptive_topk("test", per_page=10, page=5)
        assert topk_page5 >= topk_page0


# ============================================================================
# SANITIZATION TESTS
# ============================================================================

class TestSanitization:
    """Test value sanitization"""
    
    def test_sanitize_none(self):
        """Test None values"""
        assert sanitize_value(None) is None
        assert sanitize_value(None, key='score') == 0.0
    
    def test_sanitize_float(self):
        """Test float values"""
        # Normal float
        assert sanitize_value(0.5) == 0.5
        
        # NaN
        assert sanitize_value(float('nan')) == 0.0
        
        # Infinity
        assert sanitize_value(float('inf')) == 0.0
        assert sanitize_value(float('-inf')) == 0.0
        
        # Numpy float
        assert isinstance(sanitize_value(np.float64(0.5)), float)
    
    def test_sanitize_int(self):
        """Test integer values"""
        assert sanitize_value(42) == 42
        assert sanitize_value(np.int64(42)) == 42
        assert isinstance(sanitize_value(np.int64(42)), int)
    
    def test_sanitize_string(self):
        """Test string sanitization"""
        # Whitespace
        assert sanitize_value("  test  ") == "test"
        
        # Newlines
        assert sanitize_value("line1\nline2") == "line1 line2"
        
        # Tabs
        assert sanitize_value("tab\there") == "tab here"
        
        # Multiple spaces
        assert sanitize_value("too   many    spaces") == "too many spaces"
    
    def test_sanitize_list(self):
        """Test list sanitization"""
        input_list = [1, float('nan'), "test\n", None]
        result = sanitize_value(input_list)
        assert isinstance(result, list)
        assert result[0] == 1
        assert result[1] == 0.0  # NaN converted
        assert result[2] == "test"  # Newline removed
        assert result[3] is None
    
    def test_sanitize_dict(self):
        """Test dictionary sanitization"""
        input_dict = {
            'score': float('nan'),
            'name': "test\n",
            'count': np.int64(42)
        }
        result = sanitize_value(input_dict)
        assert result['score'] == 0.0
        assert result['name'] == "test"
        assert result['count'] == 42
    
    def test_sanitize_search_results(self):
        """Test full search results sanitization"""
        results = [
            {
                'nama': "Obat A",
                'score': float('nan'),
                'deskripsi': "Line 1\nLine 2",
            },
            {
                'nama': "Obat B  ",
                'score': 0.8,
                'count': np.int64(10),
            }
        ]
        
        sanitized = sanitize_search_results(results)
        
        assert len(sanitized) == 2
        assert sanitized[0]['score'] == 0.0
        assert sanitized[0]['deskripsi'] == "Line 1 Line 2"
        assert sanitized[1]['nama'] == "Obat B"
        assert isinstance(sanitized[1]['count'], int)


# ============================================================================
# MOCK SEARCH FUNCTIONS FOR TESTING
# ============================================================================

def mock_search_results(query: str, top_k: int) -> List[Dict]:
    """Mock search function for testing"""
    return [
        {
            'nama': f"Result {i}",
            'score': 1.0 - (i * 0.1),
            'deskripsi': f"Description for result {i}"
        }
        for i in range(min(top_k, 10))
    ]


# ============================================================================
# INTEGRATION TESTS (requires actual models)
# ============================================================================

@pytest.mark.integration
class TestSearchIntegration:
    """Integration tests - requires actual models to be loaded"""
    
    def test_search_returns_results(self):
        """Test that search returns results"""
        # This would require actual model_loader functions
        # Uncomment and modify when models are available
        
        # from app.model_loader import search_tfidf_docs
        # results = search_tfidf_docs("obat", top_k=10)
        # assert len(results) > 0
        # assert all('nama' in r for r in results)
        # assert all('score' in r for r in results)
        pass
    
    def test_search_score_ordering(self):
        """Test that results are ordered by score"""
        # from app.model_loader import search_tfidf_docs
        # results = search_tfidf_docs("obat", top_k=10)
        # scores = [r['score'] for r in results]
        # assert scores == sorted(scores, reverse=True)
        pass
    
    def test_search_handles_special_characters(self):
        """Test handling of special characters"""
        # special_queries = ["obat #test", "query@123", "test\nnewline"]
        # for query in special_queries:
        #     results = search_tfidf_docs(query, top_k=5)
        #     assert isinstance(results, list)
        pass


# ============================================================================
# PARAMETRIZED TESTS
# ============================================================================

@pytest.mark.parametrize("query,expected_type", [
    ("obat", "single-hop"),
    ("paracetamol", "single-hop"),
    ("obat diare", "single-hop"),
    ("obat untuk diare", "multi-hop"),
    ("obat diare dan demam", "multi-hop"),
    ("paracetamol atau ibuprofen untuk demam", "multi-hop"),
])
def test_query_type_parametrized(query, expected_type):
    """Parametrized test for query type detection"""
    result = detect_query_type(query)
    assert result == expected_type


@pytest.mark.parametrize("value,expected", [
    (None, None),
    (0.5, 0.5),
    (float('nan'), 0.0),
    (float('inf'), 0.0),
    (np.float64(0.5), 0.5),
    (42, 42),
    ("test", "test"),
    ("  test  ", "test"),
])
def test_sanitize_value_parametrized(value, expected):
    """Parametrized test for value sanitization"""
    result = sanitize_value(value)
    if expected is None:
        assert result is None
    else:
        assert result == expected


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

@pytest.mark.performance
class TestPerformance:
    """Performance tests"""
    
    def test_sanitization_performance(self):
        """Test sanitization performance on large dataset"""
        import time
        
        # Create large dataset
        large_results = [
            {
                'nama': f"Item {i}",
                'score': 0.5 + (i % 10) * 0.05,
                'deskripsi': f"Description {i}\nWith newline"
            }
            for i in range(1000)
        ]
        
        start = time.time()
        sanitized = sanitize_search_results(large_results)
        elapsed = time.time() - start
        
        assert len(sanitized) == 1000
        assert elapsed < 1.0  # Should complete in less than 1 second
    
    def test_query_type_detection_performance(self):
        """Test query type detection performance"""
        import time
        
        queries = [
            "obat",
            "paracetamol",
            "obat untuk diare dan demam",
            "antibiotik serta vitamin C",
        ] * 250  # 1000 queries
        
        start = time.time()
        for query in queries:
            detect_query_type(query)
        elapsed = time.time() - start
        
        assert elapsed < 0.5  # Should complete in less than 0.5 second


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_search_results():
    """Sample search results for testing"""
    return [
        {
            'nama': "Obat A",
            'score': 0.9,
            'deskripsi': "Description A",
            'jenis_obat': "Tablet"
        },
        {
            'nama': "Obat B",
            'score': 0.7,
            'deskripsi': "Description B",
            'jenis_obat': "Sirup"
        },
        {
            'nama': "Obat C",
            'score': 0.5,
            'deskripsi': "Description C",
            'jenis_obat': "Kapsul"
        }
    ]


@pytest.fixture
def sample_query():
    """Sample query for testing"""
    return "obat diare"


# ============================================================================
# CONFTEST (pytest configuration)
# ============================================================================

def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
