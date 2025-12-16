# 💊 MedicIR - Drug Information Retrieval System

Sistem Pencarian Obat Cerdas berbasis AI menggunakan teknologi Semantic Search (MiniLM) dan TF-IDF.

![MedicIR Banner](app/static/medicir.png)

## 🎯 Fitur

- ✅ **Pencarian Semantik**: Menggunakan MiniLM untuk memahami konteks query
- ✅ **Pencarian Keyword**: TF-IDF untuk pencarian berbasis kata kunci  
- ✅ **Hybrid Search**: Kombinasi terbaik dari kedua metode
- ✅ **Web Interface**: UI modern dan user-friendly
- ✅ **API Documentation**: Swagger UI dan ReDoc
- ✅ **Dataset**: 3,137 obat dari sumber terpercaya

## 🖼️ Screenshots

### Landing Page
![Landing Page](app/static/Group_7.png)

### Search Interface
Aplikasi memiliki interface pencarian yang intuitif dengan hasil yang akurat dan relevan.

## 🔧 Teknologi

- **Backend**: FastAPI, Python 3.10+
- **ML Model**: sentence-transformers/all-MiniLM-L6-v2
- **Search**: TF-IDF, BM25, Hybrid Search
- **Frontend**: HTML, CSS, JavaScript
- **Dataset**: 3,137 obat dengan informasi lengkap

## 📊 Dataset

Dataset berisi informasi lengkap obat:
- Nama obat
- Tipe (tablet, sirup, kapsul, dll)
- Komposisi
- Indikasi umum
- Nomor registrasi

**File**: `final_clean_data_20112024_halodoc_based.csv`

## 📦 Struktur Project
```
UASNLP/
├── app/
│   ├── __init__.py
│   ├── main_hybrid.py              # FastAPI application
│   ├── model_loader_hybrid.py      # Model loader & search logic
│   └── static/                     # Static assets (images)
│       ├── medicir.png
│       ├── Group_7.png
│       └── 370.jpg
├── model/
│   └── semantic_model_export/
│       └── minilm_model/           # MiniLM model (auto-download)
├── index.html                      # Frontend interface
├── requirements.txt                # Python dependencies
├── generate_embeddings.py          # Generate pre-computed embeddings
├── generate_tfidf.py               # Generate TF-IDF matrix
├── final_clean_data_20112024_halodoc_based.csv  # Dataset
├── doc_embeddings_minilm.npy       # Pre-computed embeddings
├── tfidf_matrix.npz                # Pre-computed TF-IDF matrix
├── tfidf_vectorizer.joblib         # TF-IDF vectorizer
└── README.md                       # This file
```

## 🚀 Cara Install & Menjalankan

### **Prasyarat**

- Python 3.10 atau lebih baru
- pip
- 4GB RAM minimum
- 2GB storage untuk model dan dependencies

### **Langkah 1: Clone Repository**
```bash
git clone https://github.com/nauvalsyhfr/UASNLP.git
cd UASNLP
```

### **Langkah 2: Buat Virtual Environment**

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### **Langkah 3: Install Dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Catatan untuk Mac M1/M2**: Jika ada error saat install PyTorch:
```bash
pip install torch torchvision torchaudio
```

### **Langkah 4: Generate Pre-computed Files (Opsional)**

**JIKA file pre-computed sudah ada di repository** (sudah di-include saat clone), **LEWATI langkah ini!**

**JIKA file pre-computed TIDAK ada**, generate dengan:
```bash
# Generate embeddings (~5 menit)
python generate_embeddings.py

# Generate TF-IDF matrix (~1 menit)  
python generate_tfidf.py
```

File yang dihasilkan:
- ✅ `doc_embeddings_minilm.npy` (~5 MB)
- ✅ `tfidf_matrix.npz` (~3 MB)
- ✅ `tfidf_vectorizer.joblib` (~500 KB)

### **Langkah 5: Jalankan Server**
```bash
python -m uvicorn app.main_hybrid:app --reload --host 0.0.0.0 --port 8000
```

**Atau jalankan langsung:**
```bash
cd app
python main_hybrid.py
```

### **Langkah 6: Buka di Browser**
```
http://localhost:8000
```

**Endpoints tersedia:**
- 🌐 Web Interface: `http://localhost:8000`
- 📚 API Docs (Swagger): `http://localhost:8000/docs`
- 📖 API Docs (ReDoc): `http://localhost:8000/redoc`
- ❤️ Health Check: `http://localhost:8000/health`

## 🔍 Cara Menggunakan

### **Via Web Interface**

1. Buka `http://localhost:8000`
2. Klik "Temukan Obat Sekarang"
3. Masukkan query (contoh: "obat sakit kepala", "paracetamol")
4. Klik "Cari"
5. Lihat hasil pencarian

### **Via API**

**Search endpoint:**
```bash
curl "http://localhost:8000/search?query=obat%20diare&method=hybrid&top_k=10"
```

**Response:**
```json
{
  "query": "obat diare",
  "method_used": "hybrid_minilm_tfidf",
  "total_results": 10,
  "results": [
    {
      "nama": "Diapet NR",
      "tipe": "tablet",
      "komposisi": "Attapulgite 600 mg",
      "indikasi_umum": "Diare non spesifik",
      "score": 0.892
    }
  ]
}
```

## 🧪 Testing

### **Test Search Functions**
```bash
# Test keywords loading
python test_keywords.py

# Test file locations
python check_files.py
```

### **Manual Test via Python**
```python
from app.model_loader_hybrid import smart_search

# Test search
results, method = smart_search("obat diare", top_k=5, method="hybrid")
print(f"Found {len(results)} results using {method}")
for r in results:
    print(f"- {r['nama']} (score: {r['score']:.3f})")
```

## 📖 API Documentation

### **Main Endpoints**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/health` | GET | Health check |
| `/search` | GET | Search medicines |
| `/doc/{doc_id}` | GET | Get specific document |
| `/stats` | GET | System statistics |
| `/methods` | GET | Available search methods |

### **Search Parameters**

- `query` (required): Search query string
- `method` (optional): `hybrid` (default), `minilm`, `tfidf`
- `top_k` (optional): Number of results (default: 20)
- `page` (optional): Page number (default: 0)
- `per_page` (optional): Results per page (default: 20)

## 🐛 Troubleshooting

### **Error: Module not found**

**Solusi**: Pastikan virtual environment aktif dan dependencies terinstall:
```bash
source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
```

### **Error: File not found (CSV atau .npy)**

**Solusi**: Pastikan Anda di root directory project:
```bash
pwd  # Cek current directory
ls -la  # Cek file yang ada
```

### **Error: Port 8000 already in use**

**Solusi**: Gunakan port lain:
```bash
python -m uvicorn app.main_hybrid:app --port 8001
```

### **Search sangat lambat**

**Solusi**: Generate pre-computed files jika belum ada:
```bash
python generate_embeddings.py
python generate_tfidf.py
```

### **Out of Memory**

**Solusi**: Tutup aplikasi lain, atau kurangi batch size di `generate_embeddings.py`:
```python
batch_size=16  # Kurangi dari 32
```

## 📚 Dependencies

Main dependencies (lihat `requirements.txt` untuk detail):

- `fastapi>=0.104.0` - Web framework
- `uvicorn>=0.24.0` - ASGI server
- `sentence-transformers>=2.2.0` - Semantic search
- `torch>=2.0.0` - Deep learning framework
- `scikit-learn>=1.3.0` - TF-IDF & ML utilities
- `pandas>=2.0.0` - Data processing
- `numpy>=1.24.0` - Numerical computing

## 🤝 Contributing

Kontribusi sangat diterima! Untuk berkontribusi:

1. Fork repository ini
2. Buat branch baru (`git checkout -b feature/AmazingFeature`)
3. Commit perubahan (`git commit -m 'Add some AmazingFeature'`)
4. Push ke branch (`git push origin feature/AmazingFeature`)
5. Buat Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 👥 Tim Pengembang

**Kelompok B NLP SD-A1**  
Teknologi Sains Data  
Fakultas Teknologi Maju dan Multidisiplin  
Universitas Airlangga

## 📧 Kontak

- GitHub: [@nauvalsyhfr](https://github.com/nauvalsyhfr)
- Repository: [UASNLP](https://github.com/nauvalsyhfr/UASNLP)

## 🙏 Acknowledgments

- Dataset dari berbagai sumber medis terpercaya
- Model: Sentence Transformers (HuggingFace)
- Framework: FastAPI
- Inspiration: Modern medical information retrieval systems

---

⭐ Jika project ini membantu, jangan lupa kasih Star di GitHub!

**Made with ❤️ by Kelompok B NLP SD-A1**
