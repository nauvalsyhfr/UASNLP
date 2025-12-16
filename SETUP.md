## Quick Setup Guide

Panduan singkat untuk menjalankan MedicIR di laptop baru.

```bash
# 1. Clone
git clone https://github.com/nauvalsyhfr/UASNLP.git
cd UASNLP

# 2. Setup Virtual Environment
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate  # Windows

# 3. Install Dependencies
pip install -r requirements.txt

# 4. Run Server
python -m uvicorn app.main_hybrid:app --reload --port 8000

# 5. Open Browser
# http://localhost:8000
```

## Checklist

- [ ] Python 3.10+ installed
- [ ] Git installed
- [ ] 4GB RAM available
- [ ] 2GB disk space
- [ ] Internet connection (first run)

## Common Issues

**Import Error?**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Port in use?**
```bash
python -m uvicorn app.main_hybrid:app --port 8001
```

**Slow search?**
```bash
python generate_embeddings.py
python generate_tfidf.py
```

Check `README.md` for detailed documentation.
