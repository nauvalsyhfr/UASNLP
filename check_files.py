#!/usr/bin/env python3
"""
Diagnostic script untuk cek lokasi index.html
Run: python check_files.py
"""

import os
import sys

print("=" * 70)
print("🔍 MEDICIR FILE LOCATION DIAGNOSTIC")
print("=" * 70)

# Get current working directory
cwd = os.getcwd()
print(f"\n1. Current Working Directory:")
print(f"   {cwd}")

# Check if we're in the right place
print(f"\n2. Contents of current directory:")
try:
    files = os.listdir(cwd)
    print(f"   Total files: {len(files)}")
    
    # Check for key files
    key_files = [
        'index.html',
        'index_integrated.html',
        'final_clean_data_20112024_halodoc_based.csv',
        'main_hybrid.py',
        'generate_embeddings.py',
        'generate_tfidf.py'
    ]
    
    print(f"\n3. Looking for key files:")
    for filename in key_files:
        if filename in files:
            filepath = os.path.join(cwd, filename)
            size = os.path.getsize(filepath)
            print(f"   ✅ {filename:45s} ({size:>10,} bytes)")
        else:
            print(f"   ❌ {filename:45s} NOT FOUND")
    
    # Check for app directory
    print(f"\n4. Checking app/ directory:")
    app_dir = os.path.join(cwd, 'app')
    if os.path.isdir(app_dir):
        print(f"   ✅ app/ directory exists")
        app_files = os.listdir(app_dir)
        app_key_files = ['main_hybrid.py', 'model_loader_hybrid.py', '__init__.py']
        for filename in app_key_files:
            if filename in app_files:
                filepath = os.path.join(app_dir, filename)
                size = os.path.getsize(filepath)
                print(f"   ✅ app/{filename:40s} ({size:>10,} bytes)")
            else:
                print(f"   ❌ app/{filename:40s} NOT FOUND")
    else:
        print(f"   ❌ app/ directory NOT FOUND")
    
    # List all HTML files
    print(f"\n5. All HTML files in current directory:")
    html_files = [f for f in files if f.endswith('.html')]
    if html_files:
        for f in html_files:
            filepath = os.path.join(cwd, f)
            size = os.path.getsize(filepath)
            print(f"   - {f:50s} ({size:>10,} bytes)")
    else:
        print(f"   ❌ No .html files found!")
    
    # Calculate paths like main_hybrid.py does
    print(f"\n6. Path calculation (simulating app/main_hybrid.py):")
    
    # Simulate being in app/main_hybrid.py
    app_main_path = os.path.join(cwd, 'app', 'main_hybrid.py')
    if os.path.exists(app_main_path):
        current_dir = os.path.dirname(app_main_path)
        parent_dir = os.path.dirname(current_dir)
        expected_html = os.path.join(parent_dir, 'index.html')
        
        print(f"   If running from: {app_main_path}")
        print(f"   Current dir:     {current_dir}")
        print(f"   Parent dir:      {parent_dir}")
        print(f"   Expected HTML:   {expected_html}")
        print(f"   File exists:     {os.path.exists(expected_html)}")
        
        if not os.path.exists(expected_html):
            print(f"\n   ❌ PROBLEM FOUND!")
            print(f"   index.html should be at: {expected_html}")
            print(f"   But file is not there!")
    else:
        print(f"   ❌ app/main_hybrid.py not found at: {app_main_path}")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("💡 RECOMMENDATIONS:")
print("=" * 70)

# Give recommendations
if 'index_integrated.html' in files and 'index.html' not in files:
    print("\n✅ Action Required:")
    print("   You have 'index_integrated.html' but not 'index.html'")
    print("   Run this command:")
    print(f"   cd {cwd}")
    print("   mv index_integrated.html index.html")
    
elif 'index.html' not in files:
    print("\n❌ Critical Issue:")
    print("   index.html is missing!")
    print("   Please download index_integrated.html and rename it to index.html")
    print(f"   Place it in: {cwd}/")
    
else:
    print("\n✅ index.html found!")
    print("   File location looks correct")
    print("   If still getting 404, try:")
    print("   1. Restart the server completely")
    print("   2. Clear browser cache")
    print("   3. Try accessing http://127.0.0.1:8000")

print("\n" + "=" * 70)
