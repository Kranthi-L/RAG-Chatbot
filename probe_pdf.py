# probe_pdf.py
from pypdf import PdfReader
import glob, os

# find all PDFs under data/ (recursively)
files = sorted(glob.glob("data/**/*.pdf", recursive=True))

if not files:
    print("No PDFs found under ./data — check folder structure.")
else:
    for fn in files:
        try:
            r = PdfReader(fn)
            n = len(r.pages)
            sample = (r.pages[min(10, n-1)].extract_text() or "")[:500]
            print(f"\n{fn} → pages={n}\nSample text:\n{sample}\n")
        except Exception as e:
            print(f"\n{fn} → ERROR: {e}\n")
