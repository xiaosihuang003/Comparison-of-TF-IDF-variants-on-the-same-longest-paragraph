from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# ---------- Read and split into paragraphs ----------
text_path = Path("wizard_of_oz.txt")
raw = text_path.read_text(encoding="utf-8", errors="ignore")
U = raw.upper()
start = 0
for m in ("*** START OF THE PROJECT GUTENBERG EBOOK","*** START OF THIS PROJECT GUTENBERG EBOOK"):
    i = U.find(m)
    if i!=-1:
        j = U.find("\n", i)
        start = j if j!=-1 else i
        break
end = len(raw)
for m in ("*** END OF THE PROJECT GUTENBERG EBOOK","*** END OF THIS PROJECT GUTENBERG EBOOK"):
    i = U.find(m)
    if i!=-1:
        end = i
        break
content = raw[start:end]
paragraphs = [p.strip() for p in re.split(r"\n[ \n]*\n", content) if p.strip()]

# tokenizer (lowercase + letters+hyphens only)

token_pat = re.compile(r"[a-zA-Z][a-zA-Z\-']+")
def tokenize(s): return token_pat.findall(s.lower())

print(f"[INFO] Paragraphs: {len(paragraphs)}")

# ---------- (a) Find the longest paragraph ----------
lens_tokens = [len(tokenize(p)) for p in paragraphs]
longest_idx = int(np.argmax(lens_tokens))
longest_para = paragraphs[longest_idx]
longest_chars = len(longest_para)
longest_tokens = lens_tokens[longest_idx]
Path("results").mkdir(exist_ok=True)
(Path("results")/"longest_paragraph.txt").write_text(longest_para, encoding="utf-8")
print(f"[INFO] Longest paragraph id={longest_idx}, tokens={longest_tokens}, chars={longest_chars}")
print("[INFO] Written results/longest_paragraph.txt")

# ---------- Three TF-IDF variants, compute weights only for "longest paragraph" and take Top-20 ----------
# First fit on "all paragraphs" to obtain global vocabulary and document frequency 
# Unified preprocessing configuration: lowercase, English stop words, filter extremely common/rare words

common_kwargs = dict(lowercase=True, stop_words="english", max_df=0.9, min_df=2)

# === Scheme A: Length-normalized TF + Smoothed log IDF ===
# Corresponding sklearn: norm='l2', smooth_idf=True, sublinear_tf=False
tfidf_A = TfidfVectorizer(norm='l2', smooth_idf=True, sublinear_tf=False, **common_kwargs)
XA = tfidf_A.fit_transform(paragraphs)   # [n_paras, n_terms]
termsA = np.array(tfidf_A.get_feature_names_out())
rowA = XA[longest_idx].toarray().ravel()
topA_idx = np.argsort(rowA)[-20:][::-1]
A_pairs = list(zip(termsA[topA_idx], rowA[topA_idx]))

# === Scheme B: log(count) TF + Smoothed log IDF (no vector normalization) ===
# Corresponding sklearn: sublinear_tf=True, norm=None, smooth_idf=True
tfidf_B = TfidfVectorizer(norm=None, smooth_idf=True, sublinear_tf=True, **common_kwargs)
XB = tfidf_B.fit_transform(paragraphs)
termsB = np.array(tfidf_B.get_feature_names_out())
rowB = XB[longest_idx].toarray().ravel()
topB_idx = np.argsort(rowB)[-20:][::-1]
B_pairs = list(zip(termsB[topB_idx], rowB[topB_idx]))

# === Scheme C: Augmented TF (relative to most frequent term) + IDF normalized to [0,1] "proportional version" ===
# Explanation: For "Count relative to most frequent term (TF)" I use the common
# augmented frequency: tf = 0.5 + 0.5*(f/max_f) (when f>0), otherwise 0.
# "Version proportional to most common term (IDF)" is interpreted as normalizing smoothed IDF by max value to [0,1],
cv = CountVectorizer(**common_kwargs)
Xc = cv.fit_transform(paragraphs)   # counts
vocab = np.array(cv.get_feature_names_out())
# frequency
df = (Xc > 0).sum(axis=0).A1
N = Xc.shape[0]
# IDF
idf_smooth = np.log((1 + N) / (1 + df)) + 1.0
idf_prop = idf_smooth / idf_smooth.max()

# Take the word frequency of the "longest paragraph"
counts = Xc[longest_idx].toarray().ravel()
if counts.max() == 0:
    aug_tf = np.zeros_like(counts, dtype=float)
else:
    aug_tf = np.zeros_like(counts, dtype=float)
    nz = counts > 0
    aug_tf[nz] = counts[nz] / counts.max()

weightsC = aug_tf * idf_prop
topC_idx = np.argsort(weightsC)[-20:][::-1]
C_pairs = list(zip(vocab[topC_idx], weightsC[topC_idx]))

# ---------- Output results  ----------
def write_pairs(path, pairs, title):
    lines = [f"{w}\t{val:.6f}" for w, val in pairs]
    Path(path).write_text(title + "\n" + "\n".join(lines) + "\n", encoding="utf-8")

write_pairs("results/optionA_top20.txt", A_pairs, "Option A: L2-normalized TF + Smoothed log IDF")
write_pairs("results/optionB_top20.txt", B_pairs, "Option B: log(TF) + Smoothed log IDF (no vector norm)")
write_pairs("results/optionC_top20.txt", C_pairs, "Option C: Augmented TF + IDF normalized by max")

summary = f"""Longest paragraph
- id: {longest_idx}
- tokens: {longest_tokens}
- chars: {longest_chars}

Files:
- results/longest_paragraph.txt
- results/optionA_top20.txt
- results/optionB_top20.txt
- results/optionC_top20.txt
"""
(Path("results")/"summary.txt").write_text(summary, encoding="utf-8")

print("[INFO] Wrote:")
print("  results/longest_paragraph.txt")
print("  results/optionA_top20.txt")
print("  results/optionB_top20.txt")
print("  results/optionC_top20.txt")
print("  results/summary.txt")


