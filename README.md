# Comparison of TF–IDF Variants (Wizard of Oz)

This repository contains a reproducible pipeline: we find the **longest paragraph** in L. Frank Baum’s *The Wonderful Wizard of Oz* and compute three TF–IDF variants for that paragraph, using the whole book’s paragraphs as the corpus. We then compare the top‑weighted terms across the three variants.

---

## What this does

1) Read the Gutenberg `.txt` of *The Wonderful Wizard of Oz*, trim header/footer, and **split into paragraphs** using the regex `\n[ \n]*\n` (blank-line separation).  
2) Identify the **longest paragraph** and save it to `results/longest_paragraph.txt`.  
3) Build three TF–IDF **variants** over the full paragraph corpus (Exercise 4.1 style), then extract weights only for the **longest paragraph** and report **Top‑20 terms** for each variant:
   - **Option A** — *Length‑normalized TF (L2)* + *Smoothed logarithmic IDF*.  
     `TfidfVectorizer(norm='l2', smooth_idf=True, sublinear_tf=False, stop_words='english')`
   - **Option B** — *log(TF)* + *Smoothed logarithmic IDF*, **no vector normalization**.  
     `TfidfVectorizer(norm=None, smooth_idf=True, sublinear_tf=True, stop_words='english')`
   - **Option C** — *TF relative to the most frequent term* (`f / max_f`) + *IDF normalized by its maximum* (values in [0,1]).

All outputs are written under `results/`.

---

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy pandas scikit-learn

# Download the book (UTF‑8)
curl -L -o wizard_of_oz.txt https://www.gutenberg.org/files/55/55-0.txt

# Run
python main.py
```

---

## Results 

**Corpus stats**  
- Paragraphs: **1140**  
- **Longest paragraph**: id **505**, ~**208 tokens**, **1061 chars**  
  → saved as `results/longest_paragraph.txt`

**Top‑20 terms for each TF–IDF option**  
*(weights are comparable **within** each option; lists are ordered by descending weight)*

### Option A — L2‑normalized TF + Smoothed log IDF
```
bed 0.317008
room 0.253183
rooms 0.247217
night 0.205258
just 0.177461
like 0.165049
sense 0.143950
remained 0.143950
moving 0.143950
wasted 0.143950
cat 0.143950
worry 0.137983
lie 0.137983
doorway 0.137983
spider 0.133356
remembered 0.133356
pleasant 0.133356
shut 0.129575
spot 0.129575
sprang 0.129575
```

### Option B — log(TF) + Smoothed log IDF (no vector norm)
```
bed 10.692894
rooms 10.091528
room 8.540027
night 8.378738
just 7.244063
sense 6.941048
remained 6.941048
moving 6.941048
wasted 6.941048
cat 6.941048
like 6.737401
lie 6.653366
worry 6.653366
doorway 6.653366
remembered 6.430222
pleasant 6.430222
spider 6.430222
sprang 6.247901
shut 6.247901
spot 6.247901
```

### Option C — Relative‑to‑max TF + IDF normalized by max
```
bed 0.734071
room 0.586276
rooms 0.572461
night 0.475300
just 0.410933
like 0.382192
sense 0.333333
remained 0.333333
moving 0.333333
wasted 0.333333
cat 0.333333
worry 0.319518
lie 0.319518
doorway 0.319518
spider 0.308802
remembered 0.308802
pleasant 0.308802
shut 0.300046
spot 0.300046
sprang 0.300046
```

---

## Discussion (what the comparison shows)

- The same semantic theme emerges across all three lists: a **night‑time room/bed** scene; terms like *bed, room(s), night,* etc., dominate.  
- **Option A** (L2 normalization) down‑weights long paragraphs overall; weights are balanced and highlight distinctive content words across the corpus.  
- **Option B** (log‑TF, no vector norm) **reduces burstiness** within the paragraph; extremely frequent words don’t explode in weight. Rankings are often close to A but slightly favor mid‑frequency topical terms.
- **Option C** (relative‑to‑max TF with IDF ∈ [0,1]) tracks the **paragraph‑internal relative frequencies** most directly; because IDF is compressed to [0,1], the ranking is influenced more by TF than by corpus rarity. This can be preferable if your goal is *within‑paragraph* salience rather than corpus distinctiveness.
- If a crisper IDF contrast is desired for **C**, remove the max‑normalization (use the smoothed IDF directly) or try a BM25‑style IDF `log((N-df+0.5)/(df+0.5))`.

---

## Files produced

```
results/
├── longest_paragraph.txt           # the full text of the longest paragraph
├── optionA_top20.txt               # Option A top-20 terms
├── optionB_top20.txt               # Option B top-20 terms
├── optionC_top20.txt               # Option C top-20 terms
├── top20_comparison.csv            # (optional) merged table for side-by-side analysis
└── summary.txt                     # paragraph id / token & char counts
```

> `.venv/` and editor/system files are ignored via `.gitignore`; keep `results/` in the repo so the grader can see outputs without re-running.

---

## Reproducibility notes

- We use `stop_words='english'`, `max_df=0.9`, `min_df=2` to drop extreme terms.  
- Randomness does not affect 4.2 (no clustering), so results should be deterministic given the same text file.  
- Text source: Project Gutenberg (eBook #55).

---

## Citation

- L. Frank Baum, *The Wonderful Wizard of Oz*, Project Gutenberg, eBook #55 (public domain).  
- scikit‑learn: Pedregosa et al., JMLR 12, 2011.
