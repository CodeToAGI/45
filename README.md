# EP45 — Tokenization & Embeddings Explained

**Deep Learning Series · Module 9 (NLP) begins**

How text becomes numbers that keep meaning:
- Tokenization (word / character / subword)
- Byte-Pair Encoding (BPE) with real merge trace
- WordPiece · SentencePiece · byte-level BPE
- Hugging Face AutoTokenizer
- word2vec (skip-gram + negative sampling)
- GloVe + pretrained vectors
- Static vs Contextual embeddings
- `nn.Embedding`
- Real GloVe vectors visualised with PCA & t-SNE

## Files

| File | Description |
|------|-------------|
| `generate_dl_ep45.py` | Full production pipeline (TTS + Manim + merge) |
| `manim_dl_ep45.py` | All Manim scenes (clock-synced to narration) |
| `ep45_embeddings_viz.py` | **Challenge solution** — plot real GloVe embeddings |
| `output/` | Rendered video, thumbnail, chapters, voice files |

## Challenge

1. `pip install gensim scikit-learn matplotlib`
2. Load `glove-wiki-gigaword-50`
3. Pick ~30 words from 3 categories of your choice
4. Project with PCA and t-SNE
5. Find **one analogy that works** and **one that fails**
6. Post your favourite failure in the YouTube comments

Solution: [`ep45_embeddings_viz.py`](ep45_embeddings_viz.py)

## Quick Start (Challenge only)

```bash
pip install gensim scikit-learn matplotlib
python ep45.py
