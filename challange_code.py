"""
EP45 Challenge Solution — Real GloVe Embeddings in 2D
CodeToAGI · Deep Learning Series

Requirements:
    pip install gensim scikit-learn matplotlib

Run:
    python ep45_embeddings_viz.py
"""

import gensim.downloader as api
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

print("Loading glove-wiki-gigaword-50 ... (first run downloads ~66 MB)")
glove = api.load("glove-wiki-gigaword-50")

# ── Choose your own 3 categories (example) ──────────────────────────────────
groups = {
    "animals":   ["cat", "dog", "horse", "lion", "rabbit", "tiger", "wolf", "bear"],
    "food":      ["bread", "cheese", "pizza", "rice", "apple", "cake", "soup", "meat"],
    "royalty":   ["king", "queen", "prince", "princess", "throne", "crown", "royal", "palace"],
}

words = [w for g in groups.values() for w in g]
# Keep only words that actually exist in the model
words = [w for w in words if w in glove]
print(f"Using {len(words)} words")

X = np.array([glove[w] for w in words])

# ── PCA ─────────────────────────────────────────────────────────────────────
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

# ── t-SNE ───────────────────────────────────────────────────────────────────
tsne = TSNE(n_components=2, perplexity=8, random_state=3, init="pca", learning_rate="auto")
X_tsne = tsne.fit_transform(X)

# ── Plot side-by-side ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
colors = {"animals": "#50fa7b", "food": "#ffb86c", "royalty": "#ff79c6"}

for ax, coords, title in zip(axes, [X_pca, X_tsne], ["PCA", "t-SNE"]):
    for group, members in groups.items():
        idxs = [i for i, w in enumerate(words) if w in members]
        ax.scatter(coords[idxs, 0], coords[idxs, 1],
                   c=colors[group], s=80, label=group, edgecolors="k", linewidths=0.5)
        for i in idxs:
            ax.annotate(words[i], (coords[i, 0], coords[i, 1]),
                        fontsize=9, ha="left", va="bottom")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle("Real GloVe-50 embeddings · EP45 Challenge", fontsize=16)
plt.tight_layout()
plt.savefig("ep45_embeddings_map.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → ep45_embeddings_map.png")

# ── Analogies ───────────────────────────────────────────────────────────────
print("\n── Working analogy ──")
print(glove.most_similar(positive=["king", "woman"], negative=["man"], topn=3))

print("\n── Try to find a failure (example) ──")
# Classic failure cases people discover:
#   "doctor" - "man" + "woman" sometimes still returns male-biased results
#   country/capital analogies with low-resource languages
#   abstract concepts (love - hate + joy, etc.)
print(glove.most_similar(positive=["doctor", "woman"], negative=["man"], topn=5))
