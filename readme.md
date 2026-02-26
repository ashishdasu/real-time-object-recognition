# Real-time 2D Object Recognition
**CS5330 — Ashish Dasu**

Real-time object recognition system written in C++. A downward-facing webcam captures objects on a white surface and runs them through a five-stage pipeline: dynamic thresholding, morphological filtering, connected components segmentation, feature extraction, and KNN classification. Also includes a one-shot ResNet18 embedding classifier.

---

## Demo Videos

| Clip | Link |
|------|------|
| Live pipeline + debug view | [link](https://drive.google.com/file/d/1nbjPA6E23M6GQAh8aW5A1zoKpbvML-l8/view?usp=share_link) |
| Live KNN classification (10 objects) | [link](https://drive.google.com/file/d/1rxcIyaawi36ok8Hd1yHGzgR94PIf1Sof/view?usp=share_link) |
| Auto-learn unknown object | [link](https://drive.google.com/file/d/1MqmiSbiGZwd1yJxXUyDKE4soKkWWLjJp/view?usp=share_link) |
| Embedding classification | [link](https://drive.google.com/file/d/1f16icXiy23oFwd52yLFvmTYUvZu8YMm5/view?usp=share_link) |
| Interactive evaluation + confusion matrix | [link](https://drive.google.com/file/d/1nVoIdag5mc5G_Ad14SyGH247R5LPrPap/view?usp=share_link) |
| Batch pipeline | [link](https://drive.google.com/file/d/1au6Wgvrowp22qc3uYCz2p0IYCQsS8pJd/view?usp=share_link) |

---

## Pipeline

1. **Thresholding** — ISODATA (iterative K-means, K=2) with HSV saturation preprocessing to handle brightly colored objects
2. **Morphology** — Closing (dilate → erode) with a 7×7 kernel to fill interior holes from logos and button arrays
3. **Segmentation** — Connected components with size and border filters, top-N regions by area
4. **Features** — 6D vector: 4 log-scaled Hu moments, percent filled, aspect ratio. OBB computed analytically from covariance eigenvalues
5. **Classification** — KNN (k=3) with scaled Euclidean distance and unknown-object rejection threshold

Both thresholding and morphology are implemented from scratch with no OpenCV morphology or threshold functions.

---

## Build

```bash
mkdir -p build && cd build && cmake .. && make
```

Requires OpenCV 4. Built with Apple Clang 17, CMake 4.2.1, macOS 26 (Tahoe).

---

## Running

**Always run from the project root.**

```bash
# Webcam (default device)
./build/objrecog

# Specific camera index
./build/objrecog 1

# Static image (freezes on last frame)
./build/objrecog path/to/image.jpg
```

### Key bindings

| Key | Action |
|-----|--------|
| `d` | Cycle debug view: Original → Threshold → Morphology → Regions → Features |
| `c` | Toggle KNN classification overlay |
| `m` | Toggle embedding classifier overlay |
| `n` | Label current object and save to feature DB |
| `l` | Reload feature DB from disk |
| `a` | Auto-learn: if object is Unknown, prompt for label and add immediately |
| `b` | Save ResNet18 embedding for current object |
| `e` | Record evaluation sample (true label vs predicted) |
| `p` | Print confusion matrix to terminal |
| `s` | Save timestamped screenshots of both windows |
| `q` | Quit |

---

## Batch Commands

```bash
# Build feature DB from images in data/
./build/objrecog --train-features

# Generate all 5 pipeline-stage images for every object
./build/objrecog --batch

# Run full evaluation and save confusion matrix PNGs
./build/objrecog --evaluate

# Retrain embedding DB (one shot per class from test/*_1.jpg)
./build/objrecog --train-embeddings
```

---

## Embedding Visualization

Requires Python 3 with `numpy`, `matplotlib`, `opencv-python`:

```bash
python3 plot_embeddings.py   # PCA projection of embeddings
python3 plot_confusion.py    # Polished confusion matrix PNGs
```

---

## Results

- **KNN classifier**: 17/30 (56.7%) on 10-class held-out test set
- **ResNet18 embedding**: 12/20 (60.0%) one-shot, 1 training example per class
- Both run at full webcam framerate (640×480, 30fps)
