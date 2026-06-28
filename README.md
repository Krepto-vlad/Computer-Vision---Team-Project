# Low-Light Enhancement & Detection

Classical **OpenCV** pipeline for dark images:

**Enhance → Segment → Clean → Detect → Decide**

After low-light enhancement, the system detects **salient regions** (bounding boxes) and outputs an automatic scene decision (`OK` / `ALERT` / `NO_OBJECTS`).

---

## Setup (once)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install -e ".[dev]"
```

---

## Run in one line

Batch demo on real LOL images — saves all 6 required outputs per image:

```bash
python scripts/run_showcase.py --n 6
```

Results: `outputs/showcase/` (PNG per stage, `summary.csv`, `demo_montage.png`, `PRESENTATION.md`).

Optional GT metrics on synthetic data (experiments only):

```bash
python scripts/run_eval.py --synthetic --n 12
```

---

## Run via notebook (presentation / Colab-style)

**Local Jupyter:**

```bash
jupyter notebook
```

Open `notebooks/presentation_demo.ipynb` → **Kernel → Restart & Run All**.

**Google Colab:**

```python
!git clone <your-repo-url> TEAMproject
%cd TEAMproject
!pip install -e ".[dev]"
```

Then open `notebooks/presentation_demo.ipynb` and **Run All**.

| Step | Content |
|------|---------|
| 0 | Setup, imports |
| 1 | LOL dataset (input vs reference) |
| 2 | Enhancement methods |
| 3 | Full 5-stage pipeline + strip |
| 4 | Batch run, save to `outputs/showcase/` |
| 5 | Feature detectors (optional, Lab 08) |


---

## Project structure

```
TEAMproject/
├── notebooks/presentation_demo.ipynb   # demo 
├── scripts/
│   ├── run_showcase.py                 # one-line batch demo
│   └── run_eval.py                     # GT metrics (synthetic)
├── src/lowlight_cv/                    # pipeline code
├── data/custom/                        # custom low-light photos
└── outputs/showcase/                   # generated results
```


Team roles: `docs/CONTRIBUTION_TEMPLATE.md`
