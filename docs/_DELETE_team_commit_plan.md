# TEMP — delete before final submission

> Internal guide: split work between 2 members and replay commits on `main`  
> so `git log` shows who did what. **Remove this file from the repo before defense.**

---

## 1. Suggested role split (matches course pipeline)

| Member | Stages | Modules / files | Presentation section |
|--------|--------|-----------------|----------------------|
| **Person A** | Enhance + data + metrics | `enhancement/`, `metrics/`, `data/`, `config.py` (thresholds), `run_eval.py` | Problem, dataset LOL, enhancement methods, auto-enhancer |
| **Person B** | Segment → Clean → Detect → Decide + demo | `segmentation/`, `morphology/`, `detection/`, `decision/`, `evaluation/`, `pipeline/`, `io/`, `reporting/`, `utils/`, `scripts/`, `notebooks/` | Segmentation, morphology, detection, decision, live demo |

**Shared (mention both in contribution PDF):** `pipeline/lowlight.py` integration, README, showcase outputs.

Fill names in `docs/CONTRIBUTION_TEMPLATE.md` (recreate if missing) — must match git author names.

---

## 2. Git authors (each person on their machine)

Before committing, each member sets **their** identity (once per session):

```bash
git config user.name "First Last"
git config user.email "email@used-on-github.com"
```

Check:

```bash
git config user.name
git config user.email
```

Use the **same email as on GitHub** — then commits link to profiles on github.com.

---

## 3. Strategy: replay commits on `main`

Current code may already exist in one blob. To show history:

1. Save a backup branch: `git branch backup/full-project`
2. Checkout `main` and reset to `init` (or empty first commit)
3. Add files in **small commits** in the order below
4. Each commit = one person runs `git add` + `git commit` with their git config
5. Push: `git push origin main`

If `main` already has history you cannot rewrite: use `feature/member-a` + `feature/member-b` branches, merge with **Merge commit** (not squash), so both authors appear.

---

## 4. Commit plan (10 commits → `main`)

Execute **top to bottom**. Person column = who runs `git commit`.

### Commit 1 — Person A — project scaffold

```bash
git add pyproject.toml requirements.txt .gitignore data/.gitkeep data/custom/.gitkeep data/custom/README.md outputs/.gitkeep
git add src/lowlight_cv/__init__.py src/lowlight_cv/config.py
git commit -m "chore: add project scaffold and package layout"
```

### Commit 2 — Person A — data loading

```bash
git add src/lowlight_cv/data/
git commit -m "feat(data): add LOL download and real image loader"
```

### Commit 3 — Person A — enhancement stage

```bash
git add src/lowlight_cv/enhancement/
git commit -m "feat(enhance): add gamma, CLAHE, and Retinex methods"
```

### Commit 4 — Person A — enhancement metrics

```bash
git add src/lowlight_cv/metrics/
git commit -m "feat(metrics): add noise-aware enhancement metrics and quality score"
```

---

### Commit 5 — Person B — segmentation

```bash
git add src/lowlight_cv/segmentation/
git commit -m "feat(segment): add Otsu, adaptive threshold, and GrabCut"
```

### Commit 6 — Person B — morphology + detection

```bash
git add src/lowlight_cv/morphology/ src/lowlight_cv/detection/
git commit -m "feat(detect): add mask cleanup and contour-based object detection"
```

### Commit 7 — Person B — decision + evaluation

```bash
git add src/lowlight_cv/decision/ src/lowlight_cv/evaluation/
git commit -m "feat(decide): add scene decision logic and GT evaluation metrics"
```

### Commit 8 — Person B — pipeline integration

```bash
git add src/lowlight_cv/pipeline/ src/lowlight_cv/io/ src/lowlight_cv/utils/ src/lowlight_cv/reporting/
git commit -m "feat(pipeline): wire five-stage pipeline and export outputs"
```

### Commit 9 — Person B — CLI + notebook demo

```bash
git add scripts/ notebooks/presentation_demo.ipynb
git commit -m "feat(demo): add showcase script and presentation notebook"
```

### Commit 10 — both (either author) — docs

```bash
git add README.md docs/
git commit -m "docs: add README, presentation guide, and contribution template"
```

Optional 11th — Person A:

```bash
git add scripts/run_eval.py
git commit -m "feat(eval): add synthetic GT evaluation script"
```

---

## 5. One-shot replay (if everything is already on disk)

From project root, on branch `main`, after `git reset --soft init` (⚠️ only if safe):

```bash
git checkout main
git branch backup/before-split
# Option A: soft reset to replay commits
git reset --soft 412255c   # your init commit hash
# Then run commits 1–10 above one by one
```

**Option B — no reset:** stay on current branch, split **uncommitted** changes:

```bash
git status
# For each commit block above: git add <files> && git commit -m "..."
# Only add files listed for that commit; repeat until clean working tree
```

---

## 6. Push to `main`

```bash
git checkout main
git merge cbook          # if final code lives on cbook
# OR replay commits directly on main (section 5)

git log --oneline --stat -10
git push origin main
```

If remote `main` must be rewritten (only if team agrees):

```bash
git push --force-with-lease origin main
```

Use force push **only** on a team repo you control, never on shared main without agreement.

---

## 7. Verify before defense

```bash
git log --format="%h %an <%ae> %s" -15
git shortlog -sn
```

On GitHub: **Insights → Contributors** — both members should appear.

During presentation each person presents **their** commits/modules from the table in section 1.

---

## 8. Presentation ↔ commits mapping

| Slide | Who talks | Show in repo |
|-------|-----------|--------------|
| Enhancement | A | `enhancement/methods.py`, metrics |
| Segmentation + morphology | B | `segmentation/`, `morphology/` |
| Detection + features | B | `detection/features.py` |
| Decision | B | `decision/scene.py` |
| Live demo | B (or both) | `notebooks/presentation_demo.ipynb` |
| Dataset / LOL | A | `data/dataset.py` |

---

## 9. Checklist

- [ ] Both GitHub accounts use commit emails
- [ ] `CONTRIBUTION_TEMPLATE.md` signed, names match git authors
- [ ] At least 3 commits per person on `main`
- [ ] Delete this file: `git rm docs/_DELETE_team_commit_plan.md`
- [ ] Do **not** commit `.venv/`, `data/lol/`, `outputs/`

---

## 10. Example timeline (2 evenings)

**Day 1 — Person A:** commits 1–4 + `run_eval.py`  
**Day 2 — Person B:** commits 5–9  
**Day 2 — together:** commit 10, run showcase, push `main`
