from pathlib import Path


def write_presentation(summary_rows, out_dir: Path, stats: dict):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Low-Light Enhancement & Detection — Presentation (10 min)",
        "",
        "## Slide 1 — Title",
        "- **Low-Light Enhancement & Detection**",
        "- Classical OpenCV pipeline: 5 connected stages",
        "",
        "## Slide 2 — Problem / Motivation",
        "- Dark images lose contrast → segmentation and detection fail",
        "- Goal: automatic enhancement + region detection + scene decision",
        "",
        "## Slide 3 — Pipeline (mandatory 5 stages)",
        "```",
        "Enhance → Segment → Clean → Detect → Decide",
        "```",
        "- **Enhance:** gamma, CLAHE, Retinex (auto-selected)",
        "- **Segment:** Otsu / adaptive / GrabCut (auto on real photos)",
        "- **Clean:** morphological opening + closing",
        "- **Detect:** contours → bounding boxes (+ ORB/FAST features in notebook)",
        "- **Decide:** count, coverage, automatic label (OK / ALERT / NO_OBJECTS)",
        "",
        "## Slide 4 — Dataset",
        f"- Real low-light images: LOL = {stats.get('lol', 0)}, custom = {stats.get('custom', 0)}",
        "- LOL: paired real low/normal photos (enhancement reference)",
        "- Optional: own photos in `data/custom/`",
        "",
        "## Slide 5 — Required outputs (per image)",
        "1. Original  2. Enhanced  3. Segmentation mask  4. Cleaned mask  5. Detection  6. Decision",
        "",
        "## Slide 6 — Live demo",
        "```bash",
        "python scripts/run_showcase.py --n 6",
        "```",
        "- Open `outputs/showcase/<image>/` — 6 PNG + pipeline strip",
        "",
        "## Slide 7 — Results",
    ]

    for row in summary_rows[:6]:
        lines.append(
            f"- **{row['name']}** ({row['source']}): {row['label']} | "
            f"{row['count']} obj | coverage {row['coverage']:.1%} | "
            f"enh={row['enhancer']} seg={row['segmenter']}"
        )

    lines.extend(
        [
            "",
            "## Slide 8 — Failure cases",
            "- Over-segmentation on textured regions (sky, noise) → too many boxes",
            "- Fully overlapping objects in synthetic scenes → watershed needed",
            "- Very dark input → Retinex may amplify noise (mitigated by noise-aware score)",
            "",
            "## Slide 9 — Conclusion",
            "- Full automatic pipeline on real low-light images",
            "- Interpretable decision: object count + coverage + alert flag",
            "- Future: deep enhancers (Zero-DCE), YOLO detection",
            "",
            "## Slide 10 — Q&A",
            "- Team roles: see `docs/CONTRIBUTION_TEMPLATE.md`",
            "",
        ]
    )

    path = out_dir / "PRESENTATION.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_report_skeleton(summary_rows, out_dir: Path, stats: dict):
    out_dir = Path(out_dir)
    lines = [
        "# Low-Light Enhancement & Detection — Team Report (draft)",
        "",
        "## 1. Problem description",
        "Describe why low-light breaks classical CV and what decision the system makes.",
        "",
        "## 2. Team roles & task division",
        "Fill in `docs/CONTRIBUTION_TEMPLATE.md` and attach signed PDF.",
        "",
        "## 3. Pipeline design",
        "Enhance → Segment → Clean → Detect → Decide (see README diagram).",
        "",
        "## 4. Methods used (OpenCV)",
        "- Enhancement: gamma, adaptive gamma, histogram eq, CLAHE, SSR/MSR/MSRCR",
        "- Segmentation: global/Otsu/adaptive threshold, GrabCut",
        "- Morphology: erode/dilate/open/close, connected components",
        "- Detection: contours + bounding boxes; Harris/ORB in experiments",
        "- Decision: automatic count/coverage/load + alert label",
        "",
        "## 5. Results (stage-by-stage images)",
        f"Processed {len(summary_rows)} images (LOL={stats.get('lol', 0)}, custom={stats.get('custom', 0)}).",
        "",
        "| Image | Source | Enhancer | Segmenter | Objects | Coverage | Decision |",
        "|-------|--------|----------|-----------|---------|----------|----------|",
    ]

    for row in summary_rows:
        lines.append(
            f"| {row['name']} | {row['source']} | {row['enhancer']} | {row['segmenter']} | "
            f"{row['count']} | {row['coverage']:.1%} | {row['label']} |"
        )

    lines.extend(
        [
            "",
            "Insert pipeline strips from `outputs/showcase/<name>/`.",
            "",
            "## 6. Failure cases",
            "- Document 2–3 examples where segmentation or detection fails.",
            "",
            "## 7. Conclusion",
            "- Summarize what works on real images and limitations.",
            "",
        ]
    )

    path = out_dir / "REPORT_DRAFT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
