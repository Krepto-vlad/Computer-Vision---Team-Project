# Low-Light Enhancement & Detection

Классический CV-пайплайн для тёмных изображений:

**Enhancement → Segmentation → Morphology → Detection → Decision**

Проект опирается на лабораторные работы по сегментации (Lab 07) и детекции признаков (Lab 08).

## Структура проекта

```
TEAMproject/
├── data/                    # датасет (LOL или синтетика), не в git
├── outputs/                 # отчёты, CSV, визуализации
├── notebooks/               # Jupyter-ноутбуки
├── scripts/                 # CLI-скрипты
└── src/lowlight_cv/         # основной Python-пакет
    ├── config.py            # пути и константы
    ├── data/                # загрузка LOL, синтетические сцены
    ├── enhancement/         # gamma, CLAHE, Retinex, …
    ├── segmentation/        # пороги, watershed, GrabCut
    ├── morphology/          # opening/closing, clean_mask
    ├── detection/           # Harris, ORB, bounding boxes
    ├── metrics/             # PSNR, SSIM, no-reference метрики
    ├── decision/            # решение о сцене
    ├── pipeline/            # LowLightPipeline
    └── utils/               # визуализация
```

## Установка

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
pip install -e .
```

## Быстрый старт

```bash
python scripts/run_demo.py
```

Или в коде:

```python
from lowlight_cv.pipeline import LowLightPipeline

pipe = LowLightPipeline(enhance="auto", segment="otsu")
result = pipe.process(image_bgr)
pipe.visualize(result)
```

## Датасет

По умолчанию скрипт пытается скачать **LOL** (paired low/normal). Если загрузка недоступна — генерируется синтетический набор с GT-масками.

## Основной ноутбук

Полный интерактивный пайплайн с отчётом и презентацией: `lowlight_enhancement_detection.ipynb`.
