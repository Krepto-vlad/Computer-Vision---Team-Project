# Свои реальные фото

Положи сюда 3+ тёмных JPG/PNG — они автоматически попадут в демо:

```
data/custom/photo1.jpg
data/custom/photo2.png
```

Запуск:

```bash
python scripts/run_showcase.py --n 6
```

Приоритет: **LOL** (реальный low-light датасет) → **custom/** → синтетика (fallback).
