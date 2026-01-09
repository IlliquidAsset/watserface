# Bolt's Journal

## 2024-05-23 - Initial Setup
**Learning:** Performance journaling helps track impactful changes and avoid repeating mistakes.
**Action:** Always check this journal before starting optimization tasks.

## 2024-05-23 - Platform Check Optimization
**Learning:** `platform.system()` calls are relatively expensive (~240ns) and should be cached if used frequently in loops or hot paths. Since the OS is static during runtime, this value should be cached at module level. Module-level caching provides a significant speedup (~5x).
**Action:** Use module-level constants (cached boolean results) for static system values instead of repeated function calls or string comparisons.

## 2024-05-23 - Angle Estimation Optimization
**Learning:**  and  are expensive (~2µs) for simple scalar quantization compared to pure Python/math operations (~0.04µs). When snapping values to a grid (e.g., 90-degree increments), use O(1) math: .
**Action:** Replace array-based quantization with scalar math where the grid is regular.

## 2024-05-23 - Angle Estimation Optimization
**Learning:** `numpy.linspace` and `numpy.argmin` are expensive (~2µs) for simple scalar quantization compared to pure Python/math operations (~0.04µs). When snapping values to a grid (e.g., 90-degree increments), use O(1) math: `int(math.ceil((theta / step) - 0.5) * step)`.
**Action:** Replace array-based quantization with scalar math where the grid is regular.
