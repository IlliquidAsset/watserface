# Bolt's Journal

## 2024-05-23 - Initial Setup
**Learning:** Performance journaling helps track impactful changes and avoid repeating mistakes.
**Action:** Always check this journal before starting optimization tasks.

## 2024-05-23 - Platform Check Optimization
**Learning:** `platform.system()` calls are relatively expensive (~240ns) and should be cached if used frequently in loops or hot paths. Since the OS is static during runtime, this value should be cached at module level. Module-level caching provides a significant speedup (~5x).
**Action:** Use module-level constants (cached boolean results) for static system values instead of repeated function calls or string comparisons.
