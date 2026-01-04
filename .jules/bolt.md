## 2024-02-23 - [Platform Checks Optimization]
**Learning:** `platform.system()` calls are relatively expensive (~240ns) and should be cached if used frequently in loops or hot paths. Module-level caching provides a significant speedup (~5x).
**Action:** Identify static system properties and cache them at module level instead of re-evaluating them in function calls.
