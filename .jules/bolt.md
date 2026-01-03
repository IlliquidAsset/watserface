# Bolt's Journal

## 2024-05-23 - Initial Setup
**Learning:** Performance journaling helps track impactful changes and avoid repeating mistakes.
**Action:** Always check this journal before starting optimization tasks.

## 2024-05-23 - Platform Check Optimization
**Learning:** `platform.system()` involves a system call or overhead. Since the OS is static during runtime, this value should be cached.
**Action:** Use module-level constants for static system values instead of repeated function calls.
