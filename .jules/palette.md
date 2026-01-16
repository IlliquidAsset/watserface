## 2024-05-22 - Gradio File Type Filtering
**Learning:** Using generic Gradio file types (e.g., `['image', 'audio']`) is preferred over specific extensions (e.g., `['.png', '.mp3']`) for this codebase. This provides a cleaner file picker experience while allowing the backend (`filesystem.py`) to handle strict format validation without duplicating the exhaustive list of extensions in the UI layer.
**Action:** When adding file inputs, use generic types unless specific extension filtering is strictly required by a unique constraint.
