## 2026-01-13 - Broken Gradio Event Listeners
**Learning:** Components defined in `render()` but not registered or retrieved in `listen()` will cause runtime crashes. Gradio's separated render/listen loop requires explicit component retrieval via `get_ui_component` (or similar registry pattern) if the component variable is local to `render()`.
**Action:** Always verify that components used in `listen()` event handlers are available in the scope, either by re-fetching them from the registry or ensuring they are globally accessible (though registry is preferred).
