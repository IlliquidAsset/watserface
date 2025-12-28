## 2024-05-24 - Tooltips in Gradio
**Learning:** Gradio components have an `info` parameter that renders a tooltip, which is excellent for accessibility and UX. However, this codebase separates wording into `wording.py` but wasn't utilizing the `help` section for these `info` tooltips, leaving users guessing about complex options.
**Action:** Always check if a separate wording/strings file exists and if it contains help text that can be mapped to component `info` props.
