import gradio

from watserface import metadata


def render() -> None:
	with gradio.Row():
		gradio.HTML(f"""
		<div style="text-align: center; padding: 20px 10px; border-top: 1px solid rgba(128, 128, 128, 0.2); margin-top: 20px;">
			<p style="margin: 5px 0; font-size: 0.9em; opacity: 0.7;">
				<strong>{metadata.get('name')} v{metadata.get('version')}</strong>
			</p>
			<p style="margin: 5px 0; font-size: 0.85em; opacity: 0.6;">
				Based on <a href="https://github.com/facefusion/facefusion" target="_blank" rel="noopener noreferrer" style="color: inherit; text-decoration: underline;">FaceFusion</a> by Henry Ruhs
			</p>
			<p style="margin: 5px 0; font-size: 0.8em; opacity: 0.5;">
				Licensed under {metadata.get('license')} |
				<a href="{metadata.get('url')}" target="_blank" rel="noopener noreferrer" style="color: inherit;">View on GitHub</a>
			</p>
		</div>
		""")
