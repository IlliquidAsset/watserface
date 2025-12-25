import gradio

from facefusion import metadata


def render() -> None:
	with gradio.Row():
		gradio.HTML(f"""
		<div style="text-align: center; padding: 10px;">
			<a href="{metadata.get('url')}" target="_blank" rel="noopener noreferrer" style="text-decoration: none; color: inherit;" aria-label="Visit {metadata.get('name')} website">
				<h2 style="margin: 0;">{metadata.get('name')} {metadata.get('version')} - Training Edition</h2>
			</a>
			<p style="margin: 5px 0; font-size: 0.9em; opacity: 0.8;">{metadata.get('description')}</p>
		</div>
		""")
