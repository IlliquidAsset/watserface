import gradio

from watserface import metadata


def render() -> None:
	with gradio.Row():
		gradio.HTML(f"""
		<div style="text-align: center; padding: 5px; opacity: 0.6; font-size: 0.8em;">
			{metadata.get('description')}
		</div>
		""")
