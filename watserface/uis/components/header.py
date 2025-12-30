import gradio

from watserface import metadata
from watserface.uis.components import system_settings


def render() -> None:
	"""Render header with logo on left and system settings on right"""
	with gradio.Row(elem_classes="header-row"):
		with gradio.Column(scale=3):
			gradio.HTML("""
			<div style="text-align: left; padding: 15px 0;">
				<!-- Official WatserFace Logo -->
				<svg width="320" height="70" viewBox="0 0 1200 200" fill="none" xmlns="http://www.w3.org/2000/svg" style="max-width: 100%; height: auto;">
				  <g id="Mark">
					<path d="M100 20C60 20 30 50 30 100C30 140 55 175 90 185V25C93 22 96 20 100 20Z" fill="#4D4DFF"/>
					<path d="M110 10C114 10 118 11 122 13V175C155 165 180 130 180 90C180 40 150 10 110 10Z" fill="#FF00FF"/>
					<rect x="95" y="40" width="20" height="5" fill="#CCFF00"/>
					<rect x="95" y="150" width="20" height="5" fill="#CCFF00"/>
				  </g>
				  <g id="Text" transform="translate(220, 135)">
					<text font-family="sans-serif" font-weight="800" font-size="120" fill="#F2F2F2" letter-spacing="-2">
					  WATSER<tspan fill="#CCFF00">FACE</tspan>
					</text>
				  </g>
				</svg>
			</div>
			""")

		with gradio.Column(scale=1, min_width=200):
			# System Settings on the right
			system_settings.render()


def listen() -> None:
	"""Set up event listeners for header components"""
	system_settings.listen()
