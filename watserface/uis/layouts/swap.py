import gradio

from watserface import state_manager
from watserface.uis.components import footer, face_detector, face_landmarker, face_masker, face_selector, face_swapper_options, instant_runner, job_manager, job_runner, output, preview, source, target, terminal, trim_frame, ui_workflow, smart_presets


def pre_check() -> bool:
	return True


def render() -> gradio.Blocks:
	with gradio.Blocks() as layout:
		# Main 3-Column Layout: Source | Target | Output & Control
		with gradio.Row():
			# Column 1: Source
			with gradio.Column(scale = 1):
				with gradio.Accordion("📁 Source", open=True):
					source.render()

				with gradio.Accordion("👤 Custom Identity / Face Swapper", open=True):
					face_swapper_options.render()

				with gradio.Accordion("👤 Face Selector", open=False):
					face_selector.render()

			# Column 2: Target
			with gradio.Column(scale = 1):
				with gradio.Accordion("🎯 Target", open=True):
					target.render()

				with gradio.Accordion("✂️ Trim Frame", open=False):
					trim_frame.render()

				with gradio.Accordion("🎭 Face Masker", open=False):
					face_masker.render()

				with gradio.Accordion("🔍 Face Detector", open=False):
					face_detector.render()

				with gradio.Accordion("📍 Face Landmarker", open=False):
					face_landmarker.render()

			# Column 3: Output & Control
			with gradio.Column(scale = 1):
				with gradio.Accordion("⚡ Smart Presets", open=True):
					smart_presets.render()

				with gradio.Accordion("👁️ Preview", open=True):
					preview.render()

				with gradio.Accordion("💾 Output", open=True):
					output.render()

				with gradio.Accordion("🔄 Workflow", open=True):
					ui_workflow.render()
					instant_runner.render()
					job_runner.render()
					job_manager.render()

				with gradio.Accordion("💻 Terminal", open=False):
					terminal.render()

		# Footer
		footer.render()

	return layout


def listen() -> None:
	# Component listeners
	smart_presets.listen()
	face_swapper_options.listen()
	source.listen()
	target.listen()
	output.listen()
	instant_runner.listen()
	job_runner.listen()
	job_manager.listen()
	terminal.listen()
	preview.listen()
	trim_frame.listen()
	face_selector.listen()
	face_masker.listen()
	face_detector.listen()
	face_landmarker.listen()


def run(ui : gradio.Blocks) -> None:
	ui.launch(inbrowser = state_manager.get_item('open_browser'), server_name = state_manager.get_item('server_name'), server_port = state_manager.get_item('server_port'), show_error = True)
