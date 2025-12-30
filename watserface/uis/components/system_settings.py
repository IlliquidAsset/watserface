import gradio

from watserface.uis.components import common_options, download, execution, execution_queue_count, execution_thread_count, memory, output_options, temp_frame, processors, face_enhancer_options


def render() -> None:
	"""Render system-wide settings in a collapsible accordion"""
	with gradio.Accordion("⚙️ System Settings", open=False):
		with gradio.Row():
			with gradio.Column():
				with gradio.Group():
					gradio.Markdown("### 🎛️ Processors")
					processors.render()

				with gradio.Group():
					gradio.Markdown("### ⚡ Execution")
					execution.render()
					execution_thread_count.render()
					execution_queue_count.render()

				with gradio.Group():
					gradio.Markdown("### 💾 Memory")
					memory.render()

			with gradio.Column():
				with gradio.Group():
					gradio.Markdown("### ✨ Face Enhancer")
					face_enhancer_options.render()

				with gradio.Group():
					gradio.Markdown("### 📥 Download")
					download.render()

				with gradio.Group():
					gradio.Markdown("### 🗂️ Temp Frame")
					temp_frame.render()

			with gradio.Column():
				with gradio.Group():
					gradio.Markdown("### 📤 Output Options")
					output_options.render()

				with gradio.Group():
					gradio.Markdown("### ⚙️ Common Options")
					common_options.render()


def listen() -> None:
	"""Set up event listeners for all system settings components"""
	processors.listen()
	face_enhancer_options.listen()
	execution.listen()
	execution_thread_count.listen()
	execution_queue_count.listen()
	download.listen()
	memory.listen()
	temp_frame.listen()
	output_options.listen()
	common_options.listen()
