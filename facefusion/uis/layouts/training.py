import gradio

from facefusion import state_manager
from facefusion.uis.components import about, common_options, memory, terminal
# Import training components with fallback
try:
    from facefusion.uis.components import dataset_manager, model_trainer, training_options, training_progress, TRAINING_AVAILABLE
except ImportError:
    TRAINING_AVAILABLE = False
    # Create dummy components
    class DummyComponent:
        def render(self): 
            return gradio.HTML("⚠️ Training components unavailable due to dependency conflicts")
        def listen(self): pass
    
    dataset_manager = model_trainer = training_options = training_progress = DummyComponent()

def pre_check() -> bool:
	return True

def render() -> gradio.Blocks:
	with gradio.Blocks() as layout:
		# Header
		about.render()
		
		with gradio.Row():
			with gradio.Column(scale = 4):
				with gradio.Accordion("🎯 Training Options", open=True):
					training_options.render()
				
				with gradio.Accordion("💾 Memory", open=False):
					memory.render()
					
				with gradio.Accordion("⚙️ Common Options", open=False):
					common_options.render()
					
			with gradio.Column(scale = 4):
				with gradio.Accordion("📊 Dataset Manager", open=True):
					dataset_manager.render()
					
				with gradio.Accordion("🤖 Model Trainer", open=True):
					model_trainer.render()
					
				with gradio.Accordion("💻 Terminal", open=False):
					terminal.render()
					
			with gradio.Column(scale = 7):
				with gradio.Accordion("📈 Training Progress", open=True):
					training_progress.render()
	return layout

def listen() -> None:
	training_options.listen()
	dataset_manager.listen()
	model_trainer.listen()
	training_progress.listen()
	memory.listen()
	common_options.listen()

def run(ui : gradio.Blocks) -> None:
	# This function is called by the core but doesn't need to launch - that's handled centrally
	pass