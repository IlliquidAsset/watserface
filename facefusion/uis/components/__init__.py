# Disable training components temporarily due to torch.uint64 safetensors conflict
TRAINING_AVAILABLE = False
print(f"⚠️ Training components temporarily disabled due to PyTorch/safetensors compatibility")

# Create dummy modules to avoid import errors
class DummyModule:
    def render(self): 
        import gradio
        return gradio.HTML("🔧 Training functionality temporarily disabled due to dependency conflicts")
    def listen(self): pass

dataset_manager = DummyModule()
model_trainer = DummyModule() 
training_options = DummyModule()
training_progress = DummyModule()