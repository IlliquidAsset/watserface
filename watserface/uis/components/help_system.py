import gradio
from typing import Dict, List, Optional
from watserface.uis.core import register_ui_component


class HelpSystem:
    """Comprehensive help and guidance system"""
    
    HELP_CONTENT = {
        "upload": {
            "title": "📁 Upload Guide",
            "content": """
            ### Getting Started with Uploads
            
            **Source Image Tips:**
            - 📸 Use front-facing photos for best results
            - 💡 Ensure good lighting and clear face visibility
            - ❌ Avoid sunglasses, masks, or extreme angles
            - 📏 Higher resolution (1080p+) works better
            - 🎯 One clear face per image is ideal
            
            **Target Media Tips:**
            - 🎬 Videos: MP4, MOV formats recommended
            - 📷 Images: JPG, PNG for still image swaps
            - ⏱️ Keep test videos under 30 seconds
            - 🔍 Clear faces get better results
            - 💾 File size limit: 100MB per file
            
            **Supported Formats:**
            - Images: JPG, PNG, BMP, WEBP
            - Videos: MP4, MOV, AVI, MKV, WEBM
            """,
            "quick_tips": [
                "Drag and drop files directly onto upload areas",
                "Use Ctrl+1 to quickly navigate to upload tab",
                "Check preview images to verify correct uploads"
            ]
        },
        "preview": {
            "title": "🔍 Preview System Guide",
            "content": """
            ### Smart Preview Selection
            
            **Quality Presets Explained:**
            - 🚀 **Fast**: Quick draft quality (~15 seconds)
              - Best for: Testing, multiple iterations
              - Quality: Basic, good for previews
              - Processing: GPU optimized, minimal enhancement
            
            - ⚖️ **Balanced**: Good quality, reasonable speed (~45 seconds)
              - Best for: Most users, general use
              - Quality: High quality with face enhancement
              - Processing: Optimized balance of speed and quality
            
            - ✨ **Quality**: Best results, slower processing (~90 seconds)
              - Best for: Final outputs, professional use
              - Quality: Maximum quality with advanced enhancement
              - Processing: All quality features enabled
            
            **How to Use:**
            1. Select your preferred quality preset
            2. Click "Generate Previews" to see results
            3. Choose the option you like best
            4. Proceed to full processing
            """,
            "quick_tips": [
                "Use keyboard shortcuts 1, 2, 3 to select presets quickly",
                "Start with Balanced preset for best results",
                "Preview generation uses small samples for speed"
            ]
        },
        "process": {
            "title": "⚡ Processing Guide",
            "content": """
            ### Processing Your Media
            
            **What Happens During Processing:**
            1. 🔍 **Face Detection**: Locates faces in your media
            2. 🎯 **Face Analysis**: Extracts facial features and landmarks
            3. 🔄 **Face Swapping**: Replaces target faces with source face
            4. ✨ **Enhancement**: Improves quality and blending (if enabled)
            5. 📦 **Output Generation**: Creates final video/image file
            
            **Processing Time Factors:**
            - 🎯 Quality preset (Fast < Balanced < Quality)
            - 📏 Media resolution (720p < 1080p < 4K)
            - ⏱️ Video length (longer = more time)
            - 🖥️ Hardware (GPU much faster than CPU)
            - 👥 Number of faces in media
            
            **Monitoring Progress:**
            - Real-time status updates in progress panel
            - Step-by-step progress indicators
            - Estimated time remaining
            - Cancel option if needed
            """,
            "quick_tips": [
                "GPU processing is 5-10x faster than CPU",
                "Shorter videos process much faster",
                "You can monitor progress in the terminal"
            ]
        },
        "advanced": {
            "title": "🎓 Advanced Features",
            "content": """
            ### Advanced Settings & Customization
            
            **When to Use Advanced Settings:**
            - 🔧 Fine-tuning specific parameters
            - 🎯 Handling difficult/unusual source images
            - 📊 Professional/commercial use cases
            - 🧪 Experimental features and testing
            
            **Key Advanced Options:**
            - **Face Detector Model**: Controls detection accuracy
            - **Face Enhancer Settings**: Quality enhancement options
            - **Execution Providers**: CPU/GPU selection
            - **Memory Management**: RAM usage optimization
            - **Output Settings**: Resolution, quality, format options
            
            **Model Training (Coming Soon):**
            - Upload custom datasets
            - Train personalized face models
            - Manage and deploy trained models
            - Advanced training parameters
            
            **Performance Tuning:**
            - Adjust thread count for your system
            - Memory optimization settings
            - Execution provider selection
            - Batch processing options
            """,
            "quick_tips": [
                "Most users should stick to preset options",
                "Advanced settings can improve results for edge cases",
                "Model training requires significant computational resources"
            ]
        },
        "troubleshooting": {
            "title": "🔧 Troubleshooting",
            "content": """
            ### Common Issues & Solutions
            
            **Upload Issues:**
            - ❌ File format not supported → Use JPG, PNG, MP4, MOV
            - ❌ File too large → Compress to under 100MB
            - ❌ No face detected → Use clearer, front-facing photos
            
            **Processing Issues:**
            - ⚠️ Out of memory → Reduce video resolution or length
            - ⚠️ Processing stuck → Check terminal for detailed errors
            - ⚠️ Poor results → Try different quality preset or source image
            
            **Quality Issues:**
            - 🎭 Unnatural looking results → Use higher quality preset
            - 🎭 Face doesn't align well → Try different source angle
            - 🎭 Blurry output → Use higher resolution source images
            
            **Performance Issues:**
            - 🐌 Very slow processing → Enable GPU if available
            - 🐌 System freezing → Reduce memory usage in advanced settings
            - 🐌 Long wait times → Try shorter videos or Fast preset
            
            **Getting Help:**
            - Check terminal output for error messages
            - Try different source images if results are poor
            - Use keyboard shortcut '?' for quick help
            - Start with Fast preset for testing
            """,
            "quick_tips": [
                "Always check terminal output for errors",
                "Try Fast preset first to test your setup",
                "Good source images are crucial for quality results"
            ]
        }
    }
    
    @classmethod
    def get_help_content(cls, section: str) -> Dict:
        """Get help content for a specific section"""
        return cls.HELP_CONTENT.get(section, {
            "title": "❓ Help",
            "content": "Help content not available for this section.",
            "quick_tips": []
        })
    
    @classmethod
    def render_help_panel(cls, section: str = "upload") -> None:
        """Render contextual help panel"""
        help_data = cls.get_help_content(section)
        
        with gradio.Column() as help_panel:
            with gradio.Accordion("❓ Need Help?", open=False) as help_accordion:
                help_title = gradio.Markdown(f"## {help_data['title']}")
                help_content = gradio.Markdown(help_data['content'])
                
                if help_data.get('quick_tips'):
                    gradio.Markdown("### 💡 Quick Tips")
                    tips_html = "<ul>" + "".join([
                        f"<li>{tip}</li>" for tip in help_data['quick_tips']
                    ]) + "</ul>"
                    gradio.HTML(tips_html)
                
                # Keyboard shortcuts reminder
                shortcuts_button = gradio.Button(
                    value="⌨️ View Keyboard Shortcuts",
                    variant="secondary",
                    size="sm"
                )
        
        register_ui_component(f'help_panel_{section}', help_panel)
        register_ui_component(f'help_accordion_{section}', help_accordion)
        register_ui_component(f'shortcuts_button_{section}', shortcuts_button)
        
        return help_panel, help_accordion, shortcuts_button


def render_tooltip(text: str, tooltip_text: str, element_id: str = "") -> gradio.HTML:
    """Render an element with tooltip"""
    html_content = f"""
    <span class="tooltip" data-tooltip="{tooltip_text}" id="{element_id}">
        {text}
    </span>
    """
    return gradio.HTML(html_content)


def render_guided_tour() -> None:
    """Render guided tour component for first-time users"""
    with gradio.Column(visible=True) as tour_panel:
        gradio.Markdown(
            """
            ## 🎉 Welcome to WatserFace!
            
            **New to face swapping? Here's a quick guide to get started:**
            
            ### Step-by-Step Process:
            1. **📁 Upload**: Add your source face photo and target video/image
            2. **🔍 Preview**: Choose from 3 quality options (Fast/Balanced/Quality)  
            3. **⚡ Process**: Generate your final result
            4. **📥 Download**: Save your face-swapped media
            
            ### First-Time Tips:
            - Start with the **Balanced** preset for best results
            - Use clear, front-facing photos as source images
            - Keep test videos short (under 30 seconds)
            - Check the **Need Help?** sections for detailed guidance
            
            Press **?** anytime for keyboard shortcuts!
            """,
            elem_id="guided_tour_content"
        )
        
        tour_dismiss = gradio.Button(
            value="✅ Got it, let's start!",
            variant="primary",
            size="lg"
        )
    
    register_ui_component('guided_tour_panel', tour_panel)
    register_ui_component('guided_tour_dismiss', tour_dismiss)
    
    return tour_panel, tour_dismiss


def setup_help_system_listeners() -> None:
    """Setup event listeners for help system"""
    # This would be called from the main layout's listen() method
    
    # Hide tour when dismissed
    tour_panel = register_ui_component('guided_tour_panel', None)
    tour_dismiss = register_ui_component('guided_tour_dismiss', None)
    
    if tour_panel and tour_dismiss:
        tour_dismiss.click(
            fn=lambda: gradio.update(visible=False),
            outputs=[tour_panel]
        )
    
    # Setup keyboard shortcuts button handlers
    for section in ["upload", "preview", "process", "advanced"]:
        shortcuts_button = register_ui_component(f'shortcuts_button_{section}', None)
        if shortcuts_button:
            shortcuts_button.click(
                fn=lambda: "Keyboard shortcuts modal opened via JavaScript",
                outputs=[],
                js="() => { if (window.WatserFaceA11y) window.WatserFaceA11y.showKeyboardShortcuts(); }"
            )


def create_smart_tooltips() -> Dict[str, str]:
    """Create smart tooltips for various UI elements"""
    return {
        "source_upload": "Upload a clear, front-facing photo of the person whose face you want to use",
        "target_upload": "Upload the video or image where you want to place the face",
        "preset_fast": "Quick processing with basic quality - great for testing",
        "preset_balanced": "Good balance of quality and speed - recommended for most users",
        "preset_quality": "Highest quality output with slower processing time",
        "generate_preview": "Create preview samples using different quality settings",
        "process_button": "Start processing your full media with selected settings",
        "advanced_toggle": "Show technical parameters for fine-tuning (advanced users)",
        "progress_status": "Monitor real-time processing progress and estimated completion time"
    }


# Export help system for use in layouts
def get_help_system():
    """Get help system instance"""
    return HelpSystem()