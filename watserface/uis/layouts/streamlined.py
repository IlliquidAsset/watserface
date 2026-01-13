import gradio

from watserface import state_manager
from watserface.uis.components import about, common_options, execution, face_detector, face_landmarker, face_masker, face_selector, instant_runner, job_manager, job_runner, memory, output, output_options, processors, smart_preview, source, target, terminal, ui_workflow
from watserface.uis.components import enhanced_source, enhanced_target, progress_tracker, help_system, error_handler
from watserface.identity_profile import get_identity_manager
from watserface.face_set import list_face_sets, get_face_set_manager
from watserface.training.core import start_identity_training


def pre_check() -> bool:
    return True


def update_identity_status():
    profiles = get_identity_manager().source_intelligence.list_profiles()
    if profiles:
        names = [p.name for p in profiles]
        return f"Found {len(profiles)} identities: {', '.join(names[:5])}" + ("..." if len(names)>5 else "")
    return "No identities yet"


def render() -> gradio.Blocks:
    """Render streamlined 4-step workflow layout with enhanced UX"""
    with gradio.Blocks(
        css_paths=["watserface/uis/assets/enhanced_styles.css"],
        js_paths=["watserface/uis/assets/keyboard_navigation.js"]
    ) as layout:
        # Initialize error handling
        error_handler.setup_error_handling()
        
        # Header
        about.render()
        
        # Error display (initially hidden)
        error_handler.render_error_display()
        
        # Guided tour for new users
        guided_tour_panel, tour_dismiss = help_system.render_guided_tour()
        
        # Main workflow tabs
        with gradio.Tabs() as main_tabs:
            # Step 1: Upload
            with gradio.Tab("📁 Upload", id="upload_tab"):
                # Help panel for upload section
                help_system.HelpSystem.render_help_panel("upload")
                
                with gradio.Row():
                    with gradio.Column(scale=1):
                        # Empty state walkthrough
                        with gradio.Column(visible=True) as upload_empty_state:
                            gradio.Markdown(
                                """
                                ## 🚀 Let's Get Started!
                                
                                **Step 1: Upload Your Media**
                                
                                1. **📸 Source Image**: Upload a clear photo of the person whose face you want to use
                                2. **🎯 Target Media**: Upload the video or image where you want to place the face
                                
                                ### 💡 Pro Tips for Best Results:
                                - ✅ Use front-facing, well-lit photos
                                - ✅ Higher resolution images work better
                                - ✅ Avoid sunglasses, masks, or extreme angles
                                - ✅ Keep test videos under 30 seconds
                                
                                **Ready? Start by uploading your source image below! ⬇️**
                                """,
                                elem_id="upload_empty_state_guide"
                            )
                    
                    with gradio.Column(scale=2):
                        # Identity Status
                        with gradio.Row():
                            identity_status = gradio.Textbox(value=update_identity_status(), label="Identities Status", interactive=False)
                            refresh_btn = gradio.Button("Refresh Identities")
                            refresh_btn.click(update_identity_status, outputs=identity_status)

                        with gradio.Row():
                            with gradio.Column():
                                enhanced_source.render()
                            with gradio.Column():
                                enhanced_target.render()
                        
                        # System requirements check
                        system_warnings = gradio.Markdown(
                            "",
                            elem_id="system_warnings",
                            visible=False
                        )
                        
                        next_to_preview_button = gradio.Button(
                            value="➡️ Next: Generate Previews",
                            variant="primary",
                            size="lg",
                            interactive=False
                        )
            
            # Step 2: Preview Selection  
            with gradio.Tab("🔍 Preview", id="preview_tab"):
                # Help panel for preview section
                help_system.HelpSystem.render_help_panel("preview")
                
                with gradio.Row():
                    with gradio.Column(scale=1):
                        # Preview empty state
                        with gradio.Column(visible=True) as preview_empty_state:
                            gradio.Markdown(
                                """
                                ## 🎯 Smart Quality Selection
                                
                                **Choose your perfect balance of speed vs quality:**
                                
                                ### 🚀 Fast Preset
                                - ⏱️ **Time**: ~15 seconds
                                - 🎯 **Best for**: Testing, multiple iterations
                                - 📊 **Quality**: Good for previews
                                
                                ### ⚖️ Balanced Preset ⭐ *Recommended*
                                - ⏱️ **Time**: ~45 seconds  
                                - 🎯 **Best for**: Most users, everyday use
                                - 📊 **Quality**: High quality with enhancement
                                
                                ### ✨ Quality Preset
                                - ⏱️ **Time**: ~90+ seconds
                                - 🎯 **Best for**: Final outputs, professional work
                                - 📊 **Quality**: Maximum quality, all features
                                
                                **New to face swapping? Start with Balanced!**
                                """,
                                elem_id="preview_empty_state_guide"
                            )
                    
                    with gradio.Column(scale=3):
                        smart_preview.render()
                        
                        next_to_export_button = gradio.Button(
                            value="➡️ Next: Process Full Media",
                            variant="primary",
                            size="lg",
                            interactive=False
                        )
            
            # Step 3: Export/Processing
            with gradio.Tab("⚡ Process", id="process_tab"):
                # Help panel for processing section
                help_system.HelpSystem.render_help_panel("process")
                
                with gradio.Row():
                    with gradio.Column(scale=1):
                        # Processing empty state
                        with gradio.Column(visible=True) as process_empty_state:
                            gradio.Markdown(
                                """
                                ## ⚡ Ready to Process!
                                
                                **Your settings are configured. Here's what happens next:**
                                
                                ### Processing Steps:
                                1. 🔍 **Face Detection** - Locate faces in media
                                2. 🎯 **Face Analysis** - Extract facial features  
                                3. 🔄 **Face Swapping** - Replace target faces
                                4. ✨ **Enhancement** - Improve quality (if enabled)
                                5. 📦 **Output** - Generate final file
                                
                                ### ⏱️ Time Estimates:
                                - **GPU**: 5-10x faster processing
                                - **Video Length**: Longer = more time
                                - **Resolution**: Higher = slower but better
                                - **Your Preset**: Affects speed and quality
                                
                                **Click Process when ready!**
                                """,
                                elem_id="process_empty_state_guide"
                            )
                    
                    with gradio.Column(scale=2):
                        # Enhanced progress tracking
                        progress_tracker.render()
                        
                        # Output settings (simplified)
                        with gradio.Accordion("📤 Output", open=True):
                            output.render()
                        
                        # Execution controls
                        with gradio.Accordion("⚡ Processing", open=True):
                            ui_workflow.render()
                            instant_runner.render()
                            job_runner.render()
                        
                        # Terminal for detailed progress
                        with gradio.Accordion("📊 Detailed Progress", open=False):
                            terminal.render()
            
            # Step 4: Advanced/Training (for power users)
            with gradio.Tab("🎓 Advanced", id="advanced_tab"):
                # Help panel for advanced section
                help_system.HelpSystem.render_help_panel("advanced")
                
                with gradio.Row():
                    with gradio.Column(scale=1):
                        # Advanced empty state
                        with gradio.Column(visible=True) as advanced_empty_state:
                            gradio.Markdown(
                                """
                                ## 🎓 Advanced Features
                                
                                **For Power Users & Fine-tuning**
                                
                                ### 🔧 When to Use Advanced Settings:
                                - Fine-tuning for specific use cases
                                - Handling difficult source images
                                - Professional/commercial projects
                                - Performance optimization
                                - Experimental features
                                
                                ### ⚠️ Important Notes:
                                - **Beginners**: Stick to preset options in Preview tab
                                - **Experts**: These settings override presets
                                - **Training**: Custom models (coming in Phase 4)
                                - **Support**: Advanced features have limited guidance
                                
                                **Most users get better results with presets!**
                                """,
                                elem_id="advanced_empty_state_guide"
                            )
                    
                    with gradio.Column(scale=3):
                        # Advanced processor settings (collapsed by default)
                        with gradio.Accordion("🎛️ Processors", open=False):
                            processors.render()
                        
                        with gradio.Accordion("🔧 Technical Settings", open=False):
                            # Face detection settings
                            with gradio.Row():
                                with gradio.Column():
                                    face_detector.render()
                                with gradio.Column():
                                    face_landmarker.render()
                            
                            # Face processing settings
                            with gradio.Row():
                                with gradio.Column():
                                    face_selector.render()
                                with gradio.Column():
                                    face_masker.render()
                            
                            # System settings
                            execution.render()
                            memory.render()
                            common_options.render()
                            output_options.render()
                        
                        # Job management for advanced workflows
                        with gradio.Accordion("📋 Job Management", open=False):
                            job_manager.render()
                        
                        # Training placeholder (to be implemented)
                        with gradio.Accordion("🎓 Model Training", open=False):
                            gradio.Markdown("### Custom Model Training")

                            with gradio.Row():
                                identity_choices = [""] + [p.id for p in get_identity_manager().source_intelligence.list_profiles()]
                                identity_dropdown = gradio.Dropdown(choices=identity_choices, label="Resume Identity (Optional)")
                                
                                faceset_choices = [""] + list_face_sets()
                                face_set_dropdown = gradio.Dropdown(choices=faceset_choices, label="Reuse FaceSet (Optional)")

                            with gradio.Row():
                                epochs_slider = gradio.Slider(minimum=10, maximum=1000, value=100, label="Epochs")
                                batch_size_slider = gradio.Slider(minimum=1, maximum=16, value=4, label="Batch Size")
                                learning_rate_slider = gradio.Slider(minimum=1e-6, maximum=1e-3, value=1e-4, label="Learning Rate")

                            face_set_status = gradio.Textbox(value=f"Found {len(list_face_sets())} FaceSets", label="FaceSets Status", interactive=False)

                            with gradio.Row():
                                train_btn = gradio.Button("Start Training", variant="primary")
                                cleanup_btn = gradio.Button("Cleanup Old Sessions")

                            # Cleanup callback
                            def cleanup_action():
                                manager = get_face_set_manager()
                                msg = manager.cleanup_old_face_sets()
                                status = f"Found {len(manager.list_face_sets())} FaceSets"
                                return msg + "\n" + status, status

                            cleanup_output = gradio.Textbox(label="Cleanup Result", visible=True)
                            cleanup_btn.click(cleanup_action, outputs=[cleanup_output, face_set_status])

                            # Train callback
                            def train_action(identity, faceset, epochs, batch, lr, progress=gradio.Progress()):
                                import time
                                model_name = identity
                                if not model_name and faceset:
                                    model_name = f"Identity_from_{faceset}"
                                
                                if not model_name:
                                    # Fallback to generic name if only sources provided
                                    source_files = state_manager.get_item('source_paths')
                                    if source_files:
                                        model_name = f"Identity_{int(time.time())}"
                                    else:
                                        yield "❌ Error: Please select an identity, a faceset, or upload source files."
                                        return

                                try:
                                    source_files = state_manager.get_item('source_paths')

                                    for status in start_identity_training(
                                        model_name=model_name,
                                        epochs=epochs,
                                        face_set_id=faceset if faceset else None,
                                        source_files=source_files if not faceset else None,
                                        progress=progress
                                    ):
                                        yield status[0]
                                except Exception as e:
                                    yield f"Error: {str(e)}"

                            train_info = gradio.Textbox(label="Training Info")
                            train_btn.click(train_action, inputs=[identity_dropdown, face_set_dropdown, epochs_slider, batch_size_slider, learning_rate_slider], outputs=[train_info])
        
        # Register tab navigation components
        from watserface.uis.core import register_ui_component
        register_ui_component('main_tabs', main_tabs)
        register_ui_component('next_to_preview_button', next_to_preview_button)
        register_ui_component('next_to_export_button', next_to_export_button)
    
    return layout


def listen() -> None:
    """Set up event listeners for streamlined layout"""
    
    # Enhanced components listeners
    enhanced_source.listen()
    enhanced_target.listen()
    smart_preview.listen()
    progress_tracker.listen()
    error_handler.listen_error_events()
    help_system.setup_help_system_listeners()
    
    # Original components that need listeners
    processors.listen()
    execution.listen()
    memory.listen()
    common_options.listen()
    output_options.listen()
    source.listen()
    target.listen()
    output.listen()
    instant_runner.listen()
    job_runner.listen()
    job_manager.listen()
    terminal.listen()
    face_selector.listen()
    face_masker.listen()
    face_detector.listen()
    face_landmarker.listen()
    
    # Get navigation components
    from watserface.uis.core import get_ui_component
    main_tabs = get_ui_component('main_tabs')
    next_to_preview_button = get_ui_component('next_to_preview_button')
    next_to_export_button = get_ui_component('next_to_export_button')
    
    if main_tabs and next_to_preview_button:
        # Navigate to preview tab when "Next: Generate Previews" is clicked
        next_to_preview_button.click(
            fn=lambda: gradio.update(selected="preview_tab"),
            outputs=[main_tabs]
        )
    
    if main_tabs and next_to_export_button:
        # Navigate to process tab when "Next: Process Full Media" is clicked  
        next_to_export_button.click(
            fn=lambda: gradio.update(selected="process_tab"),
            outputs=[main_tabs]
        )


def run(ui: gradio.Blocks) -> None:
    """Launch the streamlined UI"""
    ui.launch(
        inbrowser=state_manager.get_item('open_browser'),
        server_name=state_manager.get_item('server_name'),
        server_port=state_manager.get_item('server_port')
    )