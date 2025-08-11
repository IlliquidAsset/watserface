/* Keyboard Navigation and Accessibility Enhancements */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize keyboard navigation
    initializeKeyboardNavigation();
    
    // Add focus indicators
    addFocusIndicators();
    
    // Setup tab navigation
    setupTabNavigation();
    
    // Add keyboard shortcuts
    setupKeyboardShortcuts();
});

function initializeKeyboardNavigation() {
    // Make all interactive elements focusable
    const interactiveElements = document.querySelectorAll(
        'button, input, select, textarea, [tabindex], .gradio-file, .gradio-button'
    );
    
    interactiveElements.forEach((element, index) => {
        if (!element.hasAttribute('tabindex')) {
            element.setAttribute('tabindex', '0');
        }
        
        // Add keyboard event listeners
        element.addEventListener('keydown', handleKeyboardNavigation);
    });
}

function handleKeyboardNavigation(event) {
    const element = event.target;
    
    switch (event.key) {
        case 'Enter':
        case ' ':
            // Activate buttons and file inputs with Enter or Space
            if (element.classList.contains('gradio-button') || 
                element.classList.contains('gradio-file')) {
                event.preventDefault();
                element.click();
            }
            break;
            
        case 'Escape':
            // Close modals or accordions
            const accordion = element.closest('.gradio-accordion[open]');
            if (accordion) {
                accordion.removeAttribute('open');
            }
            break;
            
        case 'ArrowRight':
        case 'ArrowLeft':
            // Navigate between tabs
            if (element.closest('.gradio-tabs .tab-nav')) {
                event.preventDefault();
                navigateTabs(event.key === 'ArrowRight' ? 1 : -1);
            }
            break;
            
        case 'ArrowDown':
        case 'ArrowUp':
            // Navigate between preset options
            if (element.closest('#smart_preview_selector')) {
                event.preventDefault();
                navigatePresets(event.key === 'ArrowDown' ? 1 : -1);
            }
            break;
    }
}

function navigateTabs(direction) {
    const tabButtons = document.querySelectorAll('.gradio-tabs .tab-nav button');
    const currentTab = document.querySelector('.gradio-tabs .tab-nav button.selected');
    
    if (!currentTab || tabButtons.length === 0) return;
    
    const currentIndex = Array.from(tabButtons).indexOf(currentTab);
    let nextIndex = currentIndex + direction;
    
    if (nextIndex < 0) nextIndex = tabButtons.length - 1;
    if (nextIndex >= tabButtons.length) nextIndex = 0;
    
    tabButtons[nextIndex].focus();
    tabButtons[nextIndex].click();
}

function navigatePresets(direction) {
    const presetOptions = document.querySelectorAll('#smart_preview_selector input[type="radio"]');
    const checkedOption = document.querySelector('#smart_preview_selector input[type="radio"]:checked');
    
    if (!checkedOption || presetOptions.length === 0) return;
    
    const currentIndex = Array.from(presetOptions).indexOf(checkedOption);
    let nextIndex = currentIndex + direction;
    
    if (nextIndex < 0) nextIndex = presetOptions.length - 1;
    if (nextIndex >= presetOptions.length) nextIndex = 0;
    
    presetOptions[nextIndex].focus();
    presetOptions[nextIndex].click();
}

function addFocusIndicators() {
    // Add visual focus indicators for better accessibility
    const style = document.createElement('style');
    style.textContent = `
        .gradio-button:focus,
        .gradio-file:focus,
        .gradio-textbox:focus,
        .gradio-tabs .tab-nav button:focus,
        input:focus,
        select:focus,
        textarea:focus {
            outline: 2px solid #007bff !important;
            outline-offset: 2px !important;
            box-shadow: 0 0 0 4px rgba(0, 123, 255, 0.25) !important;
        }
        
        .gradio-button:focus:not(:focus-visible) {
            outline: none !important;
            box-shadow: none !important;
        }
    `;
    document.head.appendChild(style);
}

function setupTabNavigation() {
    // Improve tab order for logical navigation flow
    const tabOrder = [
        '#enhanced_source_file',
        '#enhanced_target_file', 
        '#smart_preview_selector',
        '#smart_preview_generate',
        '.gradio-button.primary',
        '.gradio-tabs .tab-nav button',
        '#progress_status'
    ];
    
    let tabIndex = 1;
    tabOrder.forEach(selector => {
        const elements = document.querySelectorAll(selector);
        elements.forEach(element => {
            element.setAttribute('tabindex', tabIndex++);
        });
    });
}

function setupKeyboardShortcuts() {
    // Global keyboard shortcuts
    document.addEventListener('keydown', function(event) {
        // Only trigger shortcuts when not typing in input fields
        if (event.target.tagName === 'INPUT' || 
            event.target.tagName === 'TEXTAREA' ||
            event.target.contentEditable === 'true') {
            return;
        }
        
        // Check for modifier keys
        const isCtrl = event.ctrlKey || event.metaKey;
        const isShift = event.shiftKey;
        
        if (isCtrl) {
            switch (event.key.toLowerCase()) {
                case '1':
                    event.preventDefault();
                    switchToTab(0); // Upload tab
                    break;
                case '2':
                    event.preventDefault();
                    switchToTab(1); // Preview tab
                    break;
                case '3':
                    event.preventDefault();
                    switchToTab(2); // Process tab
                    break;
                case '4':
                    event.preventDefault();
                    switchToTab(3); // Advanced tab
                    break;
                case 'g':
                    event.preventDefault();
                    const generateButton = document.querySelector('#smart_preview_generate');
                    if (generateButton) generateButton.click();
                    break;
                case 'p':
                    event.preventDefault();
                    const processButton = document.querySelector('.gradio-button.primary');
                    if (processButton) processButton.click();
                    break;
            }
        }
        
        // Quick preset selection (without Ctrl)
        switch (event.key) {
            case '1':
                if (!isCtrl) selectPreset('fast');
                break;
            case '2':
                if (!isCtrl) selectPreset('balanced');
                break;
            case '3':
                if (!isCtrl) selectPreset('quality');
                break;
            case '?':
                event.preventDefault();
                showKeyboardShortcuts();
                break;
        }
    });
}

function switchToTab(tabIndex) {
    const tabButtons = document.querySelectorAll('.gradio-tabs .tab-nav button');
    if (tabButtons[tabIndex]) {
        tabButtons[tabIndex].click();
        tabButtons[tabIndex].focus();
    }
}

function selectPreset(presetName) {
    const presetRadio = document.querySelector(
        `#smart_preview_selector input[value="${presetName}"]`
    );
    if (presetRadio) {
        presetRadio.click();
        presetRadio.focus();
    }
}

function showKeyboardShortcuts() {
    // Create and show keyboard shortcuts help modal
    const modal = document.createElement('div');
    modal.id = 'keyboard-shortcuts-modal';
    modal.innerHTML = `
        <div style="
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0, 0, 0, 0.5);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 10000;
        ">
            <div style="
                background: white;
                border-radius: 12px;
                padding: 24px;
                max-width: 500px;
                max-height: 80vh;
                overflow-y: auto;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            ">
                <h3 style="margin-top: 0; color: #333;">⌨️ Keyboard Shortcuts</h3>
                
                <div style="margin-bottom: 16px;">
                    <h4 style="color: #666; margin-bottom: 8px;">Navigation</h4>
                    <div style="font-family: monospace; font-size: 14px;">
                        <div><kbd>Ctrl+1</kbd> - Upload tab</div>
                        <div><kbd>Ctrl+2</kbd> - Preview tab</div>
                        <div><kbd>Ctrl+3</kbd> - Process tab</div>
                        <div><kbd>Ctrl+4</kbd> - Advanced tab</div>
                        <div><kbd>Tab</kbd> - Navigate between elements</div>
                        <div><kbd>↑↓</kbd> - Navigate presets</div>
                        <div><kbd>←→</kbd> - Navigate tabs</div>
                    </div>
                </div>
                
                <div style="margin-bottom: 16px;">
                    <h4 style="color: #666; margin-bottom: 8px;">Actions</h4>
                    <div style="font-family: monospace; font-size: 14px;">
                        <div><kbd>Ctrl+G</kbd> - Generate previews</div>
                        <div><kbd>Ctrl+P</kbd> - Start processing</div>
                        <div><kbd>Enter</kbd> - Activate button</div>
                        <div><kbd>Space</kbd> - Activate button</div>
                        <div><kbd>Escape</kbd> - Close accordion/modal</div>
                    </div>
                </div>
                
                <div style="margin-bottom: 20px;">
                    <h4 style="color: #666; margin-bottom: 8px;">Quick Preset Selection</h4>
                    <div style="font-family: monospace; font-size: 14px;">
                        <div><kbd>1</kbd> - Fast preset</div>
                        <div><kbd>2</kbd> - Balanced preset</div>
                        <div><kbd>3</kbd> - Quality preset</div>
                    </div>
                </div>
                
                <button onclick="document.getElementById('keyboard-shortcuts-modal').remove()" 
                        style="
                            background: #007bff;
                            color: white;
                            border: none;
                            padding: 8px 16px;
                            border-radius: 6px;
                            cursor: pointer;
                            float: right;
                        ">
                    Close
                </button>
                
                <div style="clear: both;"></div>
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
    
    // Close modal when clicking outside
    modal.addEventListener('click', function(event) {
        if (event.target === modal) {
            modal.remove();
        }
    });
    
    // Close modal with Escape key
    document.addEventListener('keydown', function escapeHandler(event) {
        if (event.key === 'Escape') {
            modal.remove();
            document.removeEventListener('keydown', escapeHandler);
        }
    });
}

// Screen reader announcements
function announceToScreenReader(message) {
    const announcement = document.createElement('div');
    announcement.setAttribute('aria-live', 'polite');
    announcement.setAttribute('aria-atomic', 'true');
    announcement.style.position = 'absolute';
    announcement.style.left = '-10000px';
    announcement.style.width = '1px';
    announcement.style.height = '1px';
    announcement.style.overflow = 'hidden';
    announcement.textContent = message;
    
    document.body.appendChild(announcement);
    
    setTimeout(() => {
        document.body.removeChild(announcement);
    }, 1000);
}

// Export functions for use by other components
window.FaceFusionA11y = {
    announceToScreenReader,
    showKeyboardShortcuts,
    switchToTab,
    selectPreset
};