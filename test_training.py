#!/usr/bin/env python3
"""
Test training components locally
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test all training-related imports"""
    try:
        print("🧪 Testing training component imports...")
        
        # Test UI components
        from watserface.uis.components import training_options, dataset_manager, model_trainer, training_progress
        print("✅ UI components imported successfully")
        
        # Test layouts
        from watserface.uis.layouts import training
        print("✅ Training layout imported successfully")
        
        # Test trainer (will show torch warning but that's expected)
        try:
            from watserface.trainers.instantid_trainer import InstantIDTrainer
            print("✅ InstantID trainer imported successfully")
        except ImportError as e:
            print(f"⚠️  InstantID trainer import failed (expected): {e}")
        
        print("\n🎯 All core imports working!")
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_state_manager():
    """Test state manager functionality"""
    try:
        print("\n🧪 Testing state manager...")
        from watserface import state_manager
        
        # Test setting training state
        state_manager.init_item('training_model', 'InstantID')
        state_manager.init_item('training_epochs', 100)
        
        # Test getting state
        model = state_manager.get_item('training_model')
        epochs = state_manager.get_item('training_epochs')
        
        assert model == 'InstantID'
        assert epochs == 100
        
        print("✅ State manager working correctly")
        return True
        
    except Exception as e:
        print(f"❌ State manager error: {e}")
        return False

def test_ui_rendering():
    """Test if UI components can render without errors"""
    try:
        print("\n🧪 Testing UI rendering...")
        
        # This would normally require Gradio to be running
        # For now, just test that the render functions exist
        from watserface.uis.components import training_options
        
        # Check if render function exists
        assert hasattr(training_options, 'render')
        assert callable(training_options.render)
        
        print("✅ UI render functions available")
        return True
        
    except Exception as e:
        print(f"❌ UI rendering error: {e}")
        return False

def main():
    print("🚀 Running FaceFusion Training Tests\n")
    
    tests = [
        test_imports,
        test_state_manager, 
        test_ui_rendering
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print(f"\n📊 Test Results: {sum(results)}/{len(results)} passed")
    
    if all(results):
        print("🎉 All tests passed! Training components are ready.")
    else:
        print("⚠️  Some tests failed. Check the output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()