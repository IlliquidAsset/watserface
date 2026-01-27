"""Tests for ControlNet hyperparameter optimizer."""
import numpy
import pytest
from unittest.mock import Mock, MagicMock


class TestControlNetOptimizerInitialization:
    """Test optimizer initialization."""
    
    def test_class_exists(self):
        from watserface.inpainting.controlnet_optimizer import ControlNetOptimizer
        assert ControlNetOptimizer is not None
    
    def test_initializes_with_defaults(self):
        from watserface.inpainting.controlnet_optimizer import ControlNetOptimizer
        optimizer = ControlNetOptimizer()
        assert optimizer is not None
        assert optimizer.config is not None
        assert optimizer.results == []
    
    def test_accepts_custom_config(self):
        from watserface.inpainting.controlnet_optimizer import (
            ControlNetOptimizer, OptimizationConfig
        )
        config = OptimizationConfig(
            conditioning_scale_range=(0.4, 0.8),
            conditioning_scale_steps=5
        )
        optimizer = ControlNetOptimizer(config=config)
        assert optimizer.config.conditioning_scale_range == (0.4, 0.8)
        assert optimizer.config.conditioning_scale_steps == 5


class TestParameterGridGeneration:
    """Test parameter grid generation."""
    
    def test_generates_grid(self):
        from watserface.inpainting.controlnet_optimizer import ControlNetOptimizer
        optimizer = ControlNetOptimizer()
        grid = optimizer.generate_parameter_grid()
        assert len(grid) > 0
    
    def test_grid_has_required_keys(self):
        from watserface.inpainting.controlnet_optimizer import ControlNetOptimizer
        optimizer = ControlNetOptimizer()
        grid = optimizer.generate_parameter_grid()
        
        required_keys = [
            'conditioning_scale', 'guidance_start', 'guidance_end',
            'strength', 'num_inference_steps'
        ]
        for params in grid:
            for key in required_keys:
                assert key in params
    
    def test_grid_respects_config_range(self):
        from watserface.inpainting.controlnet_optimizer import (
            ControlNetOptimizer, OptimizationConfig
        )
        config = OptimizationConfig(
            conditioning_scale_range=(0.5, 0.7),
            conditioning_scale_steps=3
        )
        optimizer = ControlNetOptimizer(config=config)
        grid = optimizer.generate_parameter_grid()
        
        scales = [p['conditioning_scale'] for p in grid]
        assert min(scales) >= 0.5
        assert max(scales) <= 0.7


class TestSSIMComputation:
    """Test SSIM metric computation."""
    
    def test_ssim_identical_images(self):
        from watserface.inpainting.controlnet_optimizer import ControlNetOptimizer
        optimizer = ControlNetOptimizer()
        
        img = numpy.random.randint(0, 255, (100, 100, 3), dtype=numpy.uint8)
        ssim = optimizer.compute_ssim(img, img.copy())
        
        assert ssim > 0.99
    
    def test_ssim_different_images(self):
        from watserface.inpainting.controlnet_optimizer import ControlNetOptimizer
        optimizer = ControlNetOptimizer()
        
        img1 = numpy.zeros((100, 100, 3), dtype=numpy.uint8)
        img2 = numpy.ones((100, 100, 3), dtype=numpy.uint8) * 255
        
        ssim = optimizer.compute_ssim(img1, img2)
        assert ssim < 0.5
    
    def test_ssim_returns_float(self):
        from watserface.inpainting.controlnet_optimizer import ControlNetOptimizer
        optimizer = ControlNetOptimizer()
        
        img = numpy.random.randint(0, 255, (50, 50, 3), dtype=numpy.uint8)
        ssim = optimizer.compute_ssim(img, img)
        
        assert isinstance(ssim, float)
        assert 0 <= ssim <= 1


class TestParameterEvaluation:
    """Test parameter evaluation."""
    
    def test_evaluate_returns_result(self):
        from watserface.inpainting.controlnet_optimizer import (
            ControlNetOptimizer, OptimizationResult
        )
        optimizer = ControlNetOptimizer()
        
        mock_pipeline = MagicMock()
        mock_pipeline.process.return_value = numpy.zeros((100, 100, 3), dtype=numpy.uint8)
        
        test_img = numpy.zeros((100, 100, 3), dtype=numpy.uint8)
        ref_img = numpy.zeros((100, 100, 3), dtype=numpy.uint8)
        params = {
            'conditioning_scale': 0.75,
            'guidance_start': 0.0,
            'guidance_end': 1.0,
            'strength': 0.75,
            'num_inference_steps': 30
        }
        
        result = optimizer.evaluate_parameters(mock_pipeline, test_img, ref_img, params)
        
        assert isinstance(result, OptimizationResult)
        assert result.conditioning_scale == 0.75
        assert result.num_inference_steps == 30


class TestReportGeneration:
    """Test ablation study report generation."""
    
    def test_generates_report(self):
        from watserface.inpainting.controlnet_optimizer import (
            ControlNetOptimizer, OptimizationResult
        )
        optimizer = ControlNetOptimizer()
        
        optimizer.results = [
            OptimizationResult(0.5, 0.0, 1.0, 0.75, 30, 0.85, 2.0),
            OptimizationResult(0.75, 0.0, 1.0, 0.75, 30, 0.90, 2.5),
        ]
        optimizer.best_result = optimizer.results[1]
        
        report = optimizer.generate_report()
        
        assert 'summary' in report
        assert 'all_results' in report
        assert 'recommendation' in report
    
    def test_report_identifies_best(self):
        from watserface.inpainting.controlnet_optimizer import (
            ControlNetOptimizer, OptimizationResult
        )
        optimizer = ControlNetOptimizer()
        
        optimizer.results = [
            OptimizationResult(0.5, 0.0, 1.0, 0.75, 30, 0.70, 2.0),
            OptimizationResult(0.75, 0.0, 1.0, 0.75, 30, 0.95, 2.5),
            OptimizationResult(0.9, 0.0, 1.0, 0.75, 30, 0.80, 3.0),
        ]
        optimizer.best_result = max(optimizer.results, key=lambda r: r.ssim_score)
        
        report = optimizer.generate_report()
        
        assert report['summary']['best_ssim'] == 0.95
        assert report['summary']['best_configuration']['conditioning_scale'] == 0.75


class TestQuickOptimize:
    """Test quick optimization mode."""
    
    def test_quick_optimize_returns_params(self):
        from watserface.inpainting.controlnet_optimizer import ControlNetOptimizer
        optimizer = ControlNetOptimizer()
        
        mock_pipeline = MagicMock()
        mock_pipeline.process.return_value = numpy.zeros((100, 100, 3), dtype=numpy.uint8)
        
        test_img = numpy.zeros((100, 100, 3), dtype=numpy.uint8)
        ref_img = numpy.zeros((100, 100, 3), dtype=numpy.uint8)
        
        params = optimizer.quick_optimize(mock_pipeline, test_img, ref_img)
        
        assert 'conditioning_scale' in params
        assert 'num_inference_steps' in params


class TestFactoryFunctions:
    """Test factory and convenience functions."""
    
    def test_create_optimizer_exists(self):
        from watserface.inpainting.controlnet_optimizer import create_optimizer
        assert create_optimizer is not None
    
    def test_create_optimizer_returns_instance(self):
        from watserface.inpainting.controlnet_optimizer import (
            create_optimizer, ControlNetOptimizer
        )
        optimizer = create_optimizer()
        assert isinstance(optimizer, ControlNetOptimizer)
    
    def test_compute_ssim_function_exists(self):
        from watserface.inpainting.controlnet_optimizer import compute_ssim
        assert compute_ssim is not None
    
    def test_compute_ssim_works(self):
        from watserface.inpainting.controlnet_optimizer import compute_ssim
        
        img = numpy.random.randint(0, 255, (50, 50, 3), dtype=numpy.uint8)
        ssim = compute_ssim(img, img)
        
        assert ssim > 0.99
