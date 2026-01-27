"""ControlNet hyperparameter optimization for face swapping quality."""
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy
import cv2
import json
import time
from pathlib import Path


@dataclass
class OptimizationConfig:
    conditioning_scale_range: Tuple[float, float] = (0.3, 0.9)
    conditioning_scale_steps: int = 7
    guidance_start_range: Tuple[float, float] = (0.0, 0.3)
    guidance_end_range: Tuple[float, float] = (0.7, 1.0)
    strength_range: Tuple[float, float] = (0.5, 0.9)
    num_inference_steps_options: List[int] = field(default_factory=lambda: [20, 30, 50])


@dataclass
class OptimizationResult:
    conditioning_scale: float
    guidance_start: float
    guidance_end: float
    strength: float
    num_inference_steps: int
    ssim_score: float
    processing_time: float
    visual_quality_notes: str = ""


class ControlNetOptimizer:
    """Hyperparameter optimizer for ControlNet face swapping.
    
    Conducts ablation studies to find optimal parameters for
    face geometry preservation and artifact reduction.
    """
    
    DEFAULT_CONFIG = OptimizationConfig()
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or self.DEFAULT_CONFIG
        self.results: List[OptimizationResult] = []
        self.best_result: Optional[OptimizationResult] = None
    
    def generate_parameter_grid(self) -> List[Dict[str, Any]]:
        """Generate parameter combinations for grid search."""
        scales = numpy.linspace(
            self.config.conditioning_scale_range[0],
            self.config.conditioning_scale_range[1],
            self.config.conditioning_scale_steps
        )
        
        grid = []
        for scale in scales:
            for steps in self.config.num_inference_steps_options:
                grid.append({
                    'conditioning_scale': float(scale),
                    'guidance_start': 0.0,
                    'guidance_end': 1.0,
                    'strength': 0.75,
                    'num_inference_steps': steps
                })
        
        return grid
    
    def compute_ssim(
        self,
        img1: numpy.ndarray,
        img2: numpy.ndarray
    ) -> float:
        """Compute Structural Similarity Index between two images."""
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY) if len(img1.shape) == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY) if len(img2.shape) == 3 else img2
        
        gray1 = gray1.astype(numpy.float64)
        gray2 = gray2.astype(numpy.float64)
        
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        mu1 = cv2.GaussianBlur(gray1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(gray2, (11, 11), 1.5)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = cv2.GaussianBlur(gray1 ** 2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(gray2 ** 2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(gray1 * gray2, (11, 11), 1.5) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return float(numpy.mean(ssim_map))
    
    def evaluate_parameters(
        self,
        pipeline: Any,
        test_image: numpy.ndarray,
        reference_image: numpy.ndarray,
        params: Dict[str, Any]
    ) -> OptimizationResult:
        """Evaluate a single parameter combination."""
        start_time = time.time()
        
        try:
            if hasattr(pipeline, 'controlnet_conditioning_scale'):
                pipeline.controlnet_conditioning_scale = params['conditioning_scale']
            
            result_image = pipeline.process(
                test_image,
                num_inference_steps=params['num_inference_steps']
            )
            
            processing_time = time.time() - start_time
            ssim = self.compute_ssim(result_image, reference_image)
            
        except Exception as e:
            processing_time = time.time() - start_time
            ssim = 0.0
        
        return OptimizationResult(
            conditioning_scale=params['conditioning_scale'],
            guidance_start=params['guidance_start'],
            guidance_end=params['guidance_end'],
            strength=params['strength'],
            num_inference_steps=params['num_inference_steps'],
            ssim_score=ssim,
            processing_time=processing_time
        )
    
    def run_ablation_study(
        self,
        pipeline: Any,
        test_images: List[numpy.ndarray],
        reference_images: List[numpy.ndarray],
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Run full ablation study across parameter grid."""
        parameter_grid = self.generate_parameter_grid()
        self.results = []
        
        for params in parameter_grid:
            param_scores = []
            param_times = []
            
            for test_img, ref_img in zip(test_images, reference_images):
                result = self.evaluate_parameters(pipeline, test_img, ref_img, params)
                param_scores.append(result.ssim_score)
                param_times.append(result.processing_time)
            
            avg_result = OptimizationResult(
                conditioning_scale=params['conditioning_scale'],
                guidance_start=params['guidance_start'],
                guidance_end=params['guidance_end'],
                strength=params['strength'],
                num_inference_steps=params['num_inference_steps'],
                ssim_score=float(numpy.mean(param_scores)),
                processing_time=float(numpy.mean(param_times))
            )
            self.results.append(avg_result)
        
        self.best_result = max(self.results, key=lambda r: r.ssim_score)
        
        report = self.generate_report()
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(output_dir / 'ablation_results.json', 'w') as f:
                json.dump(report, f, indent=2)
        
        return report
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate ablation study report."""
        if not self.results:
            return {'error': 'No results available'}
        
        results_data = [
            {
                'conditioning_scale': r.conditioning_scale,
                'guidance_start': r.guidance_start,
                'guidance_end': r.guidance_end,
                'strength': r.strength,
                'num_inference_steps': r.num_inference_steps,
                'ssim_score': r.ssim_score,
                'processing_time': r.processing_time
            }
            for r in self.results
        ]
        
        scale_analysis = {}
        for r in self.results:
            scale_key = f"{r.conditioning_scale:.2f}"
            if scale_key not in scale_analysis:
                scale_analysis[scale_key] = []
            scale_analysis[scale_key].append(r.ssim_score)
        
        scale_averages = {k: float(numpy.mean(v)) for k, v in scale_analysis.items()}
        
        return {
            'summary': {
                'total_configurations': len(self.results),
                'best_ssim': self.best_result.ssim_score if self.best_result else 0,
                'best_configuration': {
                    'conditioning_scale': self.best_result.conditioning_scale,
                    'num_inference_steps': self.best_result.num_inference_steps,
                    'processing_time': self.best_result.processing_time
                } if self.best_result else None
            },
            'scale_analysis': scale_averages,
            'all_results': results_data,
            'recommendation': self._generate_recommendation()
        }
    
    def _generate_recommendation(self) -> str:
        """Generate optimization recommendation based on results."""
        if not self.best_result:
            return "No optimization results available."
        
        return (
            f"Optimal configuration: conditioning_scale={self.best_result.conditioning_scale:.2f}, "
            f"num_inference_steps={self.best_result.num_inference_steps}. "
            f"Achieved SSIM={self.best_result.ssim_score:.4f} with "
            f"processing time={self.best_result.processing_time:.2f}s."
        )
    
    def quick_optimize(
        self,
        pipeline: Any,
        test_image: numpy.ndarray,
        reference_image: numpy.ndarray
    ) -> Dict[str, float]:
        """Quick optimization with reduced parameter space."""
        quick_scales = [0.5, 0.65, 0.75, 0.85]
        quick_steps = [20, 30]
        
        best_ssim = 0.0
        best_params = {'conditioning_scale': 0.75, 'num_inference_steps': 30}
        
        for scale in quick_scales:
            for steps in quick_steps:
                params = {
                    'conditioning_scale': scale,
                    'guidance_start': 0.0,
                    'guidance_end': 1.0,
                    'strength': 0.75,
                    'num_inference_steps': steps
                }
                
                result = self.evaluate_parameters(
                    pipeline, test_image, reference_image, params
                )
                
                if result.ssim_score > best_ssim:
                    best_ssim = result.ssim_score
                    best_params = {
                        'conditioning_scale': scale,
                        'num_inference_steps': steps
                    }
        
        return best_params


def create_optimizer(config: Optional[OptimizationConfig] = None) -> ControlNetOptimizer:
    """Factory function for ControlNetOptimizer."""
    return ControlNetOptimizer(config)


def compute_ssim(img1: numpy.ndarray, img2: numpy.ndarray) -> float:
    """Convenience function for SSIM computation."""
    optimizer = ControlNetOptimizer()
    return optimizer.compute_ssim(img1, img2)
