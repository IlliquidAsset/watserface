"""Quality checking and comparison metrics for face swap output evaluation."""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy

from watserface.types import VisionFrame


@dataclass
class QualityMetrics:
    ssim: float
    psnr: float
    identity_similarity: float
    blur_score: float
    artifact_score: float
    overall_score: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'ssim': self.ssim,
            'psnr': self.psnr,
            'identity_similarity': self.identity_similarity,
            'blur_score': self.blur_score,
            'artifact_score': self.artifact_score,
            'overall_score': self.overall_score
        }


class QualityChecker:
    
    def __init__(self, quality_threshold: float = 0.7):
        self.quality_threshold = quality_threshold
        
        self.ssim_weight = 0.25
        self.psnr_weight = 0.15
        self.identity_weight = 0.30
        self.blur_weight = 0.15
        self.artifact_weight = 0.15
    
    def compute_ssim(
        self,
        image1: VisionFrame,
        image2: VisionFrame,
        win_size: int = 7
    ) -> float:
        if image1.shape != image2.shape:
            image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
        
        gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY).astype(numpy.float64)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY).astype(numpy.float64)
        
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        kernel = cv2.getGaussianKernel(win_size, 1.5)
        window = numpy.outer(kernel, kernel)
        
        mu1 = cv2.filter2D(gray1, -1, window)
        mu2 = cv2.filter2D(gray2, -1, window)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = cv2.filter2D(gray1 ** 2, -1, window) - mu1_sq
        sigma2_sq = cv2.filter2D(gray2 ** 2, -1, window) - mu2_sq
        sigma12 = cv2.filter2D(gray1 * gray2, -1, window) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return float(numpy.mean(ssim_map))
    
    def compute_psnr(
        self,
        image1: VisionFrame,
        image2: VisionFrame
    ) -> float:
        if image1.shape != image2.shape:
            image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
        
        mse = numpy.mean((image1.astype(numpy.float64) - image2.astype(numpy.float64)) ** 2)
        
        if mse == 0:
            return 100.0
        
        max_pixel = 255.0
        psnr = 20 * numpy.log10(max_pixel / numpy.sqrt(mse))
        
        return float(numpy.clip(psnr / 50.0, 0, 1))
    
    def compute_identity_similarity(
        self,
        source_embedding: Optional[numpy.ndarray],
        result_embedding: Optional[numpy.ndarray]
    ) -> float:
        if source_embedding is None or result_embedding is None:
            return 0.5
        
        source_norm = source_embedding / (numpy.linalg.norm(source_embedding) + 1e-8)
        result_norm = result_embedding / (numpy.linalg.norm(result_embedding) + 1e-8)
        
        similarity = numpy.dot(source_norm.flatten(), result_norm.flatten())
        
        return float((similarity + 1) / 2)
    
    def compute_blur_score(self, image: VisionFrame) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        return float(min(1.0, laplacian_var / 500))
    
    def compute_artifact_score(self, image: VisionFrame) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = numpy.sqrt(sobelx ** 2 + sobely ** 2)
        
        mean_gradient = numpy.mean(gradient)
        std_gradient = numpy.std(gradient)
        
        cv_gradient = std_gradient / (mean_gradient + 1e-8)
        
        artifact_penalty = min(1.0, cv_gradient / 2.0)
        
        return float(1.0 - artifact_penalty * 0.5)
    
    def compute_metrics(
        self,
        source_frame: VisionFrame,
        target_frame: VisionFrame,
        result_frame: VisionFrame,
        source_embedding: Optional[numpy.ndarray] = None,
        result_embedding: Optional[numpy.ndarray] = None
    ) -> QualityMetrics:
        ssim = self.compute_ssim(target_frame, result_frame)
        psnr = self.compute_psnr(target_frame, result_frame)
        identity_sim = self.compute_identity_similarity(source_embedding, result_embedding)
        blur = self.compute_blur_score(result_frame)
        artifact = self.compute_artifact_score(result_frame)
        
        overall = (
            ssim * self.ssim_weight +
            psnr * self.psnr_weight +
            identity_sim * self.identity_weight +
            blur * self.blur_weight +
            artifact * self.artifact_weight
        )
        
        return QualityMetrics(
            ssim=ssim,
            psnr=psnr,
            identity_similarity=identity_sim,
            blur_score=blur,
            artifact_score=artifact,
            overall_score=overall
        )
    
    def is_quality_acceptable(self, metrics: QualityMetrics) -> bool:
        return metrics.overall_score >= self.quality_threshold
    
    def create_comparison_view(
        self,
        source_frame: VisionFrame,
        target_frame: VisionFrame,
        result_frame: VisionFrame,
        layout: str = 'horizontal'
    ) -> VisionFrame:
        h, w = target_frame.shape[:2]
        
        source_resized = cv2.resize(source_frame, (w, h))
        result_resized = cv2.resize(result_frame, (w, h)) if result_frame.shape[:2] != (h, w) else result_frame
        
        if layout == 'horizontal':
            comparison = numpy.hstack([source_resized, target_frame, result_resized])
        elif layout == 'vertical':
            comparison = numpy.vstack([source_resized, target_frame, result_resized])
        elif layout == 'grid':
            diff = cv2.absdiff(target_frame, result_resized)
            top_row = numpy.hstack([source_resized, target_frame])
            bottom_row = numpy.hstack([result_resized, diff])
            comparison = numpy.vstack([top_row, bottom_row])
        else:
            comparison = numpy.hstack([source_resized, target_frame, result_resized])
        
        return comparison
    
    def create_slider_comparison(
        self,
        frame1: VisionFrame,
        frame2: VisionFrame,
        slider_position: float = 0.5
    ) -> VisionFrame:
        h, w = frame1.shape[:2]
        
        if frame2.shape[:2] != (h, w):
            frame2 = cv2.resize(frame2, (w, h))
        
        split_x = int(w * slider_position)
        
        result = frame1.copy()
        result[:, split_x:] = frame2[:, split_x:]
        
        cv2.line(result, (split_x, 0), (split_x, h), (255, 255, 255), 2)
        
        return result
    
    def create_difference_heatmap(
        self,
        frame1: VisionFrame,
        frame2: VisionFrame
    ) -> VisionFrame:
        if frame2.shape[:2] != frame1.shape[:2]:
            frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
        
        diff = cv2.absdiff(frame1, frame2)
        
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        
        heatmap = cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)
        
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        return heatmap_rgb
    
    def get_quality_report(
        self,
        metrics: QualityMetrics
    ) -> str:
        status = 'PASS' if self.is_quality_acceptable(metrics) else 'FAIL'
        
        report = f"""Quality Report ({status})
{'=' * 40}
SSIM:                {metrics.ssim:.3f}
PSNR (normalized):   {metrics.psnr:.3f}
Identity Similarity: {metrics.identity_similarity:.3f}
Blur Score:          {metrics.blur_score:.3f}
Artifact Score:      {metrics.artifact_score:.3f}
{'=' * 40}
Overall Score:       {metrics.overall_score:.3f}
Threshold:           {self.quality_threshold:.3f}
"""
        return report


class AutoQualityFilter:
    
    def __init__(
        self,
        min_blur_score: float = 0.3,
        min_identity_score: float = 0.5,
        min_overall_score: float = 0.6,
        max_artifact_penalty: float = 0.4
    ):
        self.min_blur_score = min_blur_score
        self.min_identity_score = min_identity_score
        self.min_overall_score = min_overall_score
        self.max_artifact_penalty = max_artifact_penalty
        self.checker = QualityChecker(quality_threshold=min_overall_score)
        
        self.accepted_count = 0
        self.rejected_count = 0
        self.rejection_reasons: Dict[str, int] = {
            'blur': 0,
            'identity': 0,
            'artifact': 0,
            'overall': 0
        }
    
    def evaluate_frame(
        self,
        source_frame: VisionFrame,
        target_frame: VisionFrame,
        result_frame: VisionFrame,
        source_embedding: Optional[numpy.ndarray] = None,
        result_embedding: Optional[numpy.ndarray] = None
    ) -> Tuple[bool, QualityMetrics, List[str]]:
        metrics = self.checker.compute_metrics(
            source_frame, target_frame, result_frame,
            source_embedding, result_embedding
        )
        
        rejection_reasons = []
        
        if metrics.blur_score < self.min_blur_score:
            rejection_reasons.append(f'blur ({metrics.blur_score:.2f} < {self.min_blur_score})')
            self.rejection_reasons['blur'] += 1
        
        if metrics.identity_similarity < self.min_identity_score:
            rejection_reasons.append(f'identity ({metrics.identity_similarity:.2f} < {self.min_identity_score})')
            self.rejection_reasons['identity'] += 1
        
        artifact_penalty = 1.0 - metrics.artifact_score
        if artifact_penalty > self.max_artifact_penalty:
            rejection_reasons.append(f'artifacts ({artifact_penalty:.2f} > {self.max_artifact_penalty})')
            self.rejection_reasons['artifact'] += 1
        
        if metrics.overall_score < self.min_overall_score:
            rejection_reasons.append(f'overall ({metrics.overall_score:.2f} < {self.min_overall_score})')
            self.rejection_reasons['overall'] += 1
        
        accepted = len(rejection_reasons) == 0
        
        if accepted:
            self.accepted_count += 1
        else:
            self.rejected_count += 1
        
        return accepted, metrics, rejection_reasons
    
    def filter_frames(
        self,
        frames: List[Tuple[VisionFrame, VisionFrame, VisionFrame]],
        embeddings: Optional[List[Tuple[numpy.ndarray, numpy.ndarray]]] = None
    ) -> Tuple[List[int], List[int]]:
        accepted_indices = []
        rejected_indices = []
        
        for i, (source, target, result) in enumerate(frames):
            source_emb = None
            result_emb = None
            
            if embeddings and i < len(embeddings):
                source_emb, result_emb = embeddings[i]
            
            accepted, _, _ = self.evaluate_frame(
                source, target, result, source_emb, result_emb
            )
            
            if accepted:
                accepted_indices.append(i)
            else:
                rejected_indices.append(i)
        
        return accepted_indices, rejected_indices
    
    def get_statistics(self) -> Dict[str, Any]:
        total = self.accepted_count + self.rejected_count
        acceptance_rate = self.accepted_count / total if total > 0 else 0
        
        return {
            'total_processed': total,
            'accepted': self.accepted_count,
            'rejected': self.rejected_count,
            'acceptance_rate': acceptance_rate,
            'rejection_breakdown': self.rejection_reasons.copy()
        }
    
    def reset_statistics(self) -> None:
        self.accepted_count = 0
        self.rejected_count = 0
        self.rejection_reasons = {
            'blur': 0,
            'identity': 0,
            'artifact': 0,
            'overall': 0
        }
    
    def adjust_thresholds(
        self,
        target_acceptance_rate: float = 0.8,
        adjustment_factor: float = 0.05
    ) -> None:
        stats = self.get_statistics()
        current_rate = stats['acceptance_rate']
        
        if current_rate < target_acceptance_rate:
            self.min_blur_score = max(0.1, self.min_blur_score - adjustment_factor)
            self.min_identity_score = max(0.3, self.min_identity_score - adjustment_factor)
            self.min_overall_score = max(0.4, self.min_overall_score - adjustment_factor)
            self.max_artifact_penalty = min(0.6, self.max_artifact_penalty + adjustment_factor)
        elif current_rate > target_acceptance_rate + 0.1:
            self.min_blur_score = min(0.6, self.min_blur_score + adjustment_factor)
            self.min_identity_score = min(0.8, self.min_identity_score + adjustment_factor)
            self.min_overall_score = min(0.85, self.min_overall_score + adjustment_factor)
            self.max_artifact_penalty = max(0.2, self.max_artifact_penalty - adjustment_factor)


def create_quality_checker(threshold: float = 0.7) -> QualityChecker:
    return QualityChecker(quality_threshold=threshold)


def create_auto_quality_filter(
    min_blur: float = 0.3,
    min_identity: float = 0.5,
    min_overall: float = 0.6
) -> AutoQualityFilter:
    return AutoQualityFilter(
        min_blur_score=min_blur,
        min_identity_score=min_identity,
        min_overall_score=min_overall
    )
