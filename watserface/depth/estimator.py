from typing import Any, Optional, Tuple
import numpy
import cv2

from watserface.types import VisionFrame


class DepthEstimator:
    
    def __init__(self, model_type: str = 'midas_small'):
        self.model_type = model_type
        self.model = None
        self.transform = None
        self.device = None
    
    def load(self) -> bool:
        try:
            import torch
            self.device = torch.device('mps' if torch.backends.mps.is_available() else 
                                       'cuda' if torch.cuda.is_available() else 'cpu')
            
            if self.model_type == 'midas_small':
                self.model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', trust_repo=True)
                midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms', trust_repo=True)
                self.transform = midas_transforms.small_transform
            elif self.model_type == 'midas_large':
                self.model = torch.hub.load('intel-isl/MiDaS', 'DPT_Large', trust_repo=True)
                midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms', trust_repo=True)
                self.transform = midas_transforms.dpt_transform
            else:
                self.model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', trust_repo=True)
                midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms', trust_repo=True)
                self.transform = midas_transforms.small_transform
            
            self.model.to(self.device)
            self.model.eval()
            return True
        except Exception as e:
            print(f'Failed to load depth model: {e}')
            return False
    
    def estimate(self, frame: VisionFrame) -> VisionFrame:
        if self.model is None:
            if not self.load():
                return numpy.zeros(frame.shape[:2], dtype=numpy.float32)
        
        import torch
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(rgb_frame).to(self.device)
        
        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode='bicubic',
                align_corners=False
            ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        if depth_max - depth_min > 0:
            depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        
        return depth_map.astype(numpy.float32)
    
    def estimate_with_mask(self, frame: VisionFrame, mask: VisionFrame) -> Tuple[VisionFrame, VisionFrame]:
        depth_map = self.estimate(frame)
        
        if mask is not None and mask.shape[:2] == depth_map.shape[:2]:
            mask_normalized = mask.astype(numpy.float32)
            if mask_normalized.max() > 1:
                mask_normalized = mask_normalized / 255.0
            
            masked_depth = depth_map * mask_normalized
            return depth_map, masked_depth
        
        return depth_map, depth_map
    
    def detect_translucent_regions(self, depth_map: VisionFrame, 
                                    frame: VisionFrame,
                                    variance_threshold: float = 0.1) -> VisionFrame:
        kernel_size = 5
        depth_variance = cv2.blur((depth_map ** 2), (kernel_size, kernel_size)) - \
                        cv2.blur(depth_map, (kernel_size, kernel_size)) ** 2
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(numpy.float32) / 255.0
        color_variance = cv2.blur((gray ** 2), (kernel_size, kernel_size)) - \
                        cv2.blur(gray, (kernel_size, kernel_size)) ** 2
        
        translucent_mask = numpy.zeros_like(depth_map)
        high_depth_var = depth_variance > variance_threshold
        low_color_var = color_variance < variance_threshold * 0.5
        
        translucent_mask[high_depth_var & low_color_var] = 1.0
        
        kernel = numpy.ones((3, 3), numpy.uint8)
        translucent_mask = cv2.morphologyEx(translucent_mask, cv2.MORPH_CLOSE, kernel)
        translucent_mask = cv2.morphologyEx(translucent_mask, cv2.MORPH_OPEN, kernel)
        
        return translucent_mask


_estimator: Optional[DepthEstimator] = None


def estimate_depth(frame: VisionFrame, model_type: str = 'midas_small') -> VisionFrame:
    global _estimator
    
    if _estimator is None or _estimator.model_type != model_type:
        _estimator = DepthEstimator(model_type)
    
    return _estimator.estimate(frame)
