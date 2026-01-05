"""Boundary zone detection for diffusion inpainting."""
from typing import Optional, Tuple
import cv2
import numpy

from watserface.types import VisionFrame


class BoundaryDetector:
    
    def __init__(
        self,
        dilation_size: int = 15,
        erosion_size: int = 5,
        blur_size: int = 7
    ):
        self.dilation_size = dilation_size
        self.erosion_size = erosion_size
        self.blur_size = blur_size
    
    def create_boundary_zone(
        self,
        mask: numpy.ndarray,
        zone_width: int = 20
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        
        mask_binary = (mask > 127).astype(numpy.uint8) * 255
        
        dilate_kernel = numpy.ones((self.dilation_size, self.dilation_size), numpy.uint8)
        dilated = cv2.dilate(mask_binary, dilate_kernel, iterations=1)
        
        erode_kernel = numpy.ones((self.erosion_size, self.erosion_size), numpy.uint8)
        eroded = cv2.erode(mask_binary, erode_kernel, iterations=1)
        
        boundary = cv2.bitwise_xor(dilated, eroded)
        
        boundary_float = boundary.astype(numpy.float32) / 255.0
        boundary_smooth = cv2.GaussianBlur(
            boundary_float,
            (self.blur_size, self.blur_size),
            0
        )
        
        return boundary_smooth, mask_binary
    
    def create_inpaint_mask(
        self,
        xseg_mask: numpy.ndarray,
        face_mask: numpy.ndarray,
        boundary_expansion: int = 10
    ) -> numpy.ndarray:
        if len(xseg_mask.shape) == 3:
            xseg_mask = cv2.cvtColor(xseg_mask, cv2.COLOR_RGB2GRAY)
        if len(face_mask.shape) == 3:
            face_mask = cv2.cvtColor(face_mask, cv2.COLOR_RGB2GRAY)
        
        occlusion_region = cv2.bitwise_and(xseg_mask, face_mask)
        
        expand_kernel = numpy.ones(
            (boundary_expansion, boundary_expansion),
            numpy.uint8
        )
        expanded_occlusion = cv2.dilate(occlusion_region, expand_kernel, iterations=1)
        
        inpaint_mask = cv2.bitwise_and(expanded_occlusion, face_mask)
        
        return inpaint_mask
    
    def create_feathered_blend_mask(
        self,
        mask: numpy.ndarray,
        feather_amount: int = 10
    ) -> numpy.ndarray:
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        
        distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        
        max_dist = min(feather_amount, distance.max())
        if max_dist > 0:
            feathered = numpy.clip(distance / max_dist, 0, 1)
        else:
            feathered = (mask > 0).astype(numpy.float32)
        
        return feathered.astype(numpy.float32)
    
    def detect_occlusion_edges(
        self,
        frame: VisionFrame,
        mask: numpy.ndarray
    ) -> numpy.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        edges = cv2.Canny(gray, 50, 150)
        
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        
        boundary, _ = self.create_boundary_zone(mask)
        boundary_uint8 = (boundary * 255).astype(numpy.uint8)
        
        occlusion_edges = cv2.bitwise_and(edges, boundary_uint8)
        
        kernel = numpy.ones((3, 3), numpy.uint8)
        occlusion_edges = cv2.dilate(occlusion_edges, kernel, iterations=1)
        
        return occlusion_edges


def create_boundary_detector(
    dilation_size: int = 15,
    erosion_size: int = 5
) -> BoundaryDetector:
    return BoundaryDetector(
        dilation_size=dilation_size,
        erosion_size=erosion_size
    )
