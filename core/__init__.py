"""
Hand Gesture Platform - Core Module

A production-grade multi-platform hand-gesture intelligence platform.
"""

__version__ = "2.0.0"
__author__ = "R-Priyadarshi"

from core.vision.hand_detector import HandDetector
from core.landmarks.normalizer import LandmarkNormalizer
from core.inference.engine import InferenceEngine

__all__ = [
    "HandDetector",
    "LandmarkNormalizer",
    "InferenceEngine",
]
