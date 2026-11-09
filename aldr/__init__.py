"""
ALDR: AI-Driven Law & Road Monitoring
"""

__version__ = "1.0.0"

from aldr.detector import VehicleDetector, ModificationDetector
from aldr.reid_model import ReIDExtractor
from aldr.association import VehicleAssociator
from aldr.fusion import EvidenceFusionEngine
from aldr.reporting import ReportGenerator
from aldr.retrieval import VehicleGalleryRetriever
from aldr.pipeline import ALDRPipeline
