"""
Recommendation algorithms
"""
from .knn import KNNRecommendationAlgorithm
from .popularity import PopularityRecommendationAlgorithm
from .analytics_popularity import AnalyticsPopularityAlgorithm

try:
    from .sequential_dagnn import SequentialDAGNNAlgorithm
except Exception as exc:  # noqa: BLE001 - keep app bootable if heavy deps are broken
    _SEQUENTIAL_IMPORT_ERROR = exc

    class SequentialDAGNNAlgorithm:  # type: ignore[no-redef]
        """Fallback stub when sequential dependencies are unavailable."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Sequential DAGNN dependencies are unavailable. "
                "Reinstall torch/torch-geometric/networkx to enable this algorithm."
            ) from _SEQUENTIAL_IMPORT_ERROR

__all__ = [
    "KNNRecommendationAlgorithm",
    "PopularityRecommendationAlgorithm",
    "AnalyticsPopularityAlgorithm",
    "SequentialDAGNNAlgorithm"
]
