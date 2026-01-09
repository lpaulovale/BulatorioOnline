"""
Evaluation Package for PharmaBula

Tools for comparing RAG implementations.
"""

from .evaluator import (
    RAGEvaluator,
    QueryMetrics,
    FrameworkMetrics,
    EvaluationRun,
    run_full_evaluation,
)
from .dataset import TestDataset, SAMPLE_QUERIES, get_dataset_stats
from .visualizer import (
    generate_evaluation_dashboard,
    generate_thesis_figure,
    plot_latency_comparison,
    plot_cost_comparison,
    plot_quality_scores,
    plot_radar_comparison,
)

__all__ = [
    # Evaluator
    "RAGEvaluator",
    "QueryMetrics",
    "FrameworkMetrics",
    "EvaluationRun",
    "run_full_evaluation",
    # Dataset
    "TestDataset",
    "SAMPLE_QUERIES",
    "get_dataset_stats",
    # Visualizer
    "generate_evaluation_dashboard",
    "generate_thesis_figure",
    "plot_latency_comparison",
    "plot_cost_comparison",
    "plot_quality_scores",
    "plot_radar_comparison",
]
