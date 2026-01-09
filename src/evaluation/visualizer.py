"""
Visualization Module for PharmaBula Evaluation Results

Generates charts and visualizations for TCC comparison analysis:
- Bar charts for metrics comparison
- Radar charts for multi-dimensional analysis
- Line charts for latency distribution
- Export to PNG for thesis inclusion
"""

import json
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

from src.evaluation.evaluator import EvaluationRun, FrameworkMetrics

logger = logging.getLogger(__name__)

# ============================================================
# Chart Style Configuration
# ============================================================

# Color palette for frameworks
FRAMEWORK_COLORS = {
    "gemini": "#4285F4",      # Google Blue
    "langchain": "#00A67E",   # LangChain Green
    "openai": "#10A37F",      # OpenAI Teal
    "anthropic": "#D97706"    # Anthropic Orange
}

# Default fallback colors
DEFAULT_COLORS = ["#8B5CF6", "#06B6D4", "#10B981", "#F59E0B", "#EF4444"]


def setup_style():
    """Setup matplotlib style for consistent visualizations."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })


def get_framework_color(framework: str) -> str:
    """Get color for a framework."""
    return FRAMEWORK_COLORS.get(framework.lower(), DEFAULT_COLORS[0])


# ============================================================
# Individual Chart Functions
# ============================================================

def plot_latency_comparison(
    summaries: dict[str, FrameworkMetrics],
    output_path: str = None,
    title: str = "Comparação de Latência"
) -> plt.Figure:
    """
    Create bar chart comparing latency across frameworks.
    
    Shows: avg, p50, p95 latencies.
    """
    setup_style()
    
    frameworks = list(summaries.keys())
    n = len(frameworks)
    x = np.arange(n)
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Data
    avg_latencies = [summaries[f].avg_latency_ms for f in frameworks]
    p50_latencies = [summaries[f].p50_latency_ms for f in frameworks]
    p95_latencies = [summaries[f].p95_latency_ms for f in frameworks]
    
    # Bars
    colors = [get_framework_color(f) for f in frameworks]
    
    bars1 = ax.bar(x - width, avg_latencies, width, label='Média', color=colors, alpha=0.9)
    bars2 = ax.bar(x, p50_latencies, width, label='P50', color=colors, alpha=0.7)
    bars3 = ax.bar(x + width, p95_latencies, width, label='P95', color=colors, alpha=0.5)
    
    # Labels and formatting
    ax.set_xlabel('Framework')
    ax.set_ylabel('Latência (ms)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([f.capitalize() for f in frameworks])
    ax.legend()
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.0f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        logger.info(f"Saved latency chart to {output_path}")
    
    return fig


def plot_cost_comparison(
    summaries: dict[str, FrameworkMetrics],
    output_path: str = None,
    title: str = "Comparação de Custo"
) -> plt.Figure:
    """Create bar chart comparing cost per query."""
    setup_style()
    
    frameworks = list(summaries.keys())
    costs = [summaries[f].avg_cost_per_query * 1000 for f in frameworks]  # mUSD for readability
    colors = [get_framework_color(f) for f in frameworks]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bars = ax.bar(frameworks, costs, color=colors)
    
    ax.set_xlabel('Framework')
    ax.set_ylabel('Custo por Query (mUSD)')
    ax.set_title(title)
    ax.set_xticklabels([f.capitalize() for f in frameworks])
    
    # Add value labels
    for bar, cost in zip(bars, costs):
        ax.annotate(f'{cost:.3f}',
                   xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    
    return fig


def plot_quality_scores(
    summaries: dict[str, FrameworkMetrics],
    output_path: str = None,
    title: str = "Scores de Qualidade por Juiz"
) -> plt.Figure:
    """Create grouped bar chart for judge scores."""
    setup_style()
    
    frameworks = list(summaries.keys())
    n = len(frameworks)
    x = np.arange(4)  # 4 judges
    width = 0.8 / n
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    judge_names = ['Segurança', 'Qualidade', 'Atribuição', 'Formato']
    
    for i, framework in enumerate(frameworks):
        s = summaries[framework]
        scores = [
            s.avg_safety_score,
            s.avg_quality_score,
            s.avg_attribution_score,
            s.avg_format_score
        ]
        
        offset = (i - n/2 + 0.5) * width
        bars = ax.bar(x + offset, scores, width, 
                     label=framework.capitalize(),
                     color=get_framework_color(framework))
    
    ax.set_xlabel('Juiz')
    ax.set_ylabel('Score (0-100)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(judge_names)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Threshold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    
    return fig


def plot_radar_comparison(
    summaries: dict[str, FrameworkMetrics],
    output_path: str = None,
    title: str = "Comparação Multi-Dimensional"
) -> plt.Figure:
    """Create radar chart for multi-dimensional comparison."""
    setup_style()
    
    frameworks = list(summaries.keys())
    
    # Dimensions (normalized to 0-100, higher is better)
    categories = ['Latência', 'Custo', 'Segurança', 'Qualidade', 'Facilidade']
    n_cats = len(categories)
    
    # Calculate normalized values
    max_latency = max(s.avg_latency_ms for s in summaries.values())
    max_cost = max(s.avg_cost_per_query for s in summaries.values())
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    for framework in frameworks:
        s = summaries[framework]
        
        # Normalize values (invert latency and cost so higher is better)
        values = [
            100 - (s.avg_latency_ms / max_latency * 100) if max_latency > 0 else 50,
            100 - (s.avg_cost_per_query / max_cost * 100) if max_cost > 0 else 50,
            s.avg_safety_score,
            s.avg_quality_score,
            70  # Placeholder for "ease of use" - would need manual input
        ]
        values += values[:1]  # Complete the circle
        
        color = get_framework_color(framework)
        ax.plot(angles, values, 'o-', linewidth=2, label=framework.capitalize(), color=color)
        ax.fill(angles, values, alpha=0.25, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 100)
    ax.set_title(title, y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    
    return fig


def plot_success_rates(
    summaries: dict[str, FrameworkMetrics],
    output_path: str = None,
    title: str = "Taxas de Sucesso e Aprovação"
) -> plt.Figure:
    """Create stacked bar chart for success rates."""
    setup_style()
    
    frameworks = list(summaries.keys())
    
    approved = [summaries[f].approved_rate * 100 for f in frameworks]
    rejected = [summaries[f].rejected_rate * 100 for f in frameworks]
    other = [100 - a - r for a, r in zip(approved, rejected)]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(frameworks))
    
    ax.bar(x, approved, label='Aprovado', color='#10B981')
    ax.bar(x, other, bottom=approved, label='Aprovado c/ Ressalvas', color='#F59E0B')
    ax.bar(x, rejected, bottom=[a + o for a, o in zip(approved, other)], 
           label='Rejeitado', color='#EF4444')
    
    ax.set_xlabel('Framework')
    ax.set_ylabel('Porcentagem (%)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([f.capitalize() for f in frameworks])
    ax.legend()
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    
    return fig


# ============================================================
# Complete Dashboard
# ============================================================

def generate_evaluation_dashboard(
    run: EvaluationRun,
    output_dir: str = None,
    prefix: str = "eval"
) -> dict[str, str]:
    """
    Generate complete visualization dashboard.
    
    Returns dict of chart names to file paths.
    """
    output_dir = Path(output_dir or "./data/evaluation_results/charts")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summaries = run.framework_summaries
    generated_files = {}
    
    # Latency comparison
    path = output_dir / f"{prefix}_latency.png"
    plot_latency_comparison(summaries, str(path))
    generated_files["latency"] = str(path)
    plt.close()
    
    # Cost comparison
    path = output_dir / f"{prefix}_cost.png"
    plot_cost_comparison(summaries, str(path))
    generated_files["cost"] = str(path)
    plt.close()
    
    # Quality scores
    path = output_dir / f"{prefix}_quality.png"
    plot_quality_scores(summaries, str(path))
    generated_files["quality"] = str(path)
    plt.close()
    
    # Radar comparison
    path = output_dir / f"{prefix}_radar.png"
    plot_radar_comparison(summaries, str(path))
    generated_files["radar"] = str(path)
    plt.close()
    
    # Success rates
    path = output_dir / f"{prefix}_success.png"
    plot_success_rates(summaries, str(path))
    generated_files["success"] = str(path)
    plt.close()
    
    logger.info(f"Generated {len(generated_files)} charts in {output_dir}")
    
    return generated_files


def generate_thesis_figure(
    run: EvaluationRun,
    output_path: str = None
) -> plt.Figure:
    """
    Generate a single combined figure suitable for thesis inclusion.
    
    Creates a 2x3 subplot layout with all key metrics.
    """
    setup_style()
    
    summaries = run.framework_summaries
    frameworks = list(summaries.keys())
    
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Latency (top left)
    ax1 = fig.add_subplot(2, 3, 1)
    latencies = [summaries[f].avg_latency_ms for f in frameworks]
    colors = [get_framework_color(f) for f in frameworks]
    ax1.bar(frameworks, latencies, color=colors)
    ax1.set_title('Latência Média (ms)')
    ax1.set_xticklabels([f.capitalize() for f in frameworks])
    
    # 2. Cost (top middle)
    ax2 = fig.add_subplot(2, 3, 2)
    costs = [summaries[f].avg_cost_per_query * 1000 for f in frameworks]
    ax2.bar(frameworks, costs, color=colors)
    ax2.set_title('Custo por Query (mUSD)')
    ax2.set_xticklabels([f.capitalize() for f in frameworks])
    
    # 3. Overall Score (top right)
    ax3 = fig.add_subplot(2, 3, 3)
    scores = [summaries[f].avg_overall_score for f in frameworks]
    ax3.bar(frameworks, scores, color=colors)
    ax3.set_title('Score Geral')
    ax3.set_ylim(0, 100)
    ax3.set_xticklabels([f.capitalize() for f in frameworks])
    
    # 4. Judge Scores (bottom left)
    ax4 = fig.add_subplot(2, 3, 4)
    x = np.arange(4)
    width = 0.25
    judge_names = ['Seg.', 'Qual.', 'Attr.', 'Form.']
    
    for i, framework in enumerate(frameworks):
        s = summaries[framework]
        values = [s.avg_safety_score, s.avg_quality_score, 
                  s.avg_attribution_score, s.avg_format_score]
        ax4.bar(x + i * width, values, width, 
                label=framework.capitalize(), color=get_framework_color(framework))
    
    ax4.set_xticks(x + width)
    ax4.set_xticklabels(judge_names)
    ax4.set_title('Scores por Juiz')
    ax4.set_ylim(0, 100)
    ax4.legend(fontsize=8)
    
    # 5. Token Usage (bottom middle)
    ax5 = fig.add_subplot(2, 3, 5)
    tokens = [summaries[f].avg_tokens_per_query for f in frameworks]
    ax5.bar(frameworks, tokens, color=colors)
    ax5.set_title('Tokens por Query')
    ax5.set_xticklabels([f.capitalize() for f in frameworks])
    
    # 6. Radar (bottom right)
    ax6 = fig.add_subplot(2, 3, 6, polar=True)
    categories = ['Lat.', 'Custo', 'Seg.', 'Qual.']
    n_cats = len(categories)
    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]
    
    max_lat = max(s.avg_latency_ms for s in summaries.values())
    max_cost = max(s.avg_cost_per_query for s in summaries.values())
    
    for framework in frameworks:
        s = summaries[framework]
        values = [
            100 - (s.avg_latency_ms / max_lat * 100) if max_lat > 0 else 50,
            100 - (s.avg_cost_per_query / max_cost * 100) if max_cost > 0 else 50,
            s.avg_safety_score,
            s.avg_quality_score
        ]
        values += values[:1]
        ax6.plot(angles, values, 'o-', label=framework.capitalize(), 
                color=get_framework_color(framework))
        ax6.fill(angles, values, alpha=0.25, color=get_framework_color(framework))
    
    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(categories)
    ax6.set_ylim(0, 100)
    ax6.set_title('Comparação')
    
    plt.suptitle(f'Comparação de Frameworks RAG - {run.run_timestamp[:10]}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if output_path:
        plt.savefig(output_path)
        logger.info(f"Saved thesis figure to {output_path}")
    
    return fig
