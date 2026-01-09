#!/usr/bin/env python
"""
Example: Run Evaluation Across All Frameworks

This script demonstrates how to use the evaluation framework
to compare Gemini, LangChain, and OpenAI implementations.

Usage:
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --queries 5
    python scripts/run_evaluation.py --frameworks gemini langchain
"""

import asyncio
import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import (
    RAGEvaluator,
    TestDataset,
    generate_evaluation_dashboard,
    generate_thesis_figure,
    get_dataset_stats,
)
from src.config import get_settings


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument(
        "--queries", "-n",
        type=int,
        default=None,
        help="Number of queries to evaluate (default: all)"
    )
    parser.add_argument(
        "--frameworks", "-f",
        nargs="+",
        default=None,
        help="Frameworks to evaluate (default: all available)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./data/evaluation_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--mode", "-m",
        type=str,
        default="patient",
        choices=["patient", "professional"],
        help="Response mode"
    )
    
    args = parser.parse_args()
    
    # Load test dataset
    dataset = TestDataset()
    stats = get_dataset_stats(dataset)
    logger.info(f"Dataset loaded: {stats}")
    
    # Get queries to evaluate
    if args.queries:
        queries = dataset.get_sample(args.queries)
    else:
        queries = dataset.get_all()
    
    logger.info(f"Evaluating {len(queries)} queries")
    
    # Initialize evaluator
    evaluator = RAGEvaluator(output_dir=args.output)
    settings = get_settings()
    
    # Register available clients
    frameworks_to_eval = args.frameworks or []
    
    # Try Gemini
    if not frameworks_to_eval or "gemini" in frameworks_to_eval:
        if settings.gemini_api_key:
            try:
                from src.llm.gemini_client import get_gemini_client
                evaluator.register_client("gemini", get_gemini_client())
                logger.info("✓ Gemini client registered")
            except Exception as e:
                logger.warning(f"✗ Gemini not available: {e}")
    
    # Try LangChain
    if not frameworks_to_eval or "langchain" in frameworks_to_eval:
        if settings.gemini_api_key:
            try:
                from src.llm.langchain_client import get_langchain_client
                evaluator.register_client("langchain", get_langchain_client())
                logger.info("✓ LangChain client registered")
            except Exception as e:
                logger.warning(f"✗ LangChain not available: {e}")
    
    # Try OpenAI
    if not frameworks_to_eval or "openai" in frameworks_to_eval:
        if settings.openai_api_key:
            try:
                from src.llm.openai_client import get_openai_client
                evaluator.register_client("openai", get_openai_client())
                logger.info("✓ OpenAI client registered")
            except Exception as e:
                logger.warning(f"✗ OpenAI not available: {e}")
    
    if not evaluator.clients:
        logger.error("No clients available for evaluation!")
        sys.exit(1)
    
    logger.info(f"Evaluating with clients: {list(evaluator.clients.keys())}")
    
    # Run evaluation
    def progress(current, total):
        if current % 5 == 0 or current == total:
            pct = (current / total) * 100
            logger.info(f"Progress: {current}/{total} ({pct:.1f}%)")
    
    await evaluator.evaluate_dataset(
        queries=queries,
        mode=args.mode,
        progress_callback=progress
    )
    
    # Create and save report
    run = evaluator.create_run_report()
    output_path = evaluator.save_results(run)
    
    logger.info(f"Results saved to: {output_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    for framework, summary in run.framework_summaries.items():
        print(f"\n{framework.upper()}")
        print("-" * 40)
        print(f"  Avg Latency:     {summary.avg_latency_ms:.0f} ms")
        print(f"  P95 Latency:     {summary.p95_latency_ms:.0f} ms")
        print(f"  Avg Cost/Query:  ${summary.avg_cost_per_query:.6f}")
        print(f"  Overall Score:   {summary.avg_overall_score:.1f}/100")
        print(f"  Safety Score:    {summary.avg_safety_score:.1f}/100")
        print(f"  Quality Score:   {summary.avg_quality_score:.1f}/100")
        print(f"  Approval Rate:   {summary.approved_rate * 100:.1f}%")
    
    print("\n" + "=" * 60)
    print("WINNERS")
    print("=" * 60)
    print(f"  Best Latency:  {run.best_latency}")
    print(f"  Best Quality:  {run.best_quality}")
    print(f"  Best Cost:     {run.best_cost}")
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    charts = generate_evaluation_dashboard(run, output_dir=args.output + "/charts")
    logger.info(f"Charts saved: {list(charts.keys())}")
    
    # Generate thesis figure
    thesis_path = Path(args.output) / "thesis_comparison.png"
    generate_thesis_figure(run, str(thesis_path))
    logger.info(f"Thesis figure saved: {thesis_path}")
    
    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    asyncio.run(main())
