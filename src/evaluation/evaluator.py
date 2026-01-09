"""
Evaluation Framework for PharmaBula RAG Systems

Comprehensive framework for comparing implementations:
- Metrics collection (latency, tokens, cost, scores)
- Multi-framework support (Gemini, LangChain, OpenAI)
- Results persistence and aggregation
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

from src.config import get_settings

logger = logging.getLogger(__name__)


# ============================================================
# Protocol for RAG Clients
# ============================================================

class RAGClient(Protocol):
    """Protocol that all RAG clients must implement for evaluation."""
    
    async def query(
        self,
        question: str,
        mode: str = "patient",
        n_context: int = 5,
        **kwargs
    ) -> str:
        """Execute a drug information query."""
        ...


# ============================================================
# Metrics Data Structures
# ============================================================

@dataclass
class QueryMetrics:
    """Metrics for a single query execution."""
    query_id: str
    query: str
    framework: str
    
    # Timing
    latency_ms: float
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    judge_ms: float = 0.0
    
    # Token usage
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    
    # Cost estimation
    estimated_cost_usd: float = 0.0
    
    # Judge scores
    safety_score: int = 0
    quality_score: int = 0
    attribution_score: int = 0
    format_score: int = 0
    overall_score: int = 0
    
    # Decision
    judge_decision: str = ""
    
    # Response
    response_length: int = 0
    response_valid_json: bool = False
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    error: Optional[str] = None


@dataclass
class FrameworkMetrics:
    """Aggregated metrics for a framework."""
    framework: str
    total_queries: int
    
    # Timing aggregates
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    
    # Token aggregates
    total_tokens: int
    avg_tokens_per_query: float
    
    # Cost aggregates
    total_cost_usd: float
    avg_cost_per_query: float
    
    # Score averages
    avg_safety_score: float
    avg_quality_score: float
    avg_attribution_score: float
    avg_format_score: float
    avg_overall_score: float
    
    # Success rate
    success_rate: float
    approved_rate: float
    rejected_rate: float


@dataclass
class EvaluationRun:
    """Complete evaluation run results."""
    run_id: str
    run_timestamp: str
    frameworks_evaluated: list[str]
    total_queries: int
    
    query_results: list[QueryMetrics]
    framework_summaries: dict[str, FrameworkMetrics]
    
    # Comparison
    best_latency: str
    best_quality: str
    best_cost: str


# ============================================================
# Cost Estimation
# ============================================================

# Pricing per 1M tokens (as of 2024)
PRICING = {
    "gemini": {
        "input": 0.075,   # Gemini Flash
        "output": 0.30
    },
    "langchain": {
        "input": 0.075,   # Using Gemini via LangChain
        "output": 0.30
    },
    "openai": {
        "input": 2.50,    # GPT-4 Turbo
        "output": 10.00
    },
    "anthropic": {
        "input": 3.00,    # Claude 3.5 Sonnet
        "output": 15.00
    }
}


def estimate_cost(
    framework: str,
    input_tokens: int,
    output_tokens: int
) -> float:
    """Estimate cost in USD based on token usage."""
    prices = PRICING.get(framework.lower(), PRICING["gemini"])
    
    input_cost = (input_tokens / 1_000_000) * prices["input"]
    output_cost = (output_tokens / 1_000_000) * prices["output"]
    
    return input_cost + output_cost


def estimate_tokens(text: str) -> int:
    """Rough token estimation (~4 chars per token)."""
    return len(text) // 4


# ============================================================
# Evaluator Class
# ============================================================

class RAGEvaluator:
    """
    Evaluator for comparing RAG implementations.
    
    Features:
    - Multi-framework evaluation
    - Comprehensive metrics collection
    - Results persistence
    - Comparison analysis
    """
    
    def __init__(self, output_dir: str = None):
        """Initialize evaluator with output directory."""
        settings = get_settings()
        self.output_dir = Path(output_dir or "./data/evaluation_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.clients: dict[str, RAGClient] = {}
        self.results: list[QueryMetrics] = []
    
    def register_client(self, name: str, client: RAGClient):
        """Register a RAG client for evaluation."""
        self.clients[name] = client
        logger.info(f"Registered client: {name}")
    
    async def evaluate_single(
        self,
        client_name: str,
        query: str,
        query_id: str = None,
        mode: str = "patient"
    ) -> QueryMetrics:
        """Evaluate a single query against a specific client."""
        client = self.clients.get(client_name)
        if not client:
            raise ValueError(f"Unknown client: {client_name}")
        
        query_id = query_id or f"{client_name}_{int(time.time())}"
        
        metrics = QueryMetrics(
            query_id=query_id,
            query=query,
            framework=client_name
        )
        
        try:
            # Measure total latency
            start = time.perf_counter()
            response = await client.query(question=query, mode=mode)
            end = time.perf_counter()
            
            metrics.latency_ms = (end - start) * 1000
            
            # Parse response
            metrics.response_length = len(response)
            try:
                parsed = json.loads(response)
                metrics.response_valid_json = True
            except json.JSONDecodeError:
                metrics.response_valid_json = False
                parsed = {}
            
            # Estimate tokens
            metrics.input_tokens = estimate_tokens(query)
            metrics.output_tokens = estimate_tokens(response)
            metrics.total_tokens = metrics.input_tokens + metrics.output_tokens
            
            # Estimate cost
            metrics.estimated_cost_usd = estimate_cost(
                client_name,
                metrics.input_tokens,
                metrics.output_tokens
            )
            
            # Extract judge scores if available from response
            if isinstance(parsed, dict):
                if "judgment" in parsed:
                    judgment = parsed["judgment"]
                    if isinstance(judgment, dict):
                        metrics.safety_score = judgment.get("score_breakdown", {}).get("safety", 0)
                        metrics.quality_score = judgment.get("score_breakdown", {}).get("quality", 0)
                        metrics.attribution_score = judgment.get("score_breakdown", {}).get("attribution", 0)
                        metrics.format_score = judgment.get("score_breakdown", {}).get("format", 0)
                        metrics.overall_score = judgment.get("score", 0)
                        metrics.judge_decision = judgment.get("decision", "")
            
        except Exception as e:
            metrics.error = str(e)
            logger.error(f"Evaluation error for {client_name}: {e}")
        
        return metrics
    
    async def evaluate_dataset(
        self,
        queries: list[dict],
        client_names: list[str] = None,
        mode: str = "patient",
        progress_callback: Callable[[int, int], None] = None
    ) -> list[QueryMetrics]:
        """
        Evaluate all queries across specified clients.
        
        Args:
            queries: List of {"id": str, "query": str} dicts
            client_names: Clients to evaluate (all if None)
            mode: Response mode
            progress_callback: Optional progress callback(current, total)
        
        Returns:
            List of QueryMetrics for all evaluations
        """
        clients_to_eval = client_names or list(self.clients.keys())
        total = len(queries) * len(clients_to_eval)
        current = 0
        
        results = []
        
        for query_item in queries:
            query_id = query_item.get("id", str(len(results)))
            query_text = query_item.get("query", query_item.get("text", ""))
            
            for client_name in clients_to_eval:
                current += 1
                if progress_callback:
                    progress_callback(current, total)
                
                metrics = await self.evaluate_single(
                    client_name=client_name,
                    query=query_text,
                    query_id=f"{query_id}_{client_name}",
                    mode=mode
                )
                
                results.append(metrics)
                self.results.append(metrics)
                
                # Small delay to avoid rate limiting
                await asyncio.sleep(0.5)
        
        return results
    
    def aggregate_results(self, results: list[QueryMetrics] = None) -> dict[str, FrameworkMetrics]:
        """Aggregate results by framework."""
        results = results or self.results
        
        from collections import defaultdict
        import numpy as np
        
        by_framework = defaultdict(list)
        for r in results:
            by_framework[r.framework].append(r)
        
        summaries = {}
        
        for framework, metrics_list in by_framework.items():
            latencies = [m.latency_ms for m in metrics_list if m.error is None]
            
            if not latencies:
                continue
            
            successful = [m for m in metrics_list if m.error is None]
            approved = [m for m in successful if "APPROVED" in m.judge_decision.upper()]
            rejected = [m for m in successful if "REJECTED" in m.judge_decision.upper()]
            
            summaries[framework] = FrameworkMetrics(
                framework=framework,
                total_queries=len(metrics_list),
                
                avg_latency_ms=float(np.mean(latencies)),
                p50_latency_ms=float(np.percentile(latencies, 50)),
                p95_latency_ms=float(np.percentile(latencies, 95)),
                p99_latency_ms=float(np.percentile(latencies, 99)),
                
                total_tokens=sum(m.total_tokens for m in metrics_list),
                avg_tokens_per_query=float(np.mean([m.total_tokens for m in metrics_list])),
                
                total_cost_usd=sum(m.estimated_cost_usd for m in metrics_list),
                avg_cost_per_query=float(np.mean([m.estimated_cost_usd for m in metrics_list])),
                
                avg_safety_score=float(np.mean([m.safety_score for m in successful])) if successful else 0,
                avg_quality_score=float(np.mean([m.quality_score for m in successful])) if successful else 0,
                avg_attribution_score=float(np.mean([m.attribution_score for m in successful])) if successful else 0,
                avg_format_score=float(np.mean([m.format_score for m in successful])) if successful else 0,
                avg_overall_score=float(np.mean([m.overall_score for m in successful])) if successful else 0,
                
                success_rate=len(successful) / len(metrics_list) if metrics_list else 0,
                approved_rate=len(approved) / len(successful) if successful else 0,
                rejected_rate=len(rejected) / len(successful) if successful else 0
            )
        
        return summaries
    
    def create_run_report(self, results: list[QueryMetrics] = None) -> EvaluationRun:
        """Create a complete evaluation run report."""
        results = results or self.results
        summaries = self.aggregate_results(results)
        
        frameworks = list(summaries.keys())
        
        # Determine best performers
        best_latency = min(summaries.items(), key=lambda x: x[1].avg_latency_ms)[0] if summaries else ""
        best_quality = max(summaries.items(), key=lambda x: x[1].avg_overall_score)[0] if summaries else ""
        best_cost = min(summaries.items(), key=lambda x: x[1].avg_cost_per_query)[0] if summaries else ""
        
        return EvaluationRun(
            run_id=f"eval_{int(time.time())}",
            run_timestamp=datetime.now().isoformat(),
            frameworks_evaluated=frameworks,
            total_queries=len(results),
            query_results=results,
            framework_summaries=summaries,
            best_latency=best_latency,
            best_quality=best_quality,
            best_cost=best_cost
        )
    
    def save_results(self, run: EvaluationRun = None, filename: str = None):
        """Save evaluation results to JSON file."""
        run = run or self.create_run_report()
        filename = filename or f"evaluation_{run.run_id}.json"
        
        output_path = self.output_dir / filename
        
        # Convert to serializable dict
        data = {
            "run_id": run.run_id,
            "run_timestamp": run.run_timestamp,
            "frameworks_evaluated": run.frameworks_evaluated,
            "total_queries": run.total_queries,
            "best_latency": run.best_latency,
            "best_quality": run.best_quality,
            "best_cost": run.best_cost,
            "query_results": [asdict(r) for r in run.query_results],
            "framework_summaries": {
                k: asdict(v) for k, v in run.framework_summaries.items()
            }
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_path}")
        return output_path
    
    def load_results(self, filename: str) -> EvaluationRun:
        """Load evaluation results from JSON file."""
        input_path = self.output_dir / filename
        
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        query_results = [QueryMetrics(**r) for r in data["query_results"]]
        framework_summaries = {
            k: FrameworkMetrics(**v)
            for k, v in data["framework_summaries"].items()
        }
        
        return EvaluationRun(
            run_id=data["run_id"],
            run_timestamp=data["run_timestamp"],
            frameworks_evaluated=data["frameworks_evaluated"],
            total_queries=data["total_queries"],
            query_results=query_results,
            framework_summaries=framework_summaries,
            best_latency=data["best_latency"],
            best_quality=data["best_quality"],
            best_cost=data["best_cost"]
        )


# ============================================================
# Convenience Functions
# ============================================================

async def run_full_evaluation(
    queries: list[dict],
    output_dir: str = None
) -> EvaluationRun:
    """
    Run full evaluation across all available frameworks.
    
    Automatically discovers and registers available clients.
    """
    evaluator = RAGEvaluator(output_dir=output_dir)
    
    # Try to register each client
    settings = get_settings()
    
    # Use framework factory
    from src.frameworks.factory import create_rag_instance, get_available_frameworks
    from config.settings import Framework
    
    available = get_available_frameworks()
    
    if Framework.MCP in available:
        try:
            evaluator.register_client("mcp", create_rag_instance(Framework.MCP))
        except Exception as e:
            logger.warning(f"Could not register MCP client: {e}")
    
    if Framework.LANGCHAIN in available:
        try:
            evaluator.register_client("langchain", create_rag_instance(Framework.LANGCHAIN))
        except Exception as e:
            logger.warning(f"Could not register LangChain client: {e}")
    
    if Framework.OPENAI in available:
        try:
            evaluator.register_client("openai", create_rag_instance(Framework.OPENAI))
        except Exception as e:
            logger.warning(f"Could not register OpenAI client: {e}")
    
    if not evaluator.clients:
        raise ValueError("No clients could be registered for evaluation")
    
    # Run evaluation
    logger.info(f"Starting evaluation with {len(queries)} queries across {len(evaluator.clients)} clients")
    
    def progress(current, total):
        if current % 5 == 0:
            logger.info(f"Progress: {current}/{total}")
    
    await evaluator.evaluate_dataset(
        queries=queries,
        progress_callback=progress
    )
    
    # Create and save report
    run = evaluator.create_run_report()
    evaluator.save_results(run)
    
    logger.info(f"Evaluation complete. Best latency: {run.best_latency}, Best quality: {run.best_quality}")
    
    return run
