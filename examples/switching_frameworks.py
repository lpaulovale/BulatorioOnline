"""
Example: Switching between frameworks.

Demonstrates how to use different RAG frameworks.
"""

import asyncio
import os

from src.frameworks.factory import create_rag_instance, get_rag
from config.settings import Framework


async def main():
    # Method 1: Use environment variable
    os.environ["ACTIVE_FRAMEWORK"] = "openai"
    rag = get_rag()
    print(f"Using: {rag.__class__.__name__}")
    
    # Method 2: Specify directly
    # mcp_rag = create_rag_instance(Framework.MCP)
    # langchain_rag = create_rag_instance(Framework.LANGCHAIN)
    # openai_rag = create_rag_instance(Framework.OPENAI)
    
    # Example query
    query = "Para que serve o paracetamol?"
    
    # Search documents
    docs = await rag.search_documents(query, top_k=3)
    print(f"Found {len(docs)} documents")
    
    # Generate answer
    response = await rag.generate_answer(query, docs)
    
    print(f"\nAnswer: {response.answer[:200]}...")
    print(f"Framework: {response.framework}")
    
    if response.judgment:
        print(f"Safety Score: {response.judgment.score_breakdown.get('safety', 'N/A')}")
        print(f"Overall Score: {response.judgment.overall_score}")
        print(f"Decision: {response.judgment.final_decision.value}")


if __name__ == "__main__":
    asyncio.run(main())
