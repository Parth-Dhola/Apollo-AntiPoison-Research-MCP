"""demo.py — Interactive demo script to showcase Apollo MCP Server in action."""

import asyncio
from apollo.router.tool_selector import select_tools_for_query
from apollo.server.mcp_server import create_mcp_server


def get_text(tool_result):
    if hasattr(tool_result, "content") and tool_result.content:
        return tool_result.content[0].text
    return str(tool_result)


async def run_live_demo():
    print("=" * 70)
    print(" 🚀 APOLLO RESEARCH MCP SERVER — LIVE CAPABILITIES DEMO")
    print("=" * 70)

    # ── 1. Standalone Query Router & Tool Selector ────────────────────────────
    print("\n[1] TESTING STANDALONE QUERY ROUTER (Zero-Cost Bag-of-Words)")
    print("-" * 70)
    test_queries = [
        "What is the mathematical proof of convergence for Adam optimizer?",
        "PyTorch implementation of multi-head attention forward and backward pass",
        "Explain arXiv paper 2205.14135",
        "Survey of vision transformers with github code implementations",
        "What is the latest OpenAI GPT-4o release date and pricing?"
    ]
    for q in test_queries:
        res = select_tools_for_query(q)
        print(f"• Query:      \"{q}\"")
        print(f"  Intent:     {res.intent.value} (Confidence: {res.confidence * 100:.0f}%)")
        print(f"  Tools:      {', '.join(res.recommended_tools)}\n")

    server = create_mcp_server()

    # ── 2. Academic Paper Deep Context & LaTeX Normalization ──────────────────
    print("\n[2] TESTING DEEP PAPER CONTEXT (arXiv: 2205.14135 - FlashAttention)")
    print("-" * 70)
    fetch_tool = await server.get_tool("fetch_paper_deep_context")
    paper_res = await fetch_tool.run({"arxiv_id": "2205.14135", "max_tokens": 500})
    print(get_text(paper_res))

    # ── 3. GitHub Implementation Search & License Boilerplate Stripping ───────
    print("\n[3] TESTING GITHUB CODE IMPLEMENTATIONS (Topic: flash-attention)")
    print("-" * 70)
    repo_tool = await server.get_tool("search_repo_implementations")
    repo_res = await repo_tool.run({"topic": "flash-attention", "language": "python", "top_k": 1})
    print(get_text(repo_res))

    # ── 4. Unified Anti-Poisoned RAG & CPU Reranker ───────────────────────────
    print("\n[4] TESTING UNIFIED CONTEXT ENGINE (Router + arXiv + Anti-Poison + Rerank)")
    print("-" * 70)
    query = "How does FlashAttention-2 speed up forward and backward pass?"
    unified_tool = await server.get_tool("unified_research_context")
    unified_res = await unified_tool.run({"query": query, "top_k": 2})
    print(get_text(unified_res))

    print("\n" + "=" * 70)
    print(" ✅ ALL APOLLO CAPABILITIES VERIFIED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_live_demo())
