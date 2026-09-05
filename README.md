# Apollo — Anti-Poison & Anti-Hallucination Multi-Source Research MCP Server

[![CI/CD](https://github.com/Parth-Dhola/Apollo-AntiPoison-AntiHallucination-Research-MCP/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/Parth-Dhola/Apollo-AntiPoison-AntiHallucination-Research-MCP/actions)
[![License: GPLv3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org)
[![Tests](https://img.shields.io/badge/tests-27%20passed-brightgreen.svg)](https://github.com/Parth-Dhola/Apollo-AntiPoison-AntiHallucination-Research-MCP/actions)
[![MCP](https://img.shields.io/badge/Protocol-MCP%201.0-orange.svg)](https://modelcontextprotocol.io)
[![Cost](https://img.shields.io/badge/Cost-%240%20(100%25%20Free)-brightgreen.svg)](#zero-cost-design)

> **Apollo** is a standalone, production-grade Model Context Protocol (MCP) server engineered with a **Dual-Pillar Defense**: protecting AI agents from **context poisoning (prompt injection & adversarial attacks)** and **retrieval-induced hallucinations (context overload & "Lost-in-the-Middle")** across academic papers (arXiv, Semantic Scholar), open-source repositories (GitHub), Wikipedia encyclopedia, and web search. Built with $0 external API fees in mind.

---

## 🏛️ Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Apollo MCP Server (Standalone)                  │
│                                                                        │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │ 1. Multi-Source Ingestion & Tool Capability RAG Indexer        │   │
│   │    ├─ arXiv Atom API (100% Free / Public XML parser)           │   │
│   │    ├─ Semantic Scholar Graph API (Free Tier Public Endpoint)   │   │
│   │    ├─ GitHub REST API (Public Repos & Code Search)             │   │
│   │    ├─ Wikipedia API (100% Free / Foundational Concepts)        │   │
│   │    └─ DuckDuckGo Fallback Search (Zero API Keys)               │   │
│   └───────────────────────────────┬────────────────────────────────┘   │
│                                   │                                    │
│                                   ▼                                    │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │ 2. Context Sanitization & Anti-Poisoning Layer (The Filter)    │   │
│   │    ├─ Prompt Injection Scanner (Adversarial regex & redaction) │   │
│   │    ├─ Invisible Unicode & BiDi Override Stripper               │   │
│   │    ├─ LaTeX & Markdown Normalizer (Preserves Math Blocks)      │   │
│   │    └─ Noise Reducer (Strips bibliographies & code licenses)    │   │
│   └───────────────────────────────┬────────────────────────────────┘   │
│                                   │                                    │
│                                   ▼                                    │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │ 3. Guardrail RAG & Relevance Reranker                          │   │
│   │    ├─ Zero-Cost Bag-of-Words & Okapi BM25 CPU Indexer          │   │
│   │    ├─ FlashRank Ultra-Fast CPU Cross-Encoder (<25ms latency)   │   │
│   │    └─ Grounded Snippet Packer (Secure XML Enclosure + Citations│   │
│   └───────────────────────────────┬────────────────────────────────┘   │
│                                   │                                    │
└───────────────────────────────────┼────────────────────────────────────┘
                                    ▼
       Exposes Clean Tools to AI Agents / Claude / Cursor / Antigravity:
       • `search_academic_papers(query, year_start, year_end, min_citations, top_k)`
       • `fetch_paper_deep_context(arxiv_id, max_tokens)`
       • `search_repo_implementations(topic, language, min_stars, top_k)`
       • `search_wikipedia(query, max_results)`
       • `fallback_web_search(query, max_results)`
       • `match_tools_for_query(query, max_tools)`
       • `unified_research_context(query, top_k)`
```

---

## ⚡ Zero-Cost Design

Apollo was designed specifically for students and researchers:
- **Zero API Costs**: arXiv and DuckDuckGo require no API keys. Semantic Scholar and GitHub run on free public rate limits.
- **Zero Embedding/Vector Database Costs**: Uses pure Python **Okapi BM25** and **FlashRank CPU Cross-Encoder** (`ms-marco-TinyBERT-L-2-v2`) running directly in RAM with <25ms CPU latency.
- **Rate Limit Caching**: Integrated disk and memory caching (`diskcache`) to respect public rate limits.

---

## 🚀 Quickstart

### 1. Installation

```bash
git clone https://github.com/Parth-Dhola/Apollo-AntiPoison-AntiHallucination-Research-MCP.git
cd Apollo-AntiPoison-AntiHallucination-Research-MCP

# Create conda environment
conda create -n apollo python=3.11 -y
conda activate apollo

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Standalone Tool Selector CLI

Test query intent classification directly from the command line:

```bash
python -m apollo.router.tool_selector "How to implement LoRA linear layer in PyTorch?"
```

Output:
```
============================================================
 Apollo Tool Selector & Intent Router
============================================================
Query:             How to implement LoRA linear layer in PyTorch?
Predicted Intent:  CODE_IMPLEMENTATION
Confidence:        95%
Keywords Matched:  pytorch, implementation, implement
Recommended Tools: search_repo_implementations, fallback_web_search
Reasoning:         Code implementation query matching keywords: pytorch, implementation, implement
============================================================
```

### 3. Run MCP Server

#### Local Stdio Mode (Claude Desktop / Cursor / Antigravity):
```bash
python -m apollo.main --transport stdio
```

#### HTTP / SSE Server Mode (Docker / EC2 Microservice):
```bash
python -m apollo.main --transport sse --port 8080
```

---

## 🛡️ Dual-Pillar Defense Architecture

Apollo solves the two fundamental failure modes of AI Agent retrieval: **Security Hijacking (Context Poisoning)** and **Model Hallucination (Context Overload)**.

```
┌────────────────────────────────────────────────────────────────────────┐
│                          APOLLO DUAL-PILLAR SHIELD                     │
├───────────────────────────────────┬────────────────────────────────────┤
│ PILLAR 1: Security Firewall       │ PILLAR 2: Anti-Hallucination Guard │
├───────────────────────────────────┼────────────────────────────────────┤
│ • Prompt Injection Scanner        │ • Adaptive Tool Gating (Tool RAG)  │
│ • Invisible Unicode Stripper      │ • Anti-Context Bloat Pruning       │
│ • LaTeX / Markdown Normalizer     │ • "Lost-in-the-Middle" Prevention  │
│ • Hardened XML Isolation Tags     │ • Hard Token Budgeting (<2500 ch)  │
│ • Academic Noise / License Strip  │ • FlashRank Cross-Encoder Precision│
└───────────────────────────────────┴────────────────────────────────────┘
```

### 1. Pillar 1: Anti-Poisoning & Security Guardrails
Apollo ensures context retrieved from external sources is safe before reaching your LLM:
* **Prompt Injection Redaction**: Detects and neutralizes prompt override attempts (`ignore previous instructions`, `system override`, `<<SYS>>`, `<|im_start|>`).
* **Invisible Unicode Stripping**: Removes zero-width spaces (`\u200B`), BiDi overrides (`\u202E`), and hidden character exploits.
* **Hardened XML Encapsulation**: Encloses external context in `<untrusted_academic_context>` tags with explicit provenance metadata.

### 2. Pillar 2: Anti-Hallucination & Context-Overload Defense
Dumping too much unstructured search text into an LLM causes the **"Lost-in-the-Middle" phenomenon** (Stanford / Liu et al.) and leads to **Retrieval-Induced Hallucinations**. Apollo prevents this through:
* **Adaptive Tool Gating**: The Tool RAG Indexer evaluates semantic fit and **prunes away unneeded tools** (e.g., skips Wikipedia & Web when arXiv matches), preventing redundant multi-source flooding.
* **Strict Score Thresholding**: Prunes any candidate snippet whose relevance score falls below the cutoff threshold, eliminating distractor noise.
* **Hard Token Budgeting**: Enforces a strict `2,500`-character context budget to preserve LLM attention density and prevent token bloat.

---

## 🏛️ Source Authority & Credibility Hierarchy

Apollo prevents unverified or crowd-sourced summaries from displacing peer-reviewed science:

| Tier | Source | Authority Weight | Role in Research Queries |
|---|---|---|---|
| **Tier 1** | **arXiv & Semantic Scholar** | `1.00x` | **Primary Ground Truth**: Peer-reviewed proofs, theorems, SOTA benchmarks. |
| **Tier 2** | **GitHub Repos** | `0.90x` | **Verified Code**: Runnable models, CUDA kernels, PyTorch modules. |
| **Tier 3** | **Wikipedia** | `0.65x` | **Secondary Encyclopedia**: Definitions & rate-limit safety net (Deprioritized for research). |
| **Tier 4** | **DuckDuckGo** | `0.55x` | **General Web**: Fallback for news and release notes. |

---

## 🐳 Docker & Compose

Run Apollo in Docker:

```bash
docker compose up -d
```

Check health:
```bash
curl http://localhost:8080/sse
```

---

## 🧪 Testing

Run test suite with coverage:

```bash
pytest tests/ -v --cov=src/apollo --cov-report=term-missing
```

---

## 🔗 Python SDK & Agent Integration

To call Apollo directly in your AI agentic workflows or Python scripts:

```python
import asyncio
from apollo.server.mcp_server import create_mcp_server

async def get_clean_research_context(query: str):
    server = create_mcp_server()
    tool = await server.get_tool("unified_research_context")
    result = await tool.run({"query": query, "top_k": 3})
    return result.content[0].text

# Run
context = asyncio.run(get_clean_research_context("FlashAttention-2 forward backward pass"))
print(context)
```

## 📜 License
 
Licensed under the **GNU General Public License v3.0 or later (GPL-3.0-or-later)**.

All derivative software and modifications must remain open-source under GPLv3. See the [LICENSE](LICENSE) file for the full license text.
