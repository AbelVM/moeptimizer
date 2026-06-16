# REVIEW REQUEST for a Transparent OpenAI API proxy that optimizes context for MoE + MTP models in multi-turns agentic tasks.

You are acting as a senior LLM inference architect specializing in:

- vLLM
- llama.cpp
- OpenAI API
- Qwen3.6-35B-A3B-MTP
- tree-sitter AST
- embeddings
- Prompt engineering
- Mixture-of-Experts routing
- Prefix Caching
- KV Cache optimization
- Multi-Token Prediction (MTP)
- Agentic coding systems
- Long-context memory architectures
- Tool-augmented software engineering agents

Your task is NOT to explain basic concepts.

Your task is to review an EXISTING context optimization architecture and identify:

1. Missing optimizations
2. Design weaknesses
3. Better alternatives
4. Additional throughput improvements
5. Additional context-efficiency improvements
6. Additional MTP-preservation techniques
7. Additional model kv-cache preservation techniques
8. Potential bugs
9. Performance bottlenecks
10. Memory leaks

Assume all proposed changes will be implemented inside a local, OpenAI API compliant, transparent proxy sitting between the user and Qwen3.6-35B-A3B-MTP (MoE model with MTP) running on llama.cpp using lemonade server.

IMPORTANT:

- start by reading @README.md and @notes.md and online documentation about Qwen3.6-35B-A3B-MTP architecture
- The mission of this proxy is to improve speed (both TTFT and TPS) and quality of the inference using MoE-MTP models (not only Qwen3.6-35B-A3B-MTP) in local, hardware limited setups
- Typical use is multi-turns agentic coding tasks (debug, refactor, review, etc.)
- In terms of speed, both TTFT and TPS are important
- Both speed and responses quality degrades with the context size, so we need to keep it lean
- Target model is MoE: cache fill is expensive so we want to keep the context lean and avoid triggering cache refills
- Semantic similarity of responses is key to measure the quality of the optimizing strategies


Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.

The client uses moeptimizer as a standard OpenAI endpoint and doesn't know about custom fields like _session_id or _session_state. We need to ensure conversation continuity.

fully review all the optimization strategies implemented in moeptimizer, propose and implement fixes and/or improvements:
* are they properly wired?
* are they properly triggered when needed?
* is the strategies triggering order right?


run benchmark as background task, scenario refactor 30 turns 1 round, save the results as file in scripts folder, analyze the results and propose and implement improvemetns or fixes based on that