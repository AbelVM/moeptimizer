# REVIEW REQUEST for a Transparent OpenAI API proxy that optimizes context for MoE + MTP models in multi-turns agentic tasks.

What do we call "context optimization"? Keeping the leanest possible context that provides the best response quality from model in multi-turns agentic coding conversations, with minimal proxy latency and without triggering model kv-cache refill (being careful with MoE models).

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

- start by reading @README.md, @notes.md, online documentation about Qwen3.6-35B-A3B-MTP architecture and existing benchmark results
- The mission of this proxy is to improve speed (both TTFT and TPS) and quality of the inference using MoE-MTP models (not only Qwen3.6-35B-A3B-MTP) in local, hardware limited setups
- The proxy is transparent: both the lemonade-server and the client that uses the proxy speak standard OpenAI API
- Typical use-case is multi-turns agentic coding tasks (debug, refactor, review, etc.) that might include a codebase
- In terms of speed, both TTFT and TPS are important
- Both speed and responses quality degrades with the context size, so we need to keep it lean
- Target model is MoE: cache fill is expensive so we want to keep the context lean and avoid triggering model cache refills
- Semantic similarity of responses, direct request vs proxified, is key to measure the quality of the optimizing strategies. 

| Similarity | Grade | Comment |
| - | - | - |
| >=0.88 | A | Excellent,Impeccable |
| >=0.82 , < 0.88 | B | Good, Solid effort |
| >=0.75 , < 0.82 | C | Average, Minimum competent |
| >=0.68 , < 0.75 | D | Poor, Passing, but critical |
| < 0.68 | F | Failing, Unacceptable |
















The client uses moeptimizer as a standard OpenAI endpoint and doesn't know about custom fields like _session_id or _session_state. We need to ensure conversation continuity.

fully review all the optimization strategies implemented in moeptimizer, propose and implement fixes and/or improvements:
* are they properly wired?
* are they properly triggered when needed?
* is the strategies triggering order right?


run benchmark as background task, scenario refactor 30 turns 1 round, save the results as file in scripts folder, analyze the results and propose and implement improvemetns or fixes based on that

