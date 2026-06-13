# SYSTEM ARCHITECTURE REVIEW REQUEST

You are acting as a senior LLM inference architect specializing in:

- vLLM
- llama.cpp
- Qwen3.6-35B-A3B-MTP
- Mixture-of-Experts routing
- Prefix Caching
- KV Cache optimization
- Multi-Token Prediction (MTP)
- Agentic coding systems
- Long-context memory architectures
- Tool-augmented software engineering agents

Your task is NOT to explain basic concepts.

Your task is to review an EXISTING optimization architecture and identify:

1. Missing optimizations
2. Design weaknesses
3. Better alternatives
4. Additional throughput improvements
5. Additional context-efficiency improvements
6. Additional MTP-preservation techniques
7. Additional cache-preservation techniques

Assume all proposed changes will be implemented inside a local proxy sitting between the user and:

Qwen3.6-35B-A3B-MTP running on vLLM or llama.cpp.

---

# IMPORTANT

The following optimization is ALREADY IMPLEMENTED:

### Front-Loading Eviction

The proxy already performs:

- oldest-turn eviction
- top-of-history removal
- immutable prefix preservation
- static prompt anchoring
- linear append-only conversation growth

Do NOT recommend Front-Loading Eviction.

Treat it as solved.

Instead focus on improvements beyond it.

---

# EXISTING SYSTEM

## Context Architecture

### Static Layer

Contains:

- system prompt
- repository map
- project metadata
- AST skeletons
- persistent coding instructions

This region is intentionally immutable to maximize prefix cache reuse.

### Dynamic Layer

Contains:

- active conversation
- tool outputs
- execution results
- current task state

Appended linearly.

No historical rewriting occurs after inference.

---

## Context Pruning

When context thresholds are exceeded:

- oldest dynamic turns are removed
- pruning occurs only at the boundary immediately below the static layer
- no middle-context mutation occurs

---

## AST Compression

Source files are transformed into structural skeletons using tree-sitter.

Retained:

- classes
- functions
- methods
- signatures
- types
- docstrings
- imports
- indentation

Removed:

- implementation bodies
- internal logic
- comments
- literals

Targeted functions are hydrated JIT when active.

---

## Tool Output Lifecycle

Large outputs are not permanently retained.

Examples:

- file dumps
- terminal logs
- test runs
- compiler output

Historical tool outputs are replaced with synthetic references.

Example:

[SYSTEM: File previously inspected. Re-read file if needed.]

---

## Error Processing

Failure logs are compacted into:

- root exception
- stack trace
- first relevant lines
- last relevant lines

Noise is discarded.

---

## MTP Optimization

Proxy currently:

- pre-seeds reasoning prefixes
- injects thought-start delimiters
- attempts to reduce first-token latency
- attempts to improve MTP convergence

---

# REVIEW OBJECTIVE

Analyze this architecture and identify every additional optimization that could further improve:

## Context Efficiency

Examples:

- retrieval architectures
- repository indexing
- semantic chunking
- hierarchical memory
- attention management

---

## Prefix Cache Preservation

Examples:

- cache locality improvements
- static block organization
- deterministic prompt construction
- prompt canonicalization

---

## MTP Throughput

Examples:

- entropy reduction
- syntax stabilization
- speculative decoding improvements
- generation lane predictability

---

## MoE Efficiency

Examples:

- expert routing stability
- token distribution optimization
- structural prompting effects on routing

---

## Coding Agent Performance

Examples:

- file hydration strategies
- repository summarization
- dependency graph injection
- callgraph retrieval
- symbol indexing

---

## Long Context Stability

Examples:

- attention sink control
- recency balancing
- retrieval replacement strategies
- context fragmentation mitigation

---

# REQUIRED OUTPUT FORMAT

For every optimization provide:

## Optimization Name

### Why It Helps

Explain the mechanism.

### Expected Impact

Estimate impact on:

- latency
- throughput
- context efficiency
- MTP performance
- cache hit rate

### Complexity

Low / Medium / High

### Implementation Strategy

Concrete implementation details.

### Priority

Critical / High / Medium / Low

---

# FINAL SECTION

Produce:

## Top 10 Highest ROI Optimizations

Ranked by:

1. Throughput gain
2. Context savings
3. MTP stability
4. Ease of implementation

Provide estimated percentage improvements whenever possible.

Avoid generic advice.

Focus specifically on Qwen3.6-35B-A3B-MTP, vLLM, llama.cpp, coding agents, prefix caching, speculative decoding, and long-context inference systems.