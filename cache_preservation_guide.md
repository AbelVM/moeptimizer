Here is the complete architectural guide of DOs and DONTs for your Python client when interfacing with a Qwen MoE-MTP model on llama.cpp to enforce absolute KV cache stability. Review the project so it adheres to this guide.

## The DOs: Principles for Preserving the Cache

* DO Append Volatile Data to the Last Turn Only
  * Why: The KV cache reads sequentially from index 0. If you need to inject changing context (like timestamps, file trees, or system metrics), inject it inside the most recent user prompt. This shields the historical message chain from changing, preserving the common prefix.
* DO Echo Every Single Token of the thinking Process
  * Why: Qwen MoE models utilize specialized internal thinking blocks. When building the message payload for the next turn, your client must capture the explicit reasoning_content (or the block bounded by <think>...</think>) and pass it back in the assistant message. Omitting this completely alters the string syntax and invalidates the cache downstream.
* DO Keep the System Prompt Completely Immutable
  * Why: Modifying the first few tokens forces the server to clear and re-process the entire prompt history. Treat the system prompt as a read-only constant established at session initialization.
* DO Use the Structural Chat Completions API exclusively
  * Why: Passing structured JSON arrays (/v1/chat/completions) lets llama.cpp handle the ChatML tokenization mapping natively under the hood. Avoid the raw /completion endpoint, which exposes the template to subtle, accidental human syntax errors.
* DO Freeze the Structure and Order of Available Tools
  * Why: If your python app leverages function calling features, ensure the tools array parameter maintains an identical dictionary schema and array sorting index throughout the entire lifecycle. Shuffling the tool list mid-chat changes the initial prompt schema structure.
* DO Slice Exclusively from the Top When Truncating
  * Why: When a long conversation nears the context window boundary, prune entire old conversation turns from the top (immediately following the fixed system prompt). Never chop out sections from the middle of the chat log, as this breaks the contiguous token chain.

------------------------------
## The DONTs: Actions that Trigger Forced Prefills

* DONT Append Trailing Whitespaces or Empty Lines Randomly
  * Why: Tokenizers interpret \n, \n\n, and \n (newline with a space) as distinct token IDs. If your message rendering routine appends invisible or variable trailing whitespace to old history entries, the token comparison will fail instantly.
* DONT Use Server-Side Parameter Variations Mid-Chat
  * Why: Modifying generation properties such as temperature, top_k, or sampling parameters within the same conversation loop won't wipe the string cache, but it will force llama.cpp to bypass linear checkpoints and re-evaluate sampling layers.
* DONT Strip Hidden Object IDs During Session State Reloads
  * Why: If your python client serializes historical tool calls or assistant responses to a database, ensure it does not alter internal keys upon retrieval. If a tool response object maps differently on reload than it did during generation, it mutates the prefix context string.
* DONT Mix Text Strings and Multi-Modal/Tool Block Formats Interchangeably
  * Why: Qwen MoE architectures handle structural block transitions precisely. If Turn 1 passes a user response as a raw text string, but Turn 2 formats historical strings as explicit JSON type blocks [{"type": "text", "text": "..."}], tokenization footprints shift and dump the cache.
* DONT Allow Background Async Tasks to Interleave into the Active Context Slot
  * Why: If your script triggers background processing evaluations (e.g., hidden linter checks or diagnostic sub-prompts) using the same connection slot configuration, it will evict your main conversation cache to execute the mini-task. Ensure background micro-tasks utilize completely separate processing pools.
  

## "System-Level Ephemeral Insertion"

If you must use a summary to offset the data lost during Front-Loading Eviction, you must structure it so that it never sits between your historical prefix and the new generation block.

Follow these steps to safely implement this strategy:

* DO Create a Fixed "Summary Slot" in the User Message Structure: If you must pass a summary, it must replace a fixed metadata placeholder that always occupies the exact same number of lines, or it should be passed as a system-pinned context if you are willing to accept a one-time prefill penalty when the summary updates.
* DO Use Incremental Summary Triggers: Do not update the summary on every single turn. Only regenerate and inject the summary when llama.cpp triggers a context shift or eviction log. If you update the summary once every 10 turns instead of every turn, you will only suffer a full prefill 10% of the time, keeping the remaining 90% of your interactions near-instantaneous. [1] 
* DO Let llama.cpp Handle Eviction Automatically: If you are running a recent version of llama.cpp, it is highly recommended to rely on native server flags like --context-shift or advanced attention compression forks like MomentKV. These tools mathematically compress the KV cache tensors under the hood, maintaining the exact token alignment without requiring string mutations from your Python client. [2] 

[1] [https://github.com](https://github.com/anthropics/claude-code/issues/35242)
[2] [https://arxiv.org](https://arxiv.org/abs/2606.01563)
