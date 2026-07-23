from moeptimizer.hierarchical_summarizer import HierarchicalSummarizer

summarizer = HierarchicalSummarizer(max_full_turns=2)

messages = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "1"}, {"role": "assistant", "content": "1"},
    {"role": "user", "content": "2"}, {"role": "assistant", "content": "2"},
    {"role": "user", "content": "3"}, {"role": "assistant", "content": "3"},
    {"role": "user", "content": "4"}, {"role": "assistant", "content": "4"},
    {"role": "user", "content": "5"}, {"role": "assistant", "content": "5"},
]

# First summary: frozen_end is 5 (system + user1 + asst1 + user2 + asst2).
res1 = summarizer.summarize_turns_cache_stable(messages, 5)
print("After first summary:")
for i, m in enumerate(res1):
    print(f"{i}: {m.get('role')} {m.get('content', '')[:30].replace(chr(10), ' ')}")

# Second summary: res1[5] is the summary block. frozen_end is 6 (which skips the summary block!)
# Wait, ContextAligner.frozen_prefix_end on res1 would return 6!
res1.append({"role": "user", "content": "6"})
res1.append({"role": "assistant", "content": "6"})
res2 = summarizer.summarize_turns_cache_stable(res1, 6)
print("\nAfter second summary:")
for i, m in enumerate(res2):
    print(f"{i}: {m.get('role')} {m.get('content', '')[:30].replace(chr(10), ' ')}")
