from moeptimizer.context_aligner import ContextAligner

messages = [
    {"role": "system"},
    {"role": "user", "content": "1"},
    {"role": "assistant"},
    {"role": "user", "content": "2"},
    {"role": "assistant"},
    {"role": "user", "content": "3"},
    {"role": "assistant"},
    {"role": "user", "content": "Context summary (rolling):"},
    {"role": "user", "content": "4"},
]
ca = ContextAligner()
n = ca.frozen_prefix_end(messages, 2)
print("frozen_end =", n)
