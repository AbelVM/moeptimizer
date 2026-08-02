import json

msg1 = {"role": "user", "content": "Context summary:\nFact 1\nFact 2", "_summary_id": "abc"}
msg2 = {"role": "user", "content": "Context summary:\nFact 1\nFact 2\nFact 3", "_summary_id": "abc"}

s1 = json.dumps(msg1, sort_keys=True)
s2 = json.dumps(msg2, sort_keys=True)

for i in range(min(len(s1), len(s2))):
    if s1[i] != s2[i]:
        print(f"Divergence at char {i}")
        print(f"s1: {s1[i-10:i+10]}")
        print(f"s2: {s2[i-10:i+10]}")
        break
else:
    print("Full prefix match!")
