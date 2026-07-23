from moeptimizer.config import AppConfig
from moeptimizer.optimizer import AgentContextOptimizer
class FakeProbe:
    def cached(self): return type("Caps", (), {"max_context_window": 8192, "remote_tokenize": False})()
    def get(self): return self.cached()

config = AppConfig()
config.agentic.quality_profile = "balanced"
config.agentic.dynamic_budget_enabled = False
config.agentic.max_optimized_tokens = 500

opt = AgentContextOptimizer(config)
opt._capability_probe = FakeProbe()

messages = [{"role": "system", "content": "sys"}]
for i in range(15):
    messages.append({"role": "user", "content": f"user {i} " * 50})
    messages.append({"role": "assistant", "content": f"asst {i} " * 50})

res = opt.optimize_messages(messages)
for m in res:
    print(m["role"], repr(m["content"][:30]))
