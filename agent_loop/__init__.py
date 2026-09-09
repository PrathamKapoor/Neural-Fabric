"""Neural Fabric autonomous manager/worker development loop.

Roles (chains never cross):
  MANAGER (Nemotron 3 Ultra free, OpenRouter) -> fallback Kimi K3 (NVIDIA).
  WORKER  (AGY headless CLI, Gemini 3.8 Flash High) -> Nemotron 3 Super
          (OpenRouter) -> backup DeepSeek V4 Flash (NVIDIA).

Entry point:  python -m agent_loop
"""

__version__ = "0.2.0"

from .config import LoopConfig
from .loop import AgentLoop
from .roles import ChainExhausted, RoleChain

__all__ = ["LoopConfig", "AgentLoop", "RoleChain", "ChainExhausted", "__version__"]
