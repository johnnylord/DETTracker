import sys

from .motreid import MOTreIDAgent

def get_agent_cls(name):
    return getattr(sys.modules[__name__], name)
