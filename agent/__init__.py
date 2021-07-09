import sys

from .motreid import MOTreIDAgent
from .market1501 import Market1501Agent

def get_agent_cls(name):
    return getattr(sys.modules[__name__], name)
