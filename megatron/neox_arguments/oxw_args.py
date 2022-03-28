import subprocess
from dataclasses import dataclass

try:
    from .template import NeoXArgsTemplate
except ImportError:
    from template import NeoXArgsTemplate

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


@dataclass
class OXWArgs(NeoXArgsTemplate):
    """
    OXW Arguments
    """

    from_pretrained_hf: str = None
    """
    Path to pretrained Huggingface model 
    """