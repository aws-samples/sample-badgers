"""
Image State Manager

Holds the original and current image across tool calls so the agent's tools
can read/write shared state.
"""

import numpy as np
from typing import Dict, Any


class ImageState:
    """Manages image state across agent tool invocations."""

    def __init__(self, image: np.ndarray):
        self.original = image.copy()
        self.current = image.copy()
        self.history: list[Dict[str, Any]] = []
        self.iteration = 0
        self.finished = False
        self.winner = "original"
        self.final_comparison: Dict[str, Any] | None = None

    def reset(self):
        """Reset current image back to original."""
        self.current = self.original.copy()

    def record_iteration(self, operations: list, comparison: Dict[str, Any]):
        self.history.append({
            "iteration": self.iteration,
            "operations": operations,
            "comparison": comparison,
        })
        self.iteration += 1
