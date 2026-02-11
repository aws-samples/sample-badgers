"""New wizard prompt generation module.

Replaces the monolithic LLM call with 6 individual per-prompt-type calls,
each using a real few-shot example from existing analyzers.
"""

from .prompt_generator import generate_all_prompts

__all__ = ["generate_all_prompts"]
