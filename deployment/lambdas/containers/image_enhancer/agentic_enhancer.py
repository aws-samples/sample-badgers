"""
Agentic Image Enhancement

Orchestrates the Strands-based agentic enhancement loop with Claude Sonnet 4.6 vision model.
"""

import os
import json
import logging
from typing import Dict, Any
from pathlib import Path

import cv2
import numpy as np
import boto3
from strands import Agent, tool
from strands.models import BedrockModel

from enhancement_tools import (
    load_image,
    image_to_base64,
    save_image,
    analyze_image,
    execute_operations,
    OPERATIONS,
)
from image_state import ImageState

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Lambda environment configuration
AWS_REGION = os.environ.get("AWS_REGION", "us-west-2")
VISION_MODEL = os.environ.get("VISION_MODEL", "us.anthropic.claude-sonnet-4-6")
MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", "2"))
MAX_IMAGE_DIMENSION = int(os.environ.get("MAX_IMAGE_DIMENSION", "4000"))
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "85"))
OUTPUT_QUALITY = int(os.environ.get("OUTPUT_QUALITY", "95"))

# ============================================================================
# System Prompt
# ============================================================================

SYSTEM_PROMPT = f"""
You are an expert document image analyst. You enhance document images for
downstream text extraction by a vision LLM.

You have {MAX_ITERATIONS} iterations maximum to get the best result.

<workflow>
1. Examine the image carefully.
2. If it's already good enough, call finish_enhancement with winner="original".
3. Otherwise, call enhance_image with targeted operations.
4. Call compare_with_original to evaluate your enhancement.
5. Based on the comparison:
   - If metrics improved: call finish_enhancement with winner="enhanced".
   - If metrics got worse: call reset_to_original, then try different
     operations with gentler intensities.
   - If marginal: you may try one more iteration or accept the result.
6. After {MAX_ITERATIONS} iterations, you MUST call finish_enhancement.
</workflow>

<operation_guidance>
Available operations: {', '.join(OPERATIONS.keys())}

- Order matters: deskew/crop FIRST, then denoise, then contrast/brightness, sharpen LAST.
- Be conservative: 0.3-0.5 intensity is usually enough. Only go above 0.7 for severe degradation.
- Use region targeting if only part of the image has issues.
- For brightness: 0.0=darken, 0.5=no change, 1.0=brighten.
- Manuscripts need gentle treatment. Sheet music/diagrams: prefer contrast over sharpening.
</operation_guidance>

<rules>
- ALWAYS call compare_with_original after enhance_image.
- ALWAYS end by calling finish_enhancement.
- Do NOT enhance images that are already clear and well-oriented.
- If you reset, try a DIFFERENT approach — don't repeat what failed.
</rules>
"""

# ============================================================================
# Tool Definitions
# ============================================================================

# Rich schema for the enhance_image tool
ENHANCE_INPUT_SCHEMA = {
    "json": {
        "type": "object",
        "properties": {
            "operations": {
                "type": "array",
                "description": "Ordered list of enhancement operations to apply.",
                "items": {
                    "type": "object",
                    "properties": {
                        "op": {
                            "type": "string",
                            "enum": list(OPERATIONS.keys()),
                            "description": "Operation name.",
                        },
                        "intensity": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Effect strength. 0.0=none, 0.5=moderate, 1.0=max. For brightness: 0.0=darken, 0.5=no change, 1.0=brighten.",
                        },
                        "region": {
                            "type": "object",
                            "description": "Optional region (normalized 0-1 coords). Omit for full image.",
                            "properties": {
                                "x1": {"type": "number", "minimum": 0, "maximum": 1},
                                "y1": {"type": "number", "minimum": 0, "maximum": 1},
                                "x2": {"type": "number", "minimum": 0, "maximum": 1},
                                "y2": {"type": "number", "minimum": 0, "maximum": 1},
                            },
                        },
                    },
                    "required": ["op", "intensity"],
                },
            },
            "reasoning": {
                "type": "string",
                "description": "Brief explanation of why these operations were chosen.",
            },
        },
        "required": ["operations", "reasoning"],
    }
}


def create_tools(state: ImageState):
    """Create tool functions bound to a specific ImageState instance."""

    @tool(
        name="enhance_image",
        description="Apply targeted image enhancement operations to improve document readability. Operations are applied sequentially.",
        inputSchema=ENHANCE_INPUT_SCHEMA
    )
    def enhance_image(operations: list, reasoning: str) -> str:
        """Apply enhancement operations to the current image."""
        logger.info(f"enhance_image called: {reasoning}")
        enhanced, op_log = execute_operations(state.current, operations)
        state.current = enhanced

        return json.dumps({
            "status": "success",
            "operations_applied": len(op_log),
            "reasoning": reasoning,
            "note": "Call compare_with_original to evaluate the result.",
        })

    @tool
    def compare_with_original() -> str:
        """Compare the current enhanced image against the original.

        Returns quality metrics for both versions so you can judge
        whether the enhancement helped. Call this after enhance_image.
        """
        orig_metrics = analyze_image(state.original)
        curr_metrics = analyze_image(state.current)

        # Compute deltas
        deltas = {}
        for key in orig_metrics:
            if isinstance(orig_metrics[key], (int, float)):
                deltas[key] = round(curr_metrics[key] - orig_metrics[key], 3)

        state.record_iteration([], {"original": orig_metrics, "enhanced": curr_metrics, "deltas": deltas})

        return json.dumps({
            "original_metrics": orig_metrics,
            "enhanced_metrics": curr_metrics,
            "deltas": deltas,
            "iteration": state.iteration,
            "note": (
                "Positive contrast_std delta = more contrast. "
                "Positive sharpness_laplacian delta = sharper. "
                "If enhancement helped, call finish_enhancement with winner='enhanced'. "
                "If it made things worse, call reset_to_original and try different operations, "
                "or call finish_enhancement with winner='original'."
            ),
        })

    @tool
    def reset_to_original(reasoning: str) -> str:
        """Reset the image back to the original, discarding all enhancements.

        Use this when the previous enhancement made things worse and you
        want to try a different approach.

        Args:
            reasoning: Why you're resetting.
        """
        state.reset()
        logger.info(f"Reset to original: {reasoning}")
        return json.dumps({"status": "reset", "reasoning": reasoning})

    @tool
    def finish_enhancement(winner: str, reasoning: str) -> str:
        """Declare the enhancement process complete.

        Call this when you're satisfied with the result OR when you've
        determined the original is better than any enhancement.

        Args:
            winner: Either 'original' or 'enhanced'.
            reasoning: Why this version was chosen as the winner.
        """
        state.finished = True
        state.winner = winner
        state.final_comparison = {"winner": winner, "reasoning": reasoning}
        logger.info(f"Finished: winner={winner} — {reasoning}")
        return json.dumps({"status": "done", "winner": winner, "reasoning": reasoning})

    return [enhance_image, compare_with_original, reset_to_original, finish_enhancement]


# ============================================================================
# Enhancement Utilities
# ============================================================================

class EnhancementUtilities:
    """Orchestrates the Strands-based agentic enhancement loop."""

    @staticmethod
    def _resize_for_llm(image: np.ndarray, max_dim: int = MAX_IMAGE_DIMENSION) -> np.ndarray:
        """Resize image for LLM submission if needed."""
        h, w = image.shape[:2]
        if max(h, w) <= max_dim:
            return image
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _build_image_message(image: np.ndarray, context: str = "") -> list:
        """Build a multimodal message with the image for the Strands agent."""
        resized = EnhancementUtilities._resize_for_llm(image)
        b64 = image_to_base64(resized, fmt="jpeg", quality=JPEG_QUALITY)
        import base64 as b64_mod
        image_bytes = b64_mod.b64decode(b64)

        blocks = [
            {
                "image": {
                    "format": "jpeg",
                    "source": {"bytes": image_bytes},
                }
            },
            {"text": "Analyze this document image and enhance it for text extraction."},
        ]

        if context:
            blocks.append({"text": f"Document context: {context}"})

        return blocks

    @staticmethod
    def create_agent(state: ImageState) -> Agent:
        """Create a Strands agent configured for image enhancement."""
        session = boto3.Session(region_name=AWS_REGION)

        model = BedrockModel(
            model_id=VISION_MODEL,
            boto_session=session,
            max_tokens=2048,
        )

        tools = create_tools(state)

        return Agent(
            model=model,
            system_prompt=SYSTEM_PROMPT,
            tools=tools,
        )

    @staticmethod
    def enhance(
        image_source: str | Path | np.ndarray,
        context: str = "",
        save_output: bool = False,
    ) -> Dict[str, Any]:
        """
        Run the full agentic enhancement loop.

        Args:
            image_source: Path to image file or numpy array.
            context: Optional document context (e.g., "18th century manuscript").
            save_output: Whether to save the winner (not used in Lambda).

        Returns:
            Dict with original, enhanced, winner, history, and metadata.
        """
        image = load_image(image_source)
        source_name = Path(image_source).stem if isinstance(image_source, (str, Path)) else "image"

        logger.info('=' * 60)
        logger.info("  STRANDS AGENTIC IMAGE ENHANCEMENT")
        logger.info(f"  Source: {source_name}")
        logger.info(f"  Dimensions: {image.shape[1]}x{image.shape[0]}")
        logger.info(f"  Model: {VISION_MODEL}")
        logger.info(f"  Max iterations: {MAX_ITERATIONS}")
        logger.info('=' * 60)

        # Set up state and agent
        state = ImageState(image)
        agent = EnhancementUtilities.create_agent(state)

        # Build the multimodal prompt
        message = EnhancementUtilities._build_image_message(image, context)

        # Run the agent
        logger.info("Running Strands agent...")
        response = agent(message)

        # Force finish if agent didn't call finish_enhancement
        if not state.finished:
            logger.warning(f"Agent did not finish after {MAX_ITERATIONS} iterations. Forcing finish with original.")
            state.finished = True
            state.winner = "original"
            state.final_comparison = {"winner": "original", "reasoning": "Timeout: Agent exceeded MAX_ITERATIONS without finishing."}

        # Collect results
        winner = state.winner
        winner_image = state.current if winner == "enhanced" else state.original

        result = {
            "original": state.original,
            "enhanced": state.current if winner == "enhanced" else None,
            "winner": winner,
            "winner_image": winner_image,
            "history": state.history,
            "final_comparison": state.final_comparison,
            "iterations": state.iteration,
        }

        # Log result
        if winner == "enhanced":
            logger.info(f"Winner: ENHANCED (after {state.iteration} iteration(s))")
        else:
            logger.info("Winner: ORIGINAL")

        if state.final_comparison:
            logger.info(f"Reasoning: {state.final_comparison.get('reasoning', 'N/A')}")

        return result
