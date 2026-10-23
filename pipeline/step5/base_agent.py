# -*- coding: utf-8 -*-
"""
Common base class for step5 agents (knowledge_chain, entity_type).

This is a lightweight replacement so that `KnowledgeChainAgent` and
`EntityTypeAgent` can share:
  - a logger
  - a reference to the LLM text client (`llm_client`)
  - a simple `log_activity` helper
"""

from __future__ import annotations

import logging
from typing import Any


class BaseAgent:
    """Minimal base class expected by KnowledgeChainAgent and EntityTypeAgent."""

    def __init__(self, name: str, text_llm_client: Any) -> None:
        self.name = name
        self.llm_client = text_llm_client
        # Use a stable logger name so logs from different agents are grouped
        logger_name = f"step5.{name.lower()}"
        self.logger = logging.getLogger(logger_name)

    # Simple helper for verbose internal logging
    def log_activity(self, message: str) -> None:
        """
        Log an internal activity message for debugging / tracing.

        Agents call this to record important processing steps; we default
        to INFO level so it appears in normal logs.
        """
        try:
            self.logger.info(message)
        except Exception:
            # Be defensive: logging failures should never break the pipeline
            pass

