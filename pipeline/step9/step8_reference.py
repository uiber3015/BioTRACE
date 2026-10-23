#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightweight access to selected step8.1 reference constants and helpers."""

from __future__ import annotations

import ast
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple


PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_EVAL_PATH = PROJECT_ROOT / "step8.1" / "evaluate_comparison.py"

_TARGET_ASSIGNMENTS = {
    "SYSTEM_PROMPT_STEP6",
    "INPUT_PROMPT_STEP6",
}
_TARGET_FUNCTIONS = {
    "_normalize_text_for_kc",
    "normalize_kc_payload",
    "extract_json_payload",
    "_compact_summary_for_history",
}


def _read_reference_source() -> str:
    if not REFERENCE_EVAL_PATH.exists():
        raise FileNotFoundError(
            f"step8.1/evaluate_comparison.py not found at {REFERENCE_EVAL_PATH}"
        )
    return REFERENCE_EVAL_PATH.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _load_reference_namespace() -> Dict[str, Any]:
    source = _read_reference_source()
    tree = ast.parse(source, filename=str(REFERENCE_EVAL_PATH))
    selected_nodes: List[ast.stmt] = []

    for node in tree.body:
        if isinstance(node, ast.Assign):
            target_names = {
                target.id
                for target in node.targets
                if isinstance(target, ast.Name)
            }
            if target_names & _TARGET_ASSIGNMENTS:
                selected_nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in _TARGET_FUNCTIONS:
            selected_nodes.append(node)

    missing = {
        *_TARGET_ASSIGNMENTS,
        *_TARGET_FUNCTIONS,
    } - {
        node.name
        for node in selected_nodes
        if isinstance(node, ast.FunctionDef)
    } - {
        target.id
        for node in selected_nodes
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    if missing:
        missing_names = ", ".join(sorted(missing))
        raise RuntimeError(
            f"Failed to extract required reference symbols from {REFERENCE_EVAL_PATH}: {missing_names}"
        )

    module = ast.Module(body=selected_nodes, type_ignores=[])
    compiled = compile(module, str(REFERENCE_EVAL_PATH), "exec")
    namespace: Dict[str, Any] = {
        "json": json,
        "re": re,
        "Any": Any,
        "Dict": Dict,
        "List": List,
        "Tuple": Tuple,
    }
    exec(compiled, namespace, namespace)
    return namespace


def get_step6_prompts() -> Tuple[str, str]:
    namespace = _load_reference_namespace()
    return (
        str(namespace["SYSTEM_PROMPT_STEP6"]),
        str(namespace["INPUT_PROMPT_STEP6"]),
    )


def normalize_kc_payload(kc: Dict[str, Any]) -> Dict[str, Any]:
    namespace = _load_reference_namespace()
    return namespace["normalize_kc_payload"](kc)


def extract_json_payload(text: str) -> str:
    namespace = _load_reference_namespace()
    return namespace["extract_json_payload"](text)


def compact_summary_for_history(summary: str, max_chars: int = 280) -> str:
    namespace = _load_reference_namespace()
    return namespace["_compact_summary_for_history"](summary, max_chars=max_chars)
