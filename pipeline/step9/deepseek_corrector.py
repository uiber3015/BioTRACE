#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Robust DeepSeek teacher corrector for Step9 PPO training."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from step9.step8_reference import (
    compact_summary_for_history,
    extract_json_payload,
    normalize_kc_payload,
)

PROJECT_ROOT = Path(__file__).parent.parent


def _default_teacher_prompts_dir() -> Path:
    # Only use step8.1, no fallback
    preferred = PROJECT_ROOT / "step8.1" / "prompts" / "teacher"
    if not preferred.exists():
        raise FileNotFoundError(f"step8.1/prompts/teacher not found at {preferred}")
    return preferred


class DeepSeekCorrector:
    """DeepSeek corrector with retry/backoff and prompt fallback."""

    _PLACEHOLDER_RE = re.compile(r"\{[\w/\u4e00-\u9fff→]+\}")

    def __init__(
        self,
        teacher_prompts_dir: Optional[Path] = None,
        max_retries: int = 3,
        timeout_seconds: int = 180,
        backoff_base_seconds: float = 2.0,
    ) -> None:
        self.teacher_prompts_dir = (
            teacher_prompts_dir
            if teacher_prompts_dir is not None
            else _default_teacher_prompts_dir()
        )
        self.max_retries = max(1, int(max_retries))
        self.timeout_seconds = int(timeout_seconds)
        self.backoff_base_seconds = float(backoff_base_seconds)

        self.api_url = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")
        fallback_env = os.getenv("DEEPSEEK_API_URL_FALLBACKS", "")
        self.api_fallback_urls = [u.strip() for u in fallback_env.split(",") if u.strip()]
        self.api_key = os.getenv("DEEPSEEK_API_KEY", "")
        self.model_name = os.getenv("DEEPSEEK_MODEL", "DeepSeek-V3.2")

        self.strategy_to_file: Dict[str, str] = {
            "Add/chain": "teacher_add_chain.txt",
            "Add/condition_kv": "teacher_add_condition_kv.txt",
            "Add/entity_type": "teacher_add_entity_type.txt",
            "Add/entity_value": "teacher_add_entity_value.txt",
            "Modify/condition_kv": "teacher_modify_condition_kv.txt",
            "Modify/result": "teacher_modify_result.txt",
            "Modify/entity_type": "teacher_modify_entity_type.txt",
            # Keep backward compatibility for older strategy aliases.
            "Modify/entity_recognition": "teacher_modify_entity_type.txt",
            "Modify/entity_value": "teacher_modify_entity_value.txt",
            "Delete/chain": "teacher_delete_chain.txt",
            "Delete/condition_kv": "teacher_delete_condition_kv.txt",
            "Delete/result": "teacher_delete_result.txt",
            "Delete/entity_type": "teacher_delete_entity_type.txt",
            "Delete/entity_value": "teacher_delete_entity_value.txt",
        }
        self._strategy_lower_map = {k.lower(): k for k in self.strategy_to_file.keys()}
        self._lint_teacher_prompt_placeholders()

    def _lint_teacher_prompt_placeholders(self) -> None:
        """Fail fast if teacher templates contain placeholders not handled by _build_prompt."""
        if not self.teacher_prompts_dir.exists():
            return
        allowed_placeholders = {
            "{TABLE_CONTENT}",
            "{CURRENT_GENERATED_JSON}",
            "{STUDENT_OPERATION_JSON}",
            "{STUDENT_REASON}",
            "{LAST_CORRECTION_SUMMARY}",
            "{chain_indices}",
            "{学生新增的具体内容摘要}",
            "{学生修改的具体内容摘要}",
            "{删除了哪些链}",
            "{删除了哪些片段}",
            "{新增的实体类型和值}",
            "{新增的实体值}",
            "{删除了哪些类型}",
            "{删除了哪些实体值}",
            "{旧result→新result}",
            "{result置空}",
            "{原类型→新类型}",
            "{旧值→新值}",
        }
        unknown_by_file: Dict[str, List[str]] = {}
        for filename in sorted(set(self.strategy_to_file.values())):
            path = self.teacher_prompts_dir / filename
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8")
            placeholders = set(self._PLACEHOLDER_RE.findall(text))
            unknown = sorted(p for p in placeholders if p not in allowed_placeholders)
            if unknown:
                unknown_by_file[path.name] = unknown
        if unknown_by_file:
            details = "; ".join(
                f"{name}: {', '.join(tokens)}"
                for name, tokens in sorted(unknown_by_file.items())
            )
            raise ValueError(
                "Unhandled teacher prompt placeholder(s) detected. "
                f"Please update _build_prompt replacements: {details}"
            )

    def _escape_untrusted_text(self, text: str) -> str:
        """Escape angle brackets to reduce tag-closing prompt-injection attempts."""
        return text.replace("<", "\\u003c").replace(">", "\\u003e")

    def _sanitize_for_prompt(self, text: Any, max_chars: int) -> str:
        out = str(text or "").strip()
        if len(out) > max_chars:
            out = out[:max_chars] + "\n...[TRUNCATED]..."
        return self._escape_untrusted_text(out)

    def _sanitize_api_error(self, exc: Exception) -> str:
        """Return compact error tag without upstream response body."""
        msg = str(exc).lower()
        if "timed out" in msg or "timeout" in msg:
            return "timeout"
        if "connection" in msg:
            return "connection_error"
        if "http 401" in msg:
            return "http_401"
        if "http 403" in msg:
            return "http_403"
        if "http 404" in msg:
            return "http_404"
        if "http 400" in msg:
            return "http_400"
        if "http 429" in msg:
            return "http_429"
        if "http 5" in msg:
            return "http_5xx"
        if "json" in msg:
            return "invalid_response_json"
        return type(exc).__name__

    def _normalize_strategy(self, strategy: str) -> str:
        raw = (strategy or "").strip()
        if not raw:
            return ""

        raw = raw.replace("\\", "/").replace(" ", "")
        low = raw.lower()
        if low in self._strategy_lower_map:
            return self._strategy_lower_map[low]

        parts = [p for p in raw.split("/") if p]
        if not parts:
            return ""

        action = parts[0].capitalize()
        if action not in {"Add", "Modify", "Delete"}:
            return raw

        target = parts[1] if len(parts) > 1 else "condition_kv"
        target_low = target.lower()
        target_alias = {
            "condition": "condition_kv",
            "conditionkv": "condition_kv",
            "entity": "entity_type",
            "entitytype": "entity_type",
            "entityrecognition": "entity_type",
            "entity_recognition": "entity_type",
            "entityvalue": "entity_value",
        }
        target = target_alias.get(target_low, target)
        candidate = f"{action}/{target}"
        if candidate.lower() in self._strategy_lower_map:
            return self._strategy_lower_map[candidate.lower()]
        return candidate

    def _pick_prompt_path(self, strategy: str) -> Optional[Path]:
        canonical = self._normalize_strategy(strategy)
        if canonical not in self.strategy_to_file:
            return None

        candidates: List[Path] = []
        if canonical in self.strategy_to_file:
            candidates.append(self.teacher_prompts_dir / self.strategy_to_file[canonical])
        if canonical:
            candidates.append(self.teacher_prompts_dir / f"teacher_{canonical.replace('/', '_')}.txt")

        seen = set()
        for path in candidates:
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            if path.exists():
                return path
        return None

    def _build_prompt(
        self,
        prompt_template: str,
        table_content: str,
        current_kc: Dict[str, Any],
        strategy_output: Dict[str, Any],
        last_summary: str,
    ) -> str:
        normalized_current_kc = normalize_kc_payload(current_kc)
        
        chain_indices = strategy_output.get("chain_indices", [])
        reason = str(strategy_output.get("reason", "") or "")
        student_operation_json = {
            "strategy": str(strategy_output.get("strategy", "")),
            "chain_indices": chain_indices,
            "reason": reason,
        }

        # 与 step8.1 对齐：直接文本替换，不使用 XML-like 标签
        # step8.1 做法: prompt.replace("{TABLE_CONTENT}", table_content)
        prompt = prompt_template.replace("{TABLE_CONTENT}", table_content)
        
        # step8.1: 使用归一化后的 KC
        prompt = prompt.replace(
            "{CURRENT_GENERATED_JSON}",
            json.dumps(normalized_current_kc, ensure_ascii=False)
        )
        prompt = prompt.replace(
            "{STUDENT_OPERATION_JSON}",
            json.dumps(student_operation_json, ensure_ascii=False)
        )
        
        # step8.1: reason 为空时使用默认文本
        reason_text = reason if reason else "（模型未提供理由，根据策略类型和表格内容自行判断）"
        prompt = prompt.replace("{STUDENT_REASON}", reason_text)
        
        compact_last_summary = compact_summary_for_history(last_summary, max_chars=500)
        if compact_last_summary:
            prompt = prompt.replace(
                "{LAST_CORRECTION_SUMMARY}",
                f"\n## 上一轮修正总结\n{compact_last_summary}",
            )
        else:
            prompt = prompt.replace("{LAST_CORRECTION_SUMMARY}", "")

        # Optional placeholders in some prompt variants.
        prompt = prompt.replace("{chain_indices}", str(chain_indices))
        strategy_safe = student_operation_json["strategy"]
        prompt = prompt.replace("{学生新增的具体内容摘要}", f"使用策略 {strategy_safe} 对链 {chain_indices} 进行修正")
        prompt = prompt.replace("{学生修改的具体内容摘要}", f"使用策略 {strategy_safe} 对链 {chain_indices} 进行修正")
        prompt = prompt.replace("{删除了哪些链}", f"删除链 {chain_indices}")
        prompt = prompt.replace("{删除了哪些片段}", f"删除链 {chain_indices} 的condition片段")
        prompt = prompt.replace("{新增的实体类型和值}", f"对链 {chain_indices} 添加实体")
        prompt = prompt.replace("{新增的实体值}", f"对链 {chain_indices} 添加实体值")
        prompt = prompt.replace("{删除了哪些类型}", f"删除链 {chain_indices} 的实体类型")
        prompt = prompt.replace("{删除了哪些实体值}", f"删除链 {chain_indices} 的实体值")
        prompt = prompt.replace("{旧result→新result}", f"修改链 {chain_indices} 的result")
        prompt = prompt.replace("{result置空}", f"置空链 {chain_indices} 的result")
        prompt = prompt.replace("{原类型→新类型}", f"修改链 {chain_indices} 的实体类型")
        prompt = prompt.replace("{旧值→新值}", f"修改链 {chain_indices} 的实体值")
        return prompt

    def _validate_kc_schema(self, parsed: Any) -> bool:
        """与 step8.1 对齐：宽松校验，接受任何有效的 JSON 对象"""
        # step8.1 不做严格的 schema 校验，而是先解析再 normalize
        # 这里放宽校验，接受任何 dict
        if not isinstance(parsed, dict):
            return False
        return True

    def _find_matching_brace(self, text: str, start_idx: int) -> Optional[int]:
        if start_idx < 0 or start_idx >= len(text) or text[start_idx] != "{":
            return None
        depth = 0
        in_string = False
        escaped = False
        for i in range(start_idx, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == "\"":
                    in_string = False
                continue
            if ch == "\"":
                in_string = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return i
        return None

    def _coerce_kc_payload(self, parsed: Any) -> Optional[Dict[str, Any]]:
        """Accept a few common near-miss formats from teacher output."""
        if isinstance(parsed, dict):
            if "knowledge_chain" in parsed and "knowledge_mining" not in parsed:
                return {"knowledge_mining": [parsed]}
            if isinstance(parsed.get("knowledge_mining"), dict):
                return {"knowledge_mining": [parsed["knowledge_mining"]]}
            return parsed

        if isinstance(parsed, list):
            if len(parsed) == 1 and isinstance(parsed[0], dict) and "knowledge_mining" in parsed[0]:
                return parsed[0]
            if all(isinstance(item, dict) for item in parsed):
                return {"knowledge_mining": list(parsed)}
        return None

    def _kc_payload_equal(self, left: Dict[str, Any], right: Dict[str, Any]) -> bool:
        try:
            return json.dumps(left, ensure_ascii=False, sort_keys=True) == json.dumps(
                right,
                ensure_ascii=False,
                sort_keys=True,
            )
        except Exception:
            return left == right

    def _extract_stop_signal(self, response_text: str) -> Optional[str]:
        txt = response_text.strip()
        match = re.search(r"(?im)^\s*STOP(?:\s*[:：-]\s*|\s+)?(.*)$", txt)
        if not match:
            return None
        stop_extra = (match.group(1) or "").strip()
        return f"TEACHER_STOP: {stop_extra}" if stop_extra else "TEACHER_STOP"

    def _looks_like_no_change_summary(self, summary: str) -> bool:
        compact = re.sub(r"\s+", "", str(summary or ""))
        if not compact:
            return False
        if compact.upper().startswith("TEACHER_STOP"):
            return True
        positive_markers = ("状态:已修改", "结果:已修改")
        negative_markers = (
            "状态:未修改",
            "结果:未修改",
            "无需修改",
            "无需新增",
            "无需删除",
            "无需替换",
            "无需修正",
            "无需更改",
            "不需要修改",
            "保持原json",
            "保持原kc",
            "保持当前kc",
            "未做修改",
        )
        return any(token in compact for token in negative_markers) and not any(
            token in compact for token in positive_markers
        )

    def _to_teacher_stop_summary(self, summary: str) -> str:
        cleaned = re.sub(r"\s+", " ", str(summary or "")).strip()
        cleaned = cleaned.replace("【修正总结】", "").strip(" -:")
        if len(cleaned) > 220:
            cleaned = cleaned[:217] + "..."
        return f"TEACHER_STOP: {cleaned}" if cleaned else "TEACHER_STOP: no_change"

    def _build_repair_prompt(
        self,
        original_prompt: str,
        invalid_response: str,
        error_tag: str,
    ) -> str:
        invalid_excerpt = self._sanitize_for_prompt(invalid_response, max_chars=1600)
        return (
            "你上一条回答格式错误，程序无法解析。请严格重新输出，并遵守以下规则：\n"
            "1. 如果无需修改，直接输出 `STOP: 原因`。\n"
            "2. 如果需要修改，先输出且只输出一个 JSON 对象，顶层必须为 "
            "{\"knowledge_mining\":[...]}。\n"
            "3. 修改时可以在 JSON 后追加简短【修正总结】，但禁止输出 markdown 代码块、"
            "禁止输出 JSON 列表、禁止输出多个 JSON、禁止输出任何前言解释。\n"
            f"上一次错误类型：{error_tag}\n"
            f"上一次错误回答：\n{invalid_excerpt}\n\n"
            "以下是原始任务，请重新作答：\n"
            f"{original_prompt}"
        )

    def _extract_corrected_json(self, response_text: str) -> Tuple[Optional[Dict[str, Any]], str]:
        txt = response_text.strip()

        stop_signal = self._extract_stop_signal(txt)
        if stop_signal is not None:
            return None, stop_signal

        # step8.1: 解析 JSON
        try:
            json_str = extract_json_payload(response_text)
        except Exception:
            short_resp = re.sub(r"\s+", " ", txt)
            if len(short_resp) > 160:
                short_resp = short_resp[:157] + "..."
            return None, f"INVALID_RESPONSE_NO_JSON: {short_resp}"

        # step8.1: 解析并 normalize
        try:
            raw_parsed = json.loads(json_str)
            parsed = self._coerce_kc_payload(raw_parsed)
            if not isinstance(parsed, dict):
                return None, "INVALID_RESPONSE_NOT_OBJECT"
            # 与 step8.1 一致：使用 normalize_kc_payload
            corrected_kc = normalize_kc_payload(parsed)
        except Exception:
            return None, "INVALID_RESPONSE_PARSE_ERROR"

        # 提取 summary（JSON 之后的文本）
        summary = ""
        json_pos = response_text.find(json_str)
        if json_pos >= 0:
            summary = response_text[json_pos + len(json_str):].strip()

        return corrected_kc, summary

    def _call_api(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        # 与 step8.1 对齐：只使用 user message，不添加额外的 system_guard
        # step8.1 做法: messages = [{"role": "user", "content": prompt}]
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 8192,
        }

        # Support multiple API URLs for fallback.
        api_urls = self.api_url.split(",") if "," in self.api_url else [self.api_url]
        api_urls.extend(self.api_fallback_urls)
        api_urls.append("https://api.deepseek.com/v1/chat/completions")
        dedup_urls: List[str] = []
        seen = set()
        for u in api_urls:
            u = u.strip()
            if not u or u in seen:
                continue
            seen.add(u)
            dedup_urls.append(u)
        api_urls = dedup_urls or [self.api_url]
        
        # Status codes that are worth retrying (transient errors)
        retryable_status_codes = {502, 503, 504, 429, 408, 500}
        # Status codes that should NOT be retried (authentication/bad request errors)
        non_retryable_status_codes = {401, 403, 404, 400}

        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            for url_idx, url in enumerate(api_urls):
                try:
                    resp = requests.post(
                        url,
                        headers=headers,
                        json=payload,
                        timeout=self.timeout_seconds,
                    )
                    if resp.status_code != 200:
                        status_code = resp.status_code
                        err_msg = f"HTTP {status_code}"
                        
                        # Check if we should retry this status code
                        if status_code in non_retryable_status_codes:
                            # Don't retry auth errors - they're unlikely to succeed on retry
                            raise RuntimeError(f"Non-retryable error: {err_msg}")
                        elif status_code in retryable_status_codes:
                            # Retry on transient errors
                            raise RuntimeError(f"Transient error: {err_msg}")
                        else:
                            # Other errors - try next URL or retry
                            raise RuntimeError(err_msg)
                    
                    data = resp.json()
                    return str(data["choices"][0]["message"]["content"])
                except RuntimeError as exc:
                    # Check if it's a non-retryable error
                    if "Non-retryable error" in str(exc):
                        raise
                    last_err = exc
                    # Try next URL in the list
                    if url_idx < len(api_urls) - 1:
                        continue
                    # All URLs failed or it's a retryable error, wait and retry
                    if attempt < self.max_retries:
                        sleep_s = self.backoff_base_seconds ** (attempt - 1)
                        time.sleep(sleep_s)
                    continue
                except Exception as exc:
                    last_err = exc
                    # Try next URL
                    if url_idx < len(api_urls) - 1:
                        continue
                    if attempt < self.max_retries:
                        sleep_s = self.backoff_base_seconds ** (attempt - 1)
                        time.sleep(sleep_s)
                    continue

        assert last_err is not None
        safe_code = self._sanitize_api_error(last_err)
        raise RuntimeError(
            f"DeepSeek API failed after {self.max_retries} retries: {safe_code}"
        )

    def correct(
        self,
        table_content: str,
        current_kc: Dict[str, Any],
        strategy_output: Dict[str, Any],
        last_summary: str = "",
    ) -> Tuple[Dict[str, Any], str]:
        strategy = str(strategy_output.get("strategy", "") or "")
        action = str(strategy_output.get("action", "") or "").upper().strip()
        if action == "STOP" or strategy.upper() == "STOP":
            return current_kc, "TEACHER_SKIP_STOP"

        prompt_path = self._pick_prompt_path(strategy)
        if prompt_path is None:
            return current_kc, f"[TEACHER_PROMPT_MISSING] strategy={strategy}"

        prompt_template = prompt_path.read_text(encoding="utf-8")
        prompt = self._build_prompt(
            prompt_template=prompt_template,
            table_content=table_content,
            current_kc=current_kc,
            strategy_output=strategy_output,
            last_summary=last_summary,
        )
        try:
            normalized_current_kc = normalize_kc_payload(current_kc)
        except Exception:
            normalized_current_kc = current_kc

        max_invalid_retries = 2
        attempt_prompt = prompt
        for invalid_retry in range(max_invalid_retries + 1):
            try:
                response_text = self._call_api(attempt_prompt)
            except Exception as exc:
                safe_code = self._sanitize_api_error(exc)
                return current_kc, f"[TEACHER_API_ERROR] strategy={strategy}; code={safe_code}"

            corrected_kc, summary = self._extract_corrected_json(response_text)
            if corrected_kc is None:
                if isinstance(summary, str) and summary.upper().startswith("TEACHER_STOP"):
                    return current_kc, summary
                if isinstance(summary, str) and summary.upper().startswith("INVALID_RESPONSE"):
                    if invalid_retry < max_invalid_retries:
                        attempt_prompt = self._build_repair_prompt(prompt, response_text, summary)
                        time.sleep(1)
                        continue
                    return current_kc, summary
                return current_kc, f"[TEACHER_PARSE_ERROR] strategy={strategy}; detail={summary[:300]}"

            if self._kc_payload_equal(corrected_kc, normalized_current_kc):
                if isinstance(summary, str) and summary.upper().startswith("TEACHER_STOP"):
                    return current_kc, summary
                if self._looks_like_no_change_summary(summary):
                    return current_kc, self._to_teacher_stop_summary(summary)
                return current_kc, "TEACHER_NO_CHANGE"
            return corrected_kc, summary

        # 不应到达这里，但作为安全网
        return current_kc, f"[TEACHER_MAX_RETRIES] strategy={strategy}"
