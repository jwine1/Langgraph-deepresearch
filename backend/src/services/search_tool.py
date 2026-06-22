"""SearchTool — unified search backend, compatible with hello_agents Tool protocol."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

import requests

from hello_agents.tools.base import Tool, ToolParameter

logger = logging.getLogger(__name__)


class SearchTool(Tool):
    """Web search tool supporting duckduckgo, tavily, perplexity, and searxng.

    Inherits from hello_agents Tool base for ToolRegistry compatibility.
    run() returns either a plain string (error notice) or a dict with
    keys: results, backend, answer, notices.
    """

    def __init__(self, backend: str = "hybrid") -> None:
        super().__init__(
            name="search",
            description="执行网络搜索，支持多种后端 (duckduckgo, tavily, perplexity, searxng)",
        )
        self._default_backend = backend

    # -- Tool ABC ---------------------------------------------------------
    def run(self, parameters: Dict[str, Any]) -> Dict[str, Any] | str:
        query = str(parameters.get("input") or parameters.get("query") or "")
        if not query:
            return "❌ 搜索查询不能为空"

        backend = str(parameters.get("backend") or self._default_backend)
        max_results = int(parameters.get("max_results", 5))

        try:
            if backend == "perplexity":
                return self._search_perplexity(query, max_results)
            elif backend == "tavily":
                return self._search_tavily(query, parameters)
            elif backend == "searxng":
                return self._search_searxng(query, max_results)
            else:
                return self._search_duckduckgo(query, max_results)
        except Exception as exc:
            logger.exception("Search backend %s failed", backend)
            return {
                "results": [],
                "backend": backend,
                "answer": None,
                "notices": [f"搜索失败: {exc}"],
            }

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="input", type="string", description="搜索查询字符串"),
            ToolParameter(name="backend", type="string", description="搜索后端", required=False),
            ToolParameter(name="max_results", type="integer", description="最大结果数", required=False),
            ToolParameter(name="fetch_full_page", type="boolean", description="是否获取完整页面", required=False),
            ToolParameter(name="mode", type="string", description="搜索模式", required=False),
            ToolParameter(name="max_tokens_per_source", type="integer", description="每源最大 token 数", required=False),
            ToolParameter(name="loop_count", type="integer", description="循环计数", required=False),
        ]

    # -- backends ---------------------------------------------------------
    def _search_duckduckgo(self, query: str, max_results: int) -> Dict[str, Any]:
        url = "https://api.duckduckgo.com/"
        resp = requests.get(url, params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        results: list[dict[str, str]] = []
        abstract = data.get("AbstractText", "")
        if abstract:
            results.append({"title": data.get("Heading", query), "url": data.get("AbstractURL", ""), "content": abstract})

        for item in data.get("RelatedTopics", [])[: max_results - len(results)]:
            if isinstance(item, dict) and "Text" in item:
                results.append({
                    "title": item.get("FirstURL", "").rsplit("/", 1)[-1].replace("_", " "),
                    "url": item.get("FirstURL", ""),
                    "content": item.get("Text", ""),
                })

        return {"results": results, "backend": "duckduckgo", "answer": None, "notices": []}

    def _search_tavily(self, query: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        api_key = os.getenv("TAVILY_API_KEY", "")
        if not api_key:
            return {"results": [], "backend": "tavily", "answer": None, "notices": ["TAVILY_API_KEY 未设置"]}

        resp = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": api_key, "query": query, "max_results": parameters.get("max_results", 5), "include_answer": True},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        results = [{"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")}
                   for r in data.get("results", [])[: parameters.get("max_results", 5)]]
        return {"results": results, "backend": "tavily", "answer": data.get("answer"), "notices": []}

    def _search_perplexity(self, query: str, max_results: int) -> Dict[str, Any]:
        api_key = os.getenv("PERPLEXITY_API_KEY", "")
        if not api_key:
            return {"results": [], "backend": "perplexity", "answer": None, "notices": ["PERPLEXITY_API_KEY 未设置"]}

        resp = requests.post(
            "https://api.perplexity.ai/chat/completions",
            json={"model": "sonar-pro", "messages": [
                {"role": "system", "content": "You are a search assistant. Provide factual results."},
                {"role": "user", "content": query},
            ]},
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        answer = data["choices"][0]["message"]["content"] if data.get("choices") else None
        return {"results": [], "backend": "perplexity", "answer": answer, "notices": []}

    def _search_searxng(self, query: str, max_results: int) -> Dict[str, Any]:
        base_url = os.getenv("SEARXNG_BASE_URL", "http://localhost:8080").rstrip("/")
        resp = requests.get(f"{base_url}/search", params={"q": query, "format": "json"}, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        results = [{"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "") or r.get("snippet", "")}
                   for r in data.get("results", [])[:max_results]]
        return {"results": results, "backend": "searxng", "answer": None, "notices": []}
