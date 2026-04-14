# graph/web_search_node.py
from typing import Dict, Any, List
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage
from graph.get_human_message import get_last_human_message
from utils.log_utils import log


def _normalize_tavily_result(result: Any) -> str:
    """
    TavilySearchResults 可能返回：
    - str（包括错误字符串）
    - list[str]
    - list[dict]
    - list[Document]
    - dict
    这里统一转换成可读文本
    """
    # 关键：如果整体就是字符串，直接返回（不要 for 遍历它）
    if isinstance(result, str):
        return result

    # list
    if isinstance(result, list):
        parts: List[str] = []
        for item in result:
            if isinstance(item, str):
                parts.append(item)
            elif hasattr(item, "page_content"):
                parts.append(item.page_content)
            elif isinstance(item, dict):
                text = item.get("content") or item.get("text") or ""
                url = item.get("url")
                if url:
                    parts.append(f"来源: {url}\n{text}")
                else:
                    parts.append(text)
            else:
                parts.append(str(item))
        return "\n\n".join([p for p in parts if p.strip()])

    # dict
    if isinstance(result, dict):
        return (
            result.get("content")
            or result.get("text")
            or result.get("answer")
            or str(result)
        )

    # 其他兜底
    return str(result)


def web_search(state: Dict[str, Any]) -> Dict[str, Any]:
    log.info("---进入联网搜索节点(Web Search)---")

    messages = state.get("messages", [])
    try:
        question = get_last_human_message(messages).content
    except Exception:
        question = ""

    if not question.strip():
        log.warning("web_search: 未获取到有效问题")
        return {"messages": [AIMessage(content="【联网搜索失败】未获取到有效问题。")]}

    tool = TavilySearchResults(k=3)

    try:
        # TavilySearchResults 通常既支持 str，也支持 {"query": str}
        # 这里用 str 更稳
        raw = tool.invoke(question)

        text = _normalize_tavily_result(raw).strip()
        if not text:
            raise ValueError("联网搜索结果为空")

        # 如果是 HTTPError 字符串，也不要拆字符，直接整体返回
        return {"messages": [AIMessage(content=f"【联网搜索结果】\n{text}")]}
    except Exception as e:
        log.error(f"联网搜索失败: {e}")
        return {"messages": [AIMessage(content=f"【联网搜索失败】{e}")]}
