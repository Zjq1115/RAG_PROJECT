# graph/agent_node.py
from typing import List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from graph.graph_state1 import AgentState
from llm_models.all_llm import llm
from tools.retriever_tools import retriever_tool
from utils.log_utils import log


def _build_safe_chat_context(messages: List[BaseMessage], max_turns: int = 10) -> List[BaseMessage]:
    """
    构建“可安全喂给 OpenAI ChatCompletions”的上下文：
    - 只保留 HumanMessage / AIMessage
    - 丢弃 ToolMessage（role=tool）
    - 丢弃包含 tool_calls 的 AIMessage（否则后面缺 tool 响应会被判非法）
    - 截断到最近 max_turns 条（避免 token 暴涨）
    """
    safe: List[BaseMessage] = []

    for m in messages:
        if isinstance(m, HumanMessage):
            safe.append(m)
            continue

        if isinstance(m, AIMessage):
            # 如果这条 assistant 含 tool_calls，则不能直接进上下文（除非你也成对提供 tool 结果）
            tool_calls = getattr(m, "tool_calls", None)
            if tool_calls:
                continue
            safe.append(m)
            continue

        # 其他类型（ToolMessage 等）一律跳过
        continue

    if max_turns > 0:
        safe = safe[-max_turns:]

    return safe


def agent_node(state: AgentState):
    """
    agent 负责决策是否调用 retriever_tool。
    这里我们让模型看到“多轮对话上下文”，但会过滤掉 tool 相关消息，保证消息序列合法。
    """
    log.info("---开始进入工作流---")

    # rewrite 达到上限后会走 web_search；路由发生前 agent 仍会执行一次，因此必须短路
    if state.get("force_web_search"):
        return {"messages": [AIMessage(content="")]}

    messages = state.get("messages", [])
    if not messages:
        return {"messages": [AIMessage(content="请先输入问题。")]}

    # 让模型看到上下文（多轮）
    chat_ctx = _build_safe_chat_context(messages, max_turns=12)

    # 兜底：如果过滤后没有 human 输入，就提示重试
    if not any(isinstance(m, HumanMessage) for m in chat_ctx):
        return {"messages": [AIMessage(content="未检测到用户输入，请重新发送问题。")]}

    model = llm.bind_tools([retriever_tool])

    # 关键：传入过滤后的多轮上下文，而不是 messages[-1]
    response = model.invoke(chat_ctx)
    return {"messages": [response]}
