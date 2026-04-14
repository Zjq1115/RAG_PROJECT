# graph/generate_node.py
from typing import Dict, Any, Callable, Optional
from langchain_core.messages import AIMessage
from graph.get_human_message import get_last_human_message
from llm_models.all_llm import llm
from utils.log_utils import log

# 全局回调，由 server.py 注入
_stream_callback: Optional[Callable[[str], None]] = None


def set_stream_callback(cb: Optional[Callable[[str], None]]):
    """设置流式输出回调函数"""
    global _stream_callback
    _stream_callback = cb


def get_stream_callback() -> Optional[Callable[[str], None]]:
    """获取当前回调"""
    return _stream_callback


def generate(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    最终回答生成节点（真流式版）：
    - 使用 llm.stream() 逐token生成
    - 通过回调函数实时推送每个token
    """
    log.info("---生成最终的答案（流式）---")

    messages = state.get("messages", [])

    # 取用户原始问题（root_question 优先）
    question = state.get("root_question")
    if not question:
        try:
            question = get_last_human_message(messages).content
        except Exception:
            question = ""

    # 拼接上下文（忽略明显的"联网失败"提示）
    context_parts = []
    for m in messages:
        if not hasattr(m, "content"):
            continue
        text = m.content or ""
        if "联网搜索失败" in text:
            continue
        context_parts.append(text)

    context = "\n\n".join(context_parts[-6:])  # 控制上下文长度

    # 构造最终 prompt
    if context.strip():
        prompt = (
            "请基于以下上下文，回答用户的问题。\n\n"
            f"【上下文】\n{context}\n\n"
            f"【问题】\n{question}\n\n"
            "请给出清晰、结构化、直接的回答。"
        )
    else:
        prompt = (
            f"请直接回答以下问题：\n\n"
            f"{question}\n\n"
            "这是一个基础知识问题，不需要引用外部资料。"
        )

    # 获取回调
    callback = _stream_callback

    # 使用流式生成
    full_content = ""

    try:
        # 关键：使用 stream() 而不是 invoke()
        for chunk in llm.stream(prompt):
            # 不同的 LLM 返回格式可能不同
            token = ""
            if hasattr(chunk, "content"):
                token = chunk.content or ""
            elif isinstance(chunk, str):
                token = chunk
            elif isinstance(chunk, dict):
                token = chunk.get("content", "") or chunk.get("text", "")

            if token:
                full_content += token
                # 通过回调实时推送
                if callback:
                    try:
                        callback(token)
                    except Exception as e:
                        log.warning(f"Stream callback error: {e}")
    except Exception as e:
        log.error(f"Stream generation failed: {e}, falling back to invoke")
        # 降级到非流式
        try:
            response = llm.invoke(prompt)
            full_content = response.content
            if callback:
                callback(full_content)
        except Exception as e2:
            log.error(f"Invoke also failed: {e2}")
            full_content = f"生成回答时出错: {e2}"

    return {"messages": [AIMessage(content=full_content)]}