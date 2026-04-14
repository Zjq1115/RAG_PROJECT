# graph/graph1.py
import json
import uuid
from typing import Any, Dict, Literal, Optional

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from graph.agent_node import agent_node as _agent_node
from graph.generate_node import generate as _generate
from graph.get_human_message import get_last_human_message
from graph.graph_state1 import AgentState
from graph.rewrite_node import rewrite as _rewrite
from graph.web_search_node import web_search as _web_search
from llm_models.all_llm import llm
from tools.retriever_tools import retriever_tool
from utils.log_utils import log

# =============================
# 防循环关键参数
# =============================
MAX_REWRITE = 2  # rewrite 最大次数
MAX_STEPS = 20  # 整体保险丝
RECURSION_LIMIT = 80  # LangGraph recursion limit

GRAPH_NODES = [
    "START",
    "agent",
    "retrieve",
    "rewrite",
    "web_search",
    "generate",
    "END",
]

GRAPH_EDGES = [
    ("START", "agent"),
    ("agent", "retrieve"),
    ("agent", "web_search"),
    ("retrieve", "rewrite"),
    ("retrieve", "generate"),
    ("rewrite", "agent"),
    ("rewrite", "web_search"),
    ("web_search", "generate"),
    ("generate", "END"),
]


def make_config(thread_id: Optional[str] = None) -> Dict[str, Any]:
    return {
        "configurable": {"thread_id": thread_id or str(uuid.uuid4())},
        "recursion_limit": RECURSION_LIMIT,
    }


# =============================
# 节点包装器 - 关键修复：返回 current_node
# =============================
def wrap_node(name: str, fn):
    """
    包装节点函数，确保：
    1. current_node 被正确返回（而不是直接修改 state）
    2. root_question 被初始化
    3. step_count 被更新
    """

    def wrapped(state: Dict[str, Any]) -> Dict[str, Any]:
        # 初始化 root_question
        if not state.get("root_question"):
            try:
                state["root_question"] = get_last_human_message(
                    state.get("messages", [])
                ).content
            except Exception:
                pass

        # 调用原始函数
        result = fn(state)

        # 确保返回的是字典
        if not isinstance(result, dict):
            result = {}

        # 关键：在返回值中设置 current_node
        result["current_node"] = name
        result["step_count"] = int(state.get("step_count", 0)) + 1

        return result

    return wrapped


# =============================
# agent 节点
# =============================
def agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    result = _agent_node(state)
    result["current_node"] = "agent"
    result["step_count"] = int(state.get("step_count", 0)) + 1
    return result


# =============================
# retrieve 节点
# =============================
_tool_node = ToolNode([retriever_tool])


def retrieve_node(state: Dict[str, Any]) -> Dict[str, Any]:
    result = _tool_node.invoke(state)
    if not isinstance(result, dict):
        result = {"messages": result} if result else {}
    result["current_node"] = "retrieve"
    result["step_count"] = int(state.get("step_count", 0)) + 1
    return result


# =============================
# rewrite 节点（唯一维护 rewrite_count 的地方）
# =============================
def rewrite_node(state: Dict[str, Any]) -> Dict[str, Any]:
    rewrite_count = int(state.get("rewrite_count", 0)) + 1

    if rewrite_count >= MAX_REWRITE:
        log.info(f"---rewrite 达到上限({MAX_REWRITE})，强制转 web_search---")
        return {
            "current_node": "rewrite",
            "rewrite_count": 0,
            "force_web_search": True,
            "step_count": int(state.get("step_count", 0)) + 1,
        }

    out = _rewrite(state)
    msgs = out.get("messages", [])

    result = {
        "current_node": "rewrite",
        "rewrite_count": rewrite_count,
        "step_count": int(state.get("step_count", 0)) + 1,
    }

    if msgs:
        result["messages"] = [HumanMessage(content=msgs[-1].content)]

    return result


# =============================
# web_search 节点
# =============================
def web_search_node(state: Dict[str, Any]) -> Dict[str, Any]:
    out = _web_search(state)

    if not isinstance(out, dict):
        out = {}

    out["current_node"] = "web_search"
    out["force_web_search"] = False
    out["step_count"] = int(state.get("step_count", 0)) + 1

    return out


# =============================
# generate 节点
# =============================
def generate_node(state: Dict[str, Any]) -> Dict[str, Any]:
    result = _generate(state)

    if not isinstance(result, dict):
        result = {}

    result["current_node"] = "generate"
    result["step_count"] = int(state.get("step_count", 0)) + 1

    return result


# =============================
# grade_documents（只判断，不计数）
# =============================
def grade_documents(state: Dict[str, Any]) -> Literal["generate", "rewrite", "web_search"]:
    log.info("---检查 document 的相关性---")

    if int(state.get("step_count", 0)) >= MAX_STEPS:
        log.warning("---step_count 达到上限，强制转 web_search---")
        return "web_search"

    messages = state.get("messages", [])
    try:
        question = get_last_human_message(messages).content
    except Exception:
        question = ""

    llm_json = llm.bind(response_format={"type": "json_object"})

    prompt = PromptTemplate(
        template=(
            "你是一个评估检索文档与用户问题相关性的评分器。\n\n"
            "文档：\n{context}\n\n"
            "问题：\n{question}\n\n"
            "只返回 JSON：{{\"binary_score\": \"yes\"}} 或 {{\"binary_score\": \"no\"}}"
        ),
        input_variables=["context", "question"],
    )

    chain = prompt | llm_json

    last = messages[-1] if messages else None
    docs = getattr(last, "content", "") if last else ""

    result = chain.invoke({"question": question, "context": docs})

    try:
        score = json.loads(result.content).get("binary_score", "no")
    except Exception:
        score = "no"

    if score == "yes":
        print("---输出：文档相关---")
        return "generate"

    print("---输出：文档不相关---")
    return "rewrite"


# =============================
# agent 路由
# =============================
def agent_router(state: Dict[str, Any]) -> str:
    if state.get("force_web_search"):
        return "web_search"
    return tools_condition(state)


def rewrite_router(state: Dict[str, Any]) -> str:
    if state.get("force_web_search"):
        return "web_search"
    return "agent"


# =============================
# 构建 Graph
# =============================
workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("rewrite", rewrite_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("generate", generate_node)

workflow.add_edge(START, "agent")

workflow.add_conditional_edges(
    "agent",
    agent_router,
    {
        "web_search": "web_search",
        "tools": "retrieve",
        END: END,
    },
)

workflow.add_conditional_edges(
    "retrieve",
    grade_documents,
    {
        "generate": "generate",
        "rewrite": "rewrite",
        "web_search": "web_search",
    },
)

workflow.add_conditional_edges(
    "rewrite",
    rewrite_router,
    {
        "agent": "agent",
        "web_search": "web_search",
    },
)

workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

# -----------------------------
# 可选：本地命令行测试（不影响 server import）
# -----------------------------
if __name__ == "__main__":
    from utils.print_utils import _print_event

    config = make_config()
    _printed = set()

    while True:
        question = input("用户：").strip()
        if question.lower() in ["q", "quit", "exit"]:
            log.info("对话结束，拜拜！")
            break

        inputs = {
            "messages": [("user", question)],
            "rewrite_count": 0,
            "step_count": 0,
            "root_question": question,
        }
        events = graph.stream(inputs, config=config, stream_mode="values")
        for event in events:
            _print_event(event, _printed)