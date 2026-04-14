import uuid
import json  # 顶部新增
from typing import Literal, List

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from graph.agent_node import agent_node
from graph.generate_node import generate
from graph.get_human_message import get_last_human_message
from graph.graph_state1 import AgentState, Grade
from graph.rewrite_node import rewrite
from llm_models.all_llm import llm
from tools.retriever_tools import retriever_tool
from utils.log_utils import log
from utils.print_utils import _print_event
from graph.web_search_node import web_search

# 全局计数器，用于跟踪重写次数
rewrite_counter = {}

def grade_documents(state) -> Literal["generate", "rewrite"]:
    """
    判断检索到的文档是否与问题相关。
    参数:
        state (messages): 当前状态
    返回:
        str: 判断结果，文档是否相关
    """
    log.info("---检查document的相关性---")

    # 获取当前线程ID用于跟踪重写次数
    messages = state["messages"]
    question = get_last_human_message(messages).content

    # 检查重写次数（最多重写2次）
    global rewrite_counter
    if question not in rewrite_counter:
        rewrite_counter[question] = 0

    # 如果已经重写超过2次，直接生成回答
    if rewrite_counter[question] >= 2:
        print("---已达到最大重写次数，转为联网搜索---")
        rewrite_counter[question] = 0  # 重置计数器
        return "web_search"

    #  带结构化输出的LLM
    # llm_with_structured = llm.with_structured_output(Grade)

    # 使用 DeepSeek 的 JSON Mode
    llm_json = llm.bind(response_format={"type": "json_object"})

    # 提示模板
    prompt = PromptTemplate(
        template="""你是一个评估检索文档与用户问题相关性的评分器。\n
            这是检索到的文档：\n\n {context} \n\n
            这是用户的问题：{question} \n
            请以 json 格式返回评分结果，格式为：{{"binary_score": "yes"}} 或 {{"binary_score": "no"}}
            只返回json，不要有其他内容。""",
        input_variables=["context", "question"],
    )

    # 处理链
    # chain = prompt | llm_with_structured
    chain = prompt | llm_json

    messages = state["messages"]
    last_message = messages[-1]

    question = get_last_human_message(messages).content
    docs = last_message.content

    # scored_result = chain.invoke({"question": question, "context": docs})
    # score = scored_result.binary_score

    result = chain.invoke({"question": question, "context": docs})
    result_text = result.content
    parsed = json.loads(result_text)
    score = parsed.get("binary_score", "no").lower()

    if score == "yes":
        print("---输出：文档相关---")
        rewrite_counter[question] = 0  # 重置计数器
        return "generate"

    else:
        print("---输出：文档不相关---")
        rewrite_counter[question] += 1
        print(score)
        return "rewrite"

# 定义一个新的工作流图
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node('agent', agent_node)
workflow.add_node('retrieve', ToolNode([retriever_tool]))
workflow.add_node('rewrite', rewrite)
workflow.add_node('generate', generate)
workflow.add_node('web_search', web_search)

workflow.add_edge(START, 'agent')
workflow.add_conditional_edges(
    'agent',
    tools_condition,
    {
        'tools': 'retrieve',
        END: END
    }
)

workflow.add_conditional_edges(
    'retrieve',
    grade_documents,{
        "generate": "generate",
        "rewrite": "rewrite",
        "web_search": "web_search"
    }
)

workflow.add_edge('rewrite', 'agent')
workflow.add_edge('web_search', 'generate')
workflow.add_edge('generate', END)

# 检查点让状态图可以持久化其状态
# 这是整个状态图的完整内存
memory = MemorySaver()

# 编译状态图，配置检查点为memory, 配置中断点
graph = workflow.compile(checkpointer=memory)

config = {
    "configurable": {
        # 检查点由session_id访问
        "thread_id": str(uuid.uuid4()),
    }
}

_printed = set()  # set集合，避免重复打印

# 执行工作流
while True:
    question = input('用户：')
    if question.lower() in ['q', 'exit', 'quit']:
        log.info('对话结束，拜拜！')
        break
    else:
        inputs = {
            "messages": [
                ("user", question),
            ]
        }
        events = graph.stream(inputs, config=config, stream_mode='values')
        # 打印消息
        for event in events:
            _print_event(event, _printed)