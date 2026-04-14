# from typing import TypedDict, Annotated
# from langchain_core.messages import BaseMessage
# from langgraph.graph import add_messages
# from pydantic import BaseModel, Field
#
#
# class AgentState(TypedDict):
#     # add_messages 函数定义了应如何处理状态更新
#     # add_messages 表示"追加"（append）
#     messages: Annotated[list[BaseMessage], add_messages]
#
#
# # 数据模型
# class Grade(BaseModel):
#     """相关性检查的二元评分"""
#
#     binary_score: str = Field(description="相关性评分 'yes' 或 'no'")

from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field


class AgentState(TypedDict, total=False):
    """
    total=False:
    - 允许 state 中缺省字段
    - 但字段一旦声明，LangGraph 才会持久化它
    """

    # 对话消息（LangGraph 负责 append）
    messages: Annotated[list[BaseMessage], add_messages]

    # ====== 为防循环 / 可视化 / 控制流新增 ======
    current_node: str        # 当前节点名（前端高亮用）
    root_question: str       # 本轮请求的原始问题
    rewrite_count: int       # rewrite 次数（严格受控）
    step_count: int          # 节点总步数保险丝
    force_web_search: bool   # 达到 rewrite 上限后强制转 web_search


# 相关性评分模型（供 PromptTemplate 使用）
class Grade(BaseModel):
    """相关性检查的二元评分"""
    binary_score: str = Field(description="相关性评分 'yes' 或 'no'")
