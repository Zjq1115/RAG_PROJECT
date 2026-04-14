from langchain_core.tools import tool
from pymilvus import MilvusClient

from utils.env_utils import MILVUS_URI, MILVUS_TOKEN, COLLECTION_NAME
from llm_models.embeddings_model import siliconflow_embedding


# 只初始化一次客户端（模块加载时）
_client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)


@tool("rag_retriever")
def retriever_tool(query: str) -> str:
    """
    搜索并返回关于半导体和芯片的信息，涵盖封装、测试、光刻胶等内容
    """
    # 1) 生成 query 的 dense 向量（硅基 Embedding API）
    qvec = siliconflow_embedding.embed_query(query)

    # 2) 向量检索（dense）
    # 你的 schema: text / dense / sparse / metadata(JSON)
    # JSON filter 必须是字符串表达式
    results = _client.search(
        collection_name=COLLECTION_NAME,
        data=[qvec],
        anns_field="dense",
        limit=4,
        output_fields=["text", "metadata"],
        filter='metadata["category"] == "content"',   # 如果你只想要 content
    )[0]

    # 3) 拼装返回给 LLM（create_retriever_tool 原来返回的是 docs，这里返回文本即可）
    if not results:
        return "未检索到相关内容。"

    chunks = []
    for hit in results:
        entity = hit.get("entity", {}) or {}
        text = entity.get("text", "")
        md = entity.get("metadata", {}) or {}
        score = hit.get("distance", None)
        chunks.append(f"[score={score}] {text}\nmeta={md}")

    return "\n\n".join(chunks)