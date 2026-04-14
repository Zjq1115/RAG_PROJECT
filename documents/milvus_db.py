from __future__ import annotations

from typing import List, Dict, Any
from langchain_core.documents import Document

from pymilvus import MilvusClient, Function
from pymilvus.client.types import DataType, FunctionType

from documents.markdown_parser import MarkdownParser
from llm_models.embeddings_model import siliconflow_embedding
from utils.env_utils import MILVUS_URI, MILVUS_TOKEN, COLLECTION_NAME


DENSE_DIM = 1024


class MilvusVectorSave:
    """把解析后的 Document 写入云端 Milvus（Zilliz Cloud）"""

    def __init__(self) -> None:
        if not MILVUS_URI or not MILVUS_TOKEN:
            raise RuntimeError("MILVUS_URI / MILVUS_TOKEN 未配置，请检查 .env 与 env_utils.py")
        if not COLLECTION_NAME:
            raise RuntimeError("COLLECTION_NAME 未配置，请在 .env 中设置 COLLECTION_NAME")

        self.client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)

    def create_collection(self, drop_old: bool = False) -> None:
        """创建 collection（仅 text + dense + sparse(BM25) + metadata(JSON)）"""
        if self.client.has_collection(COLLECTION_NAME):
            if drop_old:
                self.client.drop_collection(COLLECTION_NAME)
                print(f"已删除旧 collection: {COLLECTION_NAME}")
            else:
                print(f"collection 已存在，跳过创建: {COLLECTION_NAME}")
                return

        schema = self.client.create_schema()
        schema.enable_dynamic_field = True  # 允许额外动态字段（可选，但开着没坏处）

        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(
            field_name="text",
            datatype=DataType.VARCHAR,
            max_length=6000,
            enable_analyzer=True,
            analyzer_params={"tokenizer": "jieba", "filter": ["cnalphanumonly"]},
        )

        # BM25 输出字段
        schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)
        # Dense 向量字段
        schema.add_field(field_name="dense", datatype=DataType.FLOAT_VECTOR, dim=DENSE_DIM)

        # 所有业务元数据放这里（JSON），方便 filter：metadata["category"] == "Title"
        schema.add_field(field_name="metadata", datatype=DataType.JSON)

        # BM25 function：text -> sparse
        bm25_function = Function(
            name="text_bm25_emb",
            input_field_names=["text"],
            output_field_names=["sparse"],
            function_type=FunctionType.BM25,
        )
        schema.add_function(bm25_function)

        # 索引
        index_params = self.client.prepare_index_params()

        index_params.add_index(
            field_name="sparse",
            index_name="sparse_inverted_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
            params={"drop_ratio_build": 0.2},
        )

        index_params.add_index(
            field_name="dense",
            index_name="dense_index",
            index_type="HNSW",
            metric_type="IP",
            params={"M": 16, "efConstruction": 64},
        )

        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema,
            index_params=index_params,
        )
        print(f"成功创建 collection: {COLLECTION_NAME}")

    @staticmethod
    def _sanitize_metadata(md: Dict[str, Any]) -> Dict[str, Any]:
        """
        清洗 metadata，避免出现不可 JSON 序列化对象或循环引用。
        - 删除 languages（你 parser 里已经删过，但这里双保险）
        - 确保值都是基础类型 / dict / list
        """
        if not md:
            return {}

        md = dict(md)
        md.pop("languages", None)

        # 防止用户自己塞了奇怪对象：全部转成可序列化的基础类型
        def to_jsonable(x):
            if x is None or isinstance(x, (str, int, float, bool)):
                return x
            if isinstance(x, dict):
                return {str(k): to_jsonable(v) for k, v in x.items()}
            if isinstance(x, list):
                return [to_jsonable(v) for v in x]
            # 其他类型转字符串兜底
            return str(x)

        return to_jsonable(md)

    def insert_documents(self, docs: List[Document], batch_size: int = 32) -> None:
        """把 Document 批量写入 Milvus：显式写 text + dense + metadata(JSON)，BM25 sparse 由函数生成"""
        if not docs:
            print("docs 为空，跳过写入")
            return

        texts = [d.page_content for d in docs]
        metadatas = [self._sanitize_metadata(d.metadata or {}) for d in docs]

        # 生成 dense 向量（一次性批量）
        dense_vectors = siliconflow_embedding.embed_documents(texts)

        # 组装插入数据：不传 sparse，让 BM25 function 自动生成
        rows = [{"text": t, "dense": v, "metadata": m} for t, v, m in zip(texts, dense_vectors, metadatas)]

        # 分批插入
        total = len(rows)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            self.client.insert(collection_name=COLLECTION_NAME, data=rows[start:end])
            print(f"已插入 {end}/{total}")

    def verify_query(self, limit: int = 5) -> None:
        """简单验证：查 Title 类 chunk"""
        res = self.client.query(
            collection_name=COLLECTION_NAME,
            filter='metadata["category"] == "Title"',
            output_fields=["text", "metadata"],
            limit=limit,
        )
        print(f"验证查询结果（最多 {limit} 条）:")
        for r in res:
            print("-", r.get("text"), "| category:", (r.get("metadata") or {}).get("category"))


if __name__ == "__main__":
    # 1) 解析 markdown
    file_path = "../datas/md/tech_report_0tfhhamx.md"
    parser = MarkdownParser()
    docs = parser.parse_markdown_to_documents(file_path)

    # 2) 写入云端 Milvus
    mv = MilvusVectorSave()
    mv.create_collection(drop_old=True)  # 需要重建时改 True
    mv.insert_documents(docs, batch_size=16)

    # 3) 验证查询
    mv.verify_query(limit=5)