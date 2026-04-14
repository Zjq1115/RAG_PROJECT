from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from utils.env_utils import OPENAI_API_KEY
import requests
from typing import List
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from utils.env_utils import SILICONFLOW_API_KEY

# 官方openai嵌入模型
openai_embedding = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY,
    openai_api_base="https://xiaoai.plus/v1"
)

# # 本地bge嵌入模型
# model_name = "BAAI/bge-small-zh-v1.5"
# model_kwargs = {"device": "cpu"}
# encode_kwargs = {"normalize_embeddings": True}
# bge_embedding = HuggingFaceEmbeddings(
#     model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
# )

# 自定义SiliconFlow平台嵌入模型类
class SiliconFlowEmbeddings(Embeddings):
    def __init__(
            self,
            api_key: str = None,
            model: str = "BAAI/bge-large-zh-v1.5",
            base_url: str = "https://api.siliconflow.cn/v1/embeddings",
            batch_size: int = 20
    ):
        self.api_key = api_key or SILICONFLOW_API_KEY
        self.model = model
        self.base_url = base_url
        self.batch_size = batch_size  # 每批次处理的文本数量
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _get_embedding(self, texts: List[str]) -> List[List[float]]:
        """
        调用SiliconFlow API获取嵌入向量（支持分批处理）
        """
        all_embeddings = []

        # 分批处理，避免超出API限制
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            payload = {
                "model": self.model,
                "input": batch,
                "encoding_format": "float"
            }

            response = requests.post(self.base_url, json=payload, headers=self.headers)
            response.raise_for_status()  # 如果请求失败则抛出异常

            result = response.json()
            # SiliconFlow返回格式与OpenAI兼容，从data字段提取embedding
            embeddings = [item["embedding"] for item in result["data"]]
            all_embeddings.extend(embeddings)

        return all_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        对文档列表进行嵌入（LangChain要求实现的方法）
        """
        return self._get_embedding(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        对单个查询进行嵌入（LangChain要求实现的方法）
        """
        return self._get_embedding([text])[0]


# 创建SiliconFlow嵌入模型实例（替代原来的openai_embedding）
siliconflow_embedding = SiliconFlowEmbeddings()
