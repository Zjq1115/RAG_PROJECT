import os
from dotenv import load_dotenv

load_dotenv(override=True)

# for base chat LLM
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')

# for cloud milvus vector db
MILVUS_URI = os.getenv("ZILLIZ_URI")
MILVUS_TOKEN = os.getenv("ZILLIZ_TOKEN")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

# for beg-m3 embedding LLM
SILICONFLOW_API_KEY = os.getenv('SILICONFLOW_API_KEY')

# for web-search tools
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')