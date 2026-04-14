# 基于LangChain的RAG+联网搜索领域知识回复系统

**声明：因实习保密要求，本项目的任何数据与代码均不涉及国家、军事、商业机密，公开数据集来源于modelscope和github，代码为个人所编写的可供公开学习使用的代码。**
  
本项目基于 **LangGraph** 构建了一套模块化、可扩展的 **RAG（Retrieval-Augmented Generation）智能问答与文档生成系统**。系统融合 **向量检索、查询重写与联网搜索** 等多种信息获取机制，并通过 **多节点状态机工作流** 实现自动决策与动态路由。

项目分为两个部分：

* 第一部分：**离线数据处理阶段**，这一阶段的目标是将企业文档转化为可检索的向量化知识库。主要步骤包括：多格式文档解析、标题合并与语义切分、向量化与表结构构建、Milvus向量数据库插入这四个流程。
* 第二部分：**在线推理阶段**，目标是根据用户所提问题，执行一个工作流，给用户返回更加准确的答案。工作流包含多个关键节点：路由决策节点、混合检索与精排节点、相关性评估与问题重写节点、联网搜索节点、流失生成节点。

## 一、项目架构与展示

### 1. 平台可视化

![image-20260304180850436](/pics/image-20260304180850436.png)


### 2. 项目架构

![项目架构图](/pics/项目架构图.png)

### 3、快速介绍

项目整体分为两部分，一部分是离线的数据处理以及向量检索知识库的构造与写入，第二部分是用户提问问题、模型的在线推理；
* 第一阶段，我们对三种格式的文档进行结构化解析，将其初步拆分成document对象，每个对象包括文本内容以及对应的属性，比如类别、id、父id等。然后对这些document对象按照父id去构建“父标题、子标题”这样的具有完整上下文的语义块，如果语义块过长，我们对语义块中相邻句子计算余弦相似度，在语义断裂处进行拆分。以上完成对文档的处理，然后我们调用嵌入大模型bge生成语义块内容对应的语义向量，调用内置的BM25生成关键词向量，并结合语义块的id、文本和属性构造向量检索数据表的关键字段，存入Milvus向量数据库中。
* 第二阶段是在线推理阶段，我们基于LangGraph构建了一个由决策、检索、重写、联网搜素节点、生成节点组成的完整的工作流。首先，用户提问问题，采用基于LLM的function calling机制做路由决策，将向量检索工具绑定在LLM上，由LLM决定用户当前的问题是否需要在知识库中检索，如果不需要，直接进入生成节点；如果需要，进行检索节点，我们采用混合检索策略，包括语义向量和关键词向量，各自检索Top-20条，将排名进行RRF融合排序后，调用rerank模型对20条候选进行精排取top-4作为返回结果；然后由LLM判断检索结果是否和用户问题相关，如果相关进入生成节点；如果不相关进入重写节点，我们考虑到可能用户本身问的问题口语化较重、太泛或者缺少关键词，我们构造提示词要求LLM对用户问题改写地更具体，然后将改写的问题进入决策节点循环执行，循环次数超过两次，认为当前知识库缺少对应的知识，进入联网搜索节点，调用SerpAPI平台的API去互联网搜索n条结果，然后进入生成节点；生成节点从消息列表中取历史消息、中间消息（检索结果、联网搜索结果）和用户原始问题组装上下文给LLM做最终回复。


## 二、基本环境要求与配置

| 组件     | 版本要求 | 说明                          |
| -------- | -------- | ----------------------------- |
| Windows  | 10/11    | 主操作系统                    |
| Python   | 3.11     | 运行环境                      |
| PyCharm  | 2023.x+  | IDE（可选）                   |
| Zilliz   | —        | Milvus云托管平台              |
| Silicon  | —        | 大模型API平台（用于语义嵌入） |
| SerpApi  | —        | 联网搜索API平台               |
| Deepseek | —        | 基础大模型（用于对话）        |

### 1. Python 环境配置

#### 1.1 创建项目虚拟环境

```bash
# 创建环境
conda create -n rag_agent python=3.11 -y

# 激活环境
conda activate rag_agent
```

#### 1.2 安装项目依赖

将requirement.txt拉入当前目录，并安装依赖：

```bash
pip install -r requirements.txt
```

------

### 2. Milvus 向量数据库配置

由于本地资源受限，我们选择 Milvus 云端托管平台 Zilliz 完成本地向量数据的写入与检索，在Zilliz创建向量检索数据库集群 rag_cloud：

![image-20260304170335291](/pics/image-20260304170335291.png)

------

## 三、代码结构

```python
RAG_PROJECT/
│
├── datas/                     # 数据集
│   ├── docx/     		            # docx文档数据集
│   ├── md/     		            # markdown文档数据集
│   ├── pdf/     		            # pdf文档数据集
│
├── documents/                 # 知识库构建模块
│   ├── markdown_parser.py     		# Markdown格式文档解析
│   ├── word_parser.py				# word格式文档解析
│   ├── pdf_parser.py				# pdf格式文档解析
│   ├── milvus_db.py           		# 文档切分、写入Milvus向量数据库、混合检索方法
│
├── graph/                     # LangGraph 工作流模块
│   ├── graph1.py              		# 主工作流定义
│   ├── graph_state1.py        		# 状态定义
│   ├── agent_node.py          		# Agent 决策节点
│   ├── generate_node.py       		# 答案生成节点（流式）
│   ├── rewrite_node.py        		# 查询重写节点
│   ├── web_search_node.py     		# 联网搜索节点
│   ├── doc_processor.py       		# Word 文档处理模块
│   └── get_human_message.py   		# 消息处理工具
│
├── llm_models/                # LLM 模型配置
│   ├── embeddings_model.py         # 嵌入模型实例化
│   ├── ocr_model.py                # 图片识别模型实例化
│   └── all_llm.py             		# LLM 实例化
│
├── tools/                     # 工具定义
│   └── retriever_tools.py     		# 向量检索工具
│
├── utils/                     # 工具函数
│   ├── log_utils.py           		# 日志配置
│   └── print_utils.py         		# 打印工具
│
├── index.html                 # 前端页面
├── server.py                  # 后端服务
├── .env                       # 环境变量配置
└── README.md                  # 项目文档
```

------

## 四、核心功能介绍

### 1、文档处理与知识库写入

**（1）markdown文档处理（markdown_parser.py + milvus_db.py）**

在构建向量检索知识库之前，首先需要对 Markdown 文档进行结构化解析、语义切分，并最终写入 Milvus 向量数据库。

假设我们有如下 Markdown 文档：

```markdown
# 安装指南
	这是总览介绍。
## 环境要求
	- Python 3.10+
	- Docker
## 启动方式
	先启动 Milvus，再启动服务。
```

* **第一步：结构化解析（Markdown → Document 列表）**，将Markdown文档的文本按照句子拆分成 Document(page_content, metadata) 列表，每条document包含拆分出的文本和该条文本的一些属性，比如类别、id、父id等等，如下：

```markdown
Document("安装指南", metadata={category:"Title", element_id:"t1", parent_id: None})
Document("这是总览介绍。", metadata={category:"NarrativeText", element_id:"p1", parent_id:"t1"})
Document("环境要求", metadata={category:"Title", element_id:"t2", parent_id:"t1"})
Document("Python 3.10+", metadata={category:"ListItem", element_id:"l1", parent_id:"t2"})
```

* **第二步：标题路径合并（构建语义单元 Chunk）**，在结构化解析的基础上，系统将标题视为“语义容器”，并将其下的子内容拼接，构建具有完整上下文的语义块（Chunk）。同时构造“父标题 → 子标题”路径，使语义更加完整。

```
chunk A（t1）
"安装指南 这是总览介绍。"
chunk B（t2）
"安装指南 这是总览介绍。 -> 环境要求 Python 3.10+ Docker"
chunk C（t3）
"安装指南 这是总览介绍。 -> 启动方式 先启动 Milvus，再启动服务。"
```

* **第三步：语义切分（长文本智能拆分）**，如果chunk太长，系统会进行语义级别切分。先按句子切分，计算相邻句子的语义相似度，在语义断裂处拆成多个更小的子chunk。

* **第四步：构建 Milvus 表结构（云端托管）**，系统在 Zilliz Cloud（Milvus 托管版）中创建如下表结构：

| 字段名     | 类型                | 来源                    | 含义说明     |
| ---------- | ------------------- | ----------------------- | ------------ |
| `id`       | INT64 (AutoID)      | 系统生成                | 主键         |
| `text`     | VARCHAR             | `Document.page_content` | Chunk全文    |
| `dense`    | FLOAT_VECTOR (1024) | 由嵌入大模型生成        | 语义向量     |
| `sparse`   | SPARSE_FLOAT_VECTOR | 由Milvus内置的BM26生成  | 关键词向量   |
| `metadata` | JSON                | `Document.metadata`     | 所有结构属性 |

* **第五步：向量生成与入库**，对于每一条chunk，调嵌入大模型生成语义dense嵌入向量，Milvus调BM25生成稀疏关键词向量，提取出其他的字段内容，生成完整的一条Milvus数据。如下所示：

| id   | text                                    | dense    | sparse   | metadata             |
| ---- | --------------------------------------- | -------- | -------- | -------------------- |
| 1    | 安装指南 这是总览介绍                   | （向量） | （BM25） | {category: Title...} |
| 2    | 安装指南 → 环境要求 Python 3.10+ Docker | （向量） | （BM25） | {category: Title...} |
| 3    | 安装指南 → 启动方式 Milvus              | （向量） | （BM25） | {category: Title...} |

### 2、RAG工作流

RAG工作流主要节点包括如下：

* agent节点决定要不要走检索工具
* retrieve节点去 Milvus 向量库检索（含 dense+BM25 sparse 的混合检索/重排） 
* grade_documents 判断检索结果是否相关（相关→生成，不相关→改写问题） 
* rewrite 改写用户问题（最多 2 次，避免死循环） 
* web_search 用 Tavily 做联网搜索兜底 
* generate 基于上下文生成最终回答（支持逐 token 流式输出）

以下是各节点的主要功能介绍：

（1）Agent 节点 (`agent_node.py`)

负责决策是否需要调用向量检索工具，让 LLM 用 ”tool_calls/不 tool_calls” 这种结构化输出，来表达 “要不要检索”。图再根据这个结构化输出分流：要检索就进 retrieve，不要就结束。

```python
def agent_node(state: AgentState):
    """
    - 构建安全的多轮对话上下文
    - 过滤 ToolMessage，避免消息序列非法
    - 绑定 retriever_tool 让模型决策
    """
    ...
    model = llm.bind_tools([retriever_tool])
	response = model.invoke(chat_ctx)
    ...
```

（2）向量检索 (`retriever_tools.py`)

基于 Milvus 的向量检索工具，进行语义向量检索（调用bge-m3大模型API）和关键词向量检索（BM25），最后进行RRF融合排序，返回前k条：

```python
retriever_tool = Tool(
    name="retriever",
    description="检索与问题相关的文档",
    func=retrieve_documents
)
```

（3）文档相关性评估 (`grade_documents`)

使用 LLM 判断检索结果是否相关，让LLM读取用户的问题以及检索的文档内容，判断是否相关：

```python
def grade_documents(state) -> Literal["generate", "rewrite", "web_search"]:
    """
    返回:
    - "generate": 文档相关，直接生成答案
    - "rewrite": 文档不相关，重写查询
    - "web_search": 达到重写上限，转联网搜索
    """
```

（4）查询重写 (`rewrite_node.py`)

当检索结果不相关时，可能是用户问得问题太口语、太泛、缺关键词，导致 Milvus 检索不到真正相关的片段。我们构造一个 prompt，要求模型把问题改写得更具体、更可检索，然后调用 LLM 得到一个更好的问题，如果改写次数超限（2 次）就停止，直接走联网搜索：

```python
def rewrite(state):
    """
    分析原始问题的语义意图，生成改进后的查询
    """
```

（5）联网搜索 (`web_search_node.py`)

当本地知识库（Milvus）检索不到足够相关内容时，就用 Tavily 去互联网上搜 3 条结果，把结果整理成文本，当作上下文交给 generate 生成最终回答：

```python
def web_search(state):
    """
    - 支持多种返回格式标准化
    - 自动处理错误情况
    """
```

（6）流式生成 (`generate_node.py`)

支持流式输出的答案生成：

```python
def generate(state):
    """
    - 使用 llm.stream() 逐 token 生成
    - 通过回调函数实时推送
    - 支持降级到非流式模式
    """
```

------

## 五、使用方法

1、构建知识库（数据入库）：修改 `documents/write_milvus.py` 中的文件路径配置：

```python
if __name__ == '__main__':
    # 修改为你存放 Markdown 文件的实际路径
    md_dir = r'/path/to/your/markdown/files' 
    # ...
```

在 PyCharm 或终端中运行写入脚本：

```shell
python documents/write_milvus.py
```

脚本将启动双进程：进程 1 解析 Markdown 并进行语义切片，进程 2 将 Dense+Sparse 向量批量写入 Milvus。

2、在pycharm中启动项目根目录下的**server.py**。

3、在浏览器输入本地地址访问平台。
