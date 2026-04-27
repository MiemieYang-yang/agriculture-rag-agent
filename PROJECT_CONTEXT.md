# 农业气候与资源数据专家助手 — 项目上下文

> 每次开启新对话，把本文件内容贴给 Claude，即可恢复完整项目上下文。
> 每次有改动，在「变更日志」中追加记录。

---

## 一、背景与目标

**求职方向：** 大模型 Agent 应用开发岗（大厂）
**核心困境：** 缺乏实际项目经验，只有从 GitHub 拿来的简单智能客服项目，竞争力不足
**解决策略：** 自主开发「基于 RAG 架构的农业气候与资源数据专家助手」，迭代式开发，每完成一个阶段就更新简历

**项目定位：**
- 垂直领域 RAG 系统（农业气候 + 作物种植 + 资源数据）
- 不追求一步到位，每阶段完成后可独立演示、独立写进简历
- 技术选型贴近大厂面试考点

---

## 二、技术栈

| 组件 | 选型 | 说明 |
|------|------|------|
| 向量模型 | BGE-M3 | 多语言，中文效果优秀 |
| 向量数据库 | ChromaDB | 本地持久化，无需单独部署 |
| 文本切分 | MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter | 标题层级感知切分，保留元数据 |
| LLM | Qwen（通义千问） | 兼容 OpenAI 接口 |
| API 框架 | FastAPI | 自带 Swagger 文档 |
| 前端框架 | Streamlit | 快速构建交互式 Web 界面，界面更美观 |
| 包版本锁定 | FlagEmbedding==1.2.11, transformers==4.44.2 | 避免兼容性问题 |

---

## 三、项目文件结构

```
agri_rag/
├── core/
│   ├── config.py               # 统一配置，所有参数从 .env 读取
│   ├── document_processor.py   # 文档加载（PDF/TXT/MD）+ 切分
│   ├── embedder.py             # BGE-M3 向量化（懒加载）
│   ├── vector_store.py         # ChromaDB 写入 / 检索 / MD5去重
│   ├── llm_client.py           # Qwen 同步 + 流式调用
│   └── rag_pipeline.py         # 核心 RAG 链路 ★
├── api/
│   └── routes.py               # FastAPI 接口
├── data/
│   └── raw/                    # 原始文档目录
│       └── crop_climate_guide.md  # 示例农业文档
├── vectorstore/                # ChromaDB 持久化数据（自动生成）
├── .env.example                # 配置模板
├── ingest.py                   # 一键入库脚本
├── main.py                     # 启动入口（CLI / API / Agent / Gradio / Streamlit 五模式）
├── gradio_app.py               # Gradio 前端界面
├── streamlit_app.py            # Streamlit 前端界面 ★
├── requirements.txt            # 依赖（版本已锁定）
└── README.md                   # 项目文档
```

---

## 四、迭代路线图

### Phase 1 · 基础 RAG ✅ 已完成
**目标：** 跑通完整链路，能演示，能回答真实农业问题

**已实现功能：**
- 文档处理：支持 PDF / TXT / MD 加载，RecursiveCharacterTextSplitter 切分（chunk_size=512, overlap=64）
- 向量化：BGE-M3 批量编码，L2 归一化，懒加载
- 向量库：ChromaDB cosine 索引，MD5 去重，元数据过滤预留
- RAG 链路：Query → 检索 → 相似度过滤（< 0.4 丢弃）→ Prompt 拼接 → Qwen 生成
- 接口：FastAPI RESTful API + CLI 交互模式
- 多轮对话：历史维护，自动截断（保留最近 10 轮）

**此阶段简历写法：**
> 构建农业气候垂直领域知识库，使用 LangChain RecursiveCharacterTextSplitter 进行语义感知切分（chunk_size=512, overlap=64），采用 BGE-M3 多语言向量模型完成文本 Embedding，存入 ChromaDB 本地持久化向量库（cosine 相似度索引）；设计 RAG Pipeline 实现 Query 编码 → Top-K 语义检索 → 相似度阈值过滤（< 0.4 丢弃）→ Prompt 拼接 → Qwen LLM 生成的完整链路，基于 FastAPI 提供 RESTful 接口。

---

### Phase 2 · 检索优化 ✅ 已完成
**目标：** 有数据、有对比、能量化改进效果（面试最大亮点）

**已实现功能：**
- 评估测试集：50 条农业领域 Q&A 对，覆盖气候资源、区划、作物发育期等主题
- RAGAS 框架评估：Faithfulness / Answer Relevancy / Context Precision / Context Recall 四指标
- 混合检索：BM25（关键词匹配）+ 向量检索（语义匹配），RRF 融合
- HyDE（假设文档嵌入）：LLM 先生成假设答案，用它做检索向量
- BGE-Reranker-v2：对 Top-K 结果重排，粗排 50 条精排到 5 条
- 对比实验脚本：支持 7 组配置对比（Baseline / +Hybrid / +HyDE / +Reranker 等）

**新增文件：**
- `core/reranker.py`：BGE-Reranker-v2 重排序模块
- `core/bm25_store.py`：BM25 稀疏检索模块
- `core/hybrid_retriever.py`：混合检索 RRF 融合模块
- `evaluation/ragas_evaluator.py`：RAGAS 评估脚本
- `evaluation/run_comparison.py`：对比实验脚本

**此阶段简历写法：**
> 构建 50 条农业领域评估集，实现混合检索（BM25 + 向量检索 RRF 融合），引入 HyDE 与 BGE-Reranker 对检索链路优化，Answer Relevancy 从 X% 提升至 Y%，Context Recall 提升 Z 个百分点，使用 RAGAS 框架完成量化评估。

---

### Phase 3 · Agent 能力 ✅ 已完成
**目标：** 引入工具调用，展示 Agent 决策能力

**已实现功能：**
- 工具封装：天气查询（模拟数据）、农学指标计算（积温/降水）、知识库检索
- Qwen Tool Calling：扩展 llm_client.py 支持 chat_with_tools() 和 submit_tool_results()
- ReAct 循环：意图识别 → 工具路由 → 执行 → 结果整合 → 回复
- 多轮追问：AgentContext 维护作物/地点/时间实体，支持"那玉米呢？"追问
- 异常处理：最大迭代次数限制（5次）、工具执行异常捕获、参数校验

**新增文件：**
- `core/tools/base.py`：工具基类定义（ToolResult、BaseTool）
- `core/tools/weather_tool.py`：天气查询工具（模拟中国主要农业城市数据）
- `core/tools/agri_calculator.py`：农学计算工具（积温、降水统计、发育期推算）
- `core/tools/knowledge_search.py`：知识库检索工具
- `core/agent/agent.py`：Agent 核心逻辑（ReAct 循环、多轮追问）
- `core/agent/tool_registry.py`：工具注册中心
- `core/agent/prompts.py`：Agent Prompt 模板

**扩展文件：**
- `core/llm_client.py`：添加 chat_with_tools() 和 submit_tool_results() 方法
- `core/config.py`：添加 AGENT_MAX_ITERATIONS、AGENT_ENABLE_TOOLS 等配置
- `api/routes.py`：添加 /api/agent/query 和 /api/agent/tools 接口
- `main.py`：添加 --agent 参数，支持 Agent CLI 模式

**此阶段简历写法：**
> 在 RAG 基础上引入 Agent 能力，封装天气查询、农学指标计算等工具，基于 Qwen Tool Calling 实现意图识别 → 工具路由 → 结果整合的自主决策流程，支持多轮追问，处理工具调用异常与格式校验，最大迭代次数限制防止无限循环。

---

### Phase 4 · 工程化 🚧 进行中
**目标：** 体现生产交付意识，锦上添花

**已实现功能：**
- [x] Gradio 前端界面：Agent/RAG 模式切换、多轮对话、工具调用展示、引用来源展示

**计划任务：**
- [ ] 流式输出（Streaming）：边生成边返回
- [ ] LangSmith 链路追踪与日志
- [ ] 增量知识库更新（新文档自动入库，不重建索引）
- [ ] Docker 容器化部署

**此阶段简历写法：**
> 使用 Streamlit 构建交互式 Web 界面，支持 Agent/RAG 双模式切换，实时展示工具调用链路与引用来源，提供开箱即用的演示能力。

---

## 五、关键设计决策（面试时重点讲）

| 决策点 | 做法 | 为什么这样做 |
|--------|------|--------------|
| 相似度阈值过滤 | < 0.4 的检索结果直接丢弃 | 避免低质量上下文导致 LLM 幻觉 |
| 向量 L2 归一化 | encode 后统一归一化 | 余弦相似度等价于点积，ChromaDB 计算更快 |
| MD5 去重 | 用内容哈希作为文档 ID | 重复运行 ingest.py 不会写入重复文档 |
| BGE-M3 懒加载 | 第一次调用 encode 时才初始化 | 避免服务启动时就占用 2GB 内存 |
| chunk overlap=64 | 相邻 chunk 重叠 64 字符 | 保证跨 chunk 边界的语义连续性 |

---

## 六、已知问题与解决方案

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| `ImportError: cannot import name 'is_torch_fx_available'` | FlagEmbedding 新版与 transformers 不兼容 | 锁定 `FlagEmbedding==1.2.11` + `transformers==4.44.2` |
| Gradio 6.0 `css`/`theme` 参数报错 | Gradio 6.0 API 大幅变化 | 移除不兼容参数，使用 `fill_height=True` |
| Gradio 6.0 `show_copy_button` 报错 | 参数改为 `buttons=["copy"]` | 更新 Chatbot 组件参数 |
| Gradio 6.0 `type` 参数报错 | Gradio 6.0 默认使用 Message 格式 | 移除 `type="messages"` 参数 |
| Gradio 6.0 examples 格式错误 | 需使用 `{"text": "问题"}` 格式 | 更新示例问题数据结构 |

---

## 七、变更日志

### v0.4.1 — 文档处理模块重构（标题层级保留）
- 重构 `core/document_processor.py`：
  - 新增 Word (.docx) 文档支持
  - 使用 `mammoth` 库将 .docx 转换为 Markdown（Heading 1/2/3/4 → # ## ### ####）
  - 目录行过滤（正则匹配 `.+[．.…]{2,}\d+\s*$` 格式行）
  - 主切分器：`MarkdownHeaderTextSplitter`（识别 # ## ### #### 四级标题）
  - 二次切分器：`RecursiveCharacterTextSplitter`（处理超长 chunk）
  - 表格保护逻辑（二次切分时不切分 Markdown 表格）
  - 新增元数据字段：`h1`, `h2`, `h3`, `h4`（标题层级文本）
  - 过滤 Word 临时文件（以 `~$` 开头）
- 更新 `requirements.txt`：添加 `mammoth>=1.6.0` 依赖
- PDF/TXT 处理逻辑保持不变
- 新增方法：
  - `_load_docx()` / `_load_markdown()`：加载 Word/Markdown 文档
  - `_convert_docx_to_markdown()`：mammoth 转换
  - `_filter_toc_lines()`：过滤目录行
  - `_split_with_headers()`：按标题层级切分
  - `_split_oversized_chunk()`：超长 chunk 二次切分
  - `_contains_table()` / `_extract_table_blocks()`：表格检测与分离
  - `_clean_markdown()` / `_clean_header_text()`：文本清洗
- 测试结果：Word 文档生成 2063 chunks，标题层级分布 h1:2056 / h2:2025 / h3:1991 / h4:9

### v0.4.0 — Phase 4 前端界面
- 新增 `gradio_app.py`：Gradio 交互式 Web 界面（Gradio 6.0 兼容）
- 新增 `streamlit_app.py`：Streamlit 交互式 Web 界面（更美观）
  - Agent/RAG 双模式切换
  - 多轮对话
  - 引用来源展示（文件名、页码、相关度、片段预览）
  - 工具调用记录展示（调用名、参数、结果）
  - 知识库状态面板（文档数、模型、检索配置）
  - 5 个示例农业问题
- 扩展 `main.py`：
  - 新增 `--gradio` 参数：Gradio 前端模式
  - 新增 `--streamlit` 参数：Streamlit 前端模式
  - 新增 `--share` 参数：公网分享（仅 Gradio）
- 更新 `requirements.txt`：添加 `gradio>=6.0.0` 和 `streamlit>=1.30.0` 依赖

### v0.3.0 — Phase 3 Agent 能力完成
- 新增 `core/tools/` 目录：工具模块
  - `base.py`：ToolResult 数据类、BaseTool 抽象基类
  - `weather_tool.py`：天气查询工具（模拟 10 个主要农业城市数据）
  - `agri_calculator.py`：农学计算工具（积温、降水统计、发育期推算）
  - `knowledge_search.py`：知识库检索工具（封装 RAG Pipeline）
- 新增 `core/agent/` 目录：Agent 模块
  - `agent.py`：AgricultureAgent 核心（ReAct 循环、AgentContext 多轮追问）
  - `tool_registry.py`：工具注册中心（生成 OpenAI tool schema）
  - `prompts.py`：Agent 系统提示词模板
- 扩展 `core/llm_client.py`：
  - 新增 `chat_with_tools()` 方法：支持 Qwen Tool Calling
  - 新增 `submit_tool_results()` 方法：提交工具执行结果
- 扩展 `core/config.py`：
  - 新增 AGENT_MAX_ITERATIONS（最大迭代次数，默认 5）
  - 新增 AGENT_ENABLE_TOOLS（是否启用工具）
  - 新增 WEATHER_API_KEY/URL（天气 API 配置，预留）
- 扩展 `api/routes.py`：
  - 新增 `/api/agent/query` 接口：Agent 智能问答
  - 新增 `/api/agent/tools` 接口：获取可用工具列表
  - 新增 AgentQueryRequest/AgentQueryResponse 数据模型
- 扩展 `main.py`：
  - 新增 `--agent` 参数：Agent CLI 模式
  - 新增 `run_agent()` 函数：Agent 命令行交互

### v0.2.1 — 混合检索功能新增
- 新增 `core/bm25_store.py`：BM25 稀疏检索模块（支持本地持久化）
- 新增 `core/hybrid_retriever.py`：混合检索 RRF 融合模块
- 扩展 `core/config.py`：添加 USE_HYBRID、HYBRID_ALPHA 等配置
- 扩展 `core/rag_pipeline.py`：集成混合检索（BM25 + 向量）
- 更新 `ingest.py`：同步建立 BM25 索引
- 更新 `requirements.txt`：添加 rank_bm25 依赖
- 更新 `evaluation/run_comparison.py`：支持 7 组对比实验

### v0.2.0 — Phase 2 检索优化完成
- 新增 `core/reranker.py`：BGE-Reranker-v2 重排序模块（懒加载）
- 扩展 `core/config.py`：添加 HyDE、Reranker 配置参数
- 扩展 `core/vector_store.py`：添加 `search_by_vector()` 方法
- 扩展 `core/rag_pipeline.py`：集成 HyDE 和 Reranker
- 新增 `evaluation/ragas_evaluator.py`：RAGAS 框架评估脚本
- 新增 `evaluation/run_comparison.py`：四组对比实验脚本
- 更新 `requirements.txt`：添加 ragas、langchain 依赖

### v0.1.0 — Phase 1 初始版本
- 实现完整 RAG 基础链路
- 文档处理、BGE-M3 向量化、ChromaDB 存储、Qwen 调用、FastAPI 接口

### v0.1.1 — 修复依赖兼容性问题
- `requirements.txt` 锁定 `FlagEmbedding==1.2.11`、`transformers==4.44.2`
- 原因：FlagEmbedding 新版依赖 `is_torch_fx_available`，该函数在 transformers 4.45+ 被移除

### v0.1.2 — 文本切分策略说明更新
- 明确文本切分使用 LangChain `RecursiveCharacterTextSplitter`（替换原自实现版本）
- 简历描述更新，使用业界通用术语