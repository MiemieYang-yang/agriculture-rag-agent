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
| 文本切分 | LangChain RecursiveCharacterTextSplitter | 语义感知切分，简历友好 |
| LLM | Qwen（通义千问） | 兼容 OpenAI 接口 |
| API 框架 | FastAPI | 自带 Swagger 文档 |
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
├── main.py                     # 启动入口（CLI / API 双模式）
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

### Phase 3 · Agent 能力 ⬜ 待开始
**目标：** 引入工具调用，展示 Agent 决策能力

**计划任务：**
- [ ] 封装工具：天气 API 查询、农学指标计算（积温/降水）、本地数据库查询
- [ ] 实现 ReAct 循环：意图识别 → 工具路由 → 执行 → 整合 → 回复
- [ ] 工具调用异常处理：重试、格式校验、防循环
- [ ] 多轮追问支持（"那玉米呢？""那明天呢？"）

**预期简历写法：**
> 在 RAG 基础上引入 Agent 能力，封装天气查询、农学指标计算等工具，基于 ReAct 框架实现意图识别 → 工具路由 → 结果整合的自主决策流程，支持多轮追问，处理工具调用异常与格式校验。

---

### Phase 4 · 工程化 ⬜ 待开始
**目标：** 体现生产交付意识，锦上添花

**计划任务：**
- [ ] 流式输出（Streaming）：边生成边返回
- [ ] LangSmith 链路追踪与日志
- [ ] 增量知识库更新（新文档自动入库，不重建索引）
- [ ] Docker 容器化部署

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

---

## 七、变更日志

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