# 农业气候与资源数据专家助手

基于 RAG 架构的农业领域垂直问答系统。

## 技术栈

| 组件 | 选型 | 说明 |
|------|------|------|
| 向量模型 | BGE-M3 | 多语言，中文效果优秀 |
| 向量数据库 | ChromaDB | 本地持久化，无需单独部署 |
| LLM | Qwen（通义千问） | 兼容 OpenAI 接口 |
| API 框架 | FastAPI | 自带 Swagger 文档 |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入 QWEN_API_KEY
```

### 3. 准备数据

把农业相关文档（PDF/TXT/MD）放入 `data/raw/` 目录。
已提供示例文档 `crop_climate_guide.md` 可直接使用。

### 4. 构建知识库

```bash
python ingest.py
```

### 5. 开始使用

```bash
# 命令行交互模式
python main.py

# 命令行流式交互模式
python main.py--stream

# API 服务模式
python main.py --serve
# 访问 http://localhost:8000/docs 查看接口文档
```

```bash
python main.py --gradio
```

## 项目结构

```
agri_rag/
├── core/
│   ├── config.py            # 统一配置
│   ├── document_processor.py # 文档加载与切分
│   ├── embedder.py          # BGE-M3 向量化
│   ├── vector_store.py      # ChromaDB 管理
│   ├── llm_client.py        # Qwen 调用
│   └── rag_pipeline.py      # 核心 RAG 链路 ← 最重要
├── api/
│   └── routes.py            # FastAPI 接口
├── data/
│   └── raw/                 # 放原始文档
├── vectorstore/             # ChromaDB 持久化数据
├── ingest.py                # 入库脚本
└── main.py                  # 启动入口
```

## RAG 流程

```
用户问题
    ↓
BGE-M3 编码 Query → 向量
    ↓
ChromaDB 语义检索 → Top-K 相关文档
    ↓
相似度过滤（< 0.4 丢弃）
    ↓
构建 RAG Prompt（上下文 + 问题）
    ↓
Qwen LLM 生成回答
    ↓
返回 答案 + 来源引用
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /api/query | 单次 RAG 问答 |
| POST | /api/chat  | 多轮对话 |
| GET  | /api/stats | 知识库统计 |
| GET  | /health    | 健康检查 |

### 示例请求

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "四川盆地水稻最佳播种期是什么时候？"}'
```