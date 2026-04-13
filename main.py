"""
项目启动入口

两种运行模式：
1. 命令行交互模式（开发调试用）：python main.py
2. API 服务模式（演示/集成用）：  python main.py --serve
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import logging

logger = logging.getLogger(__name__)


def run_cli():
    """命令行交互模式：直接在终端问答"""
    from core.rag_pipeline import RAGPipeline

    logger.info("初始化 RAG Pipeline...")
    pipeline = RAGPipeline()

    if pipeline.vector_store.count() == 0:
        logger.warning("知识库为空！请先运行: python ingest.py")
        logger.info("也可以继续测试，将使用模型内置知识回答")

    print("\n" + "=" * 60)
    print("  农业气候与资源数据专家助手")
    print("  输入问题开始对话，输入 'quit' 退出，输入 'clear' 清除历史")
    print("=" * 60 + "\n")

    history = []

    while True:
        try:
            question = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "退出"}:
            print("再见！")
            break
        if question.lower() in {"clear", "清除历史"}:
            history = []
            print("对话历史已清除\n")
            continue

        try:
            result = pipeline.query(question, history=history)

            print(f"\n助手: {result['answer']}")

            # 显示引用来源
            if result["sources"]:
                print(f"\n  📄 引用来源（共 {result['retrieved_count']} 条检索结果）:")
                for src in result["sources"]:
                    page_info = f" p.{src['page']}" if src.get('page') else ""
                    print(f"     · {src['filename']}{page_info}（相关度: {src['score']:.2f}）")
                    print(f"       {src['snippet']}")
            print()

            # 维护多轮对话历史
            history.extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": result["answer"]},
            ])

            # 控制历史长度，避免 token 超限（保留最近 10 轮）
            if len(history) > 20:
                history = history[-20:]

        except Exception as e:
            logger.error(f"处理失败: {e}")
            print(f"出错了: {e}\n")


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """API 服务模式：启动 FastAPI"""
    import uvicorn
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from api.routes import router

    app = FastAPI(
        title="农业气候与资源数据专家助手 API",
        description="基于 RAG 架构的农业领域智能问答系统",
        version="1.0.0",
    )

    # 允许跨域（方便本地前端调试）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api")

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "agri-rag"}

    logger.info(f"API 服务启动: http://{host}:{port}")
    logger.info(f"接口文档:     http://{host}:{port}/docs")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="农业 RAG 专家助手")
    parser.add_argument("--serve", action="store_true", help="启动 API 服务模式")
    parser.add_argument("--host", default="0.0.0.0", help="API 监听地址")
    parser.add_argument("--port", type=int, default=8000, help="API 端口")
    args = parser.parse_args()

    if args.serve:
        run_server(args.host, args.port)
    else:
        run_cli()