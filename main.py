"""
项目启动入口

两种运行模式：
1. 命令行交互模式（开发调试用）：python main.py
2. API 服务模式（演示/集成用）：  python main.py --serve

Phase 3 新增：
3. Agent 模式（工具调用）：      python main.py --agent

Phase 4 新增：
4. Gradio 前端模式：             python main.py --gradio
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import logging

logger = logging.getLogger(__name__)


def run_cli(stream: bool = False):
    """命令行交互模式：直接在终端问答
    
    Args:
        stream: 是否启用流式输出
    """
    from core.rag_pipeline import RAGPipeline

    logger.info("初始化 RAG Pipeline...")
    pipeline = RAGPipeline()

    if pipeline.vector_store.count() == 0:
        logger.warning("知识库为空！请先运行: python ingest.py")
        logger.info("也可以继续测试，将使用模型内置知识回答")

    print("\n" + "=" * 60)
    print("  农业气候与资源数据专家助手")
    if stream:
        print("  模式: 流式输出 (边生成边显示)")
    else:
        print("  模式: 普通输出 (等待完整回答)")
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
            if stream:
                # 流式输出模式
                print("\n助手: ", end="", flush=True)
                answer_chunks = []
                for token in pipeline.query_stream(question, history=history):
                    print(token, end="", flush=True)
                    answer_chunks.append(token)
                print()  # 换行
                
                answer = "".join(answer_chunks)
                result = {
                    "answer": answer,
                    "sources": [],  # 流式模式下不显示来源（可以后续优化）
                    "retrieved_count": 0,
                }
            else:
                # 普通模式
                result = pipeline.query(question, history=history)
                print(f"\n助手: {result['answer']}")

            # 显示引用来源（仅普通模式）
            if not stream and result["sources"]:
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


def run_agent(stream: bool = False):
    """Agent 命令行交互模式：支持工具调用的智能问答

    Args:
        stream: 是否启用流式输出
    """
    from core.agent.agent import AgricultureAgent, AgentContext

    logger.info("初始化 Agriculture Agent...")
    agent = AgricultureAgent()
    context = AgentContext()

    print("\n" + "=" * 60)
    print("  农业气候与资源数据专家助手（Agent 模式）")
    print("  支持：天气查询、农学计算、知识库检索")
    if stream:
        print("  模式: 流式输出")
    else:
        print("  模式: 普通输出")
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
            context = AgentContext()
            print("对话历史和上下文已清除\n")
            continue

        try:
            # Agent 处理
            result = agent.process(question, history=history if history else None, context=context)

            # 显示回答
            print(f"\n助手: {result.answer}")

            # 显示工具调用信息
            if result.tool_calls:
                print(f"\n  🔧 工具调用（共 {len(result.tool_calls)} 次）:")
                for tc in result.tool_calls:
                    status = "✓" if tc.success else "✗"
                    print(f"     {status} {tc.name}")
                    if not tc.success:
                        print(f"       错误: {tc.result.get('error', '未知错误')}")

            # 显示引用来源
            if result.sources:
                print(f"\n  📄 引用来源（共 {len(result.sources)} 条）:")
                for src in result.sources[:3]:  # 只显示前3条
                    page_info = f" p.{src.get('page')}" if src.get('page') else ""
                    print(f"     · {src.get('filename')}{page_info}")
                    print(f"       {src.get('snippet', '')[:80]}...")

            # 显示迭代次数
            if result.iterations > 1:
                print(f"\n  🔄 Agent 迭代次数: {result.iterations}")

            print()

            # 维护对话历史
            history.extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": result.answer},
            ])

            # 更新上下文
            context = result.context or context

            # 控制历史长度
            if len(history) > 20:
                history = history[-20:]

        except Exception as e:
            logger.error(f"Agent 处理失败: {e}")
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


def run_gradio(host: str = "0.0.0.0", port: int = 7860, share: bool = False):
    """Gradio 前端模式：启动 Web 界面"""
    from gradio_app import app

    logger.info(f"Gradio 服务启动: http://{host}:{port}")
    if share:
        logger.info("公网分享已启用，将生成临时公网链接")
    app.launch(
        server_name=host,
        server_port=port,
        share=share,
    )


def run_streamlit(host: str = "0.0.0.0", port: int = 8501):
    """Streamlit 前端模式：启动 Web 界面"""
    import subprocess
    import sys

    logger.info(f"Streamlit 服务启动: http://{host}:{port}")
    cmd = [
        sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
        "--server.address", host,
        "--server.port", str(port),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]
    subprocess.run(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="农业 RAG 专家助手")
    parser.add_argument("--serve", action="store_true", help="启动 API 服务模式")
    parser.add_argument("--agent", action="store_true", help="启动 Agent 模式（支持工具调用）")
    parser.add_argument("--gradio", action="store_true", help="启动 Gradio 前端模式")
    parser.add_argument("--streamlit", action="store_true", help="启动 Streamlit 前端模式")
    parser.add_argument("--host", default="0.0.0.0", help="服务监听地址")
    parser.add_argument("--port", type=int, default=8000, help="API 端口（Gradio 默认 7860，Streamlit 默认 8501）")
    parser.add_argument("--share", action="store_true", help="Gradio 模式下启用公网分享")
    parser.add_argument("--stream", action="store_true", help="启用流式输出（CLI模式）")
    args = parser.parse_args()

    if args.serve:
        run_server(args.host, args.port)
    elif args.agent:
        run_agent(stream=args.stream)
    elif args.gradio:
        port = args.port if args.port != 8000 else 7860
        run_gradio(args.host, port, args.share)
    elif args.streamlit:
        port = args.port if args.port != 8000 else 8501
        run_streamlit(args.host, port)
    else:
        run_cli(stream=args.stream)