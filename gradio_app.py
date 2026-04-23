"""
Gradio 前端界面
农业 RAG Agent 智能问答系统

功能：
- Agent 智能问答（支持工具调用）
- RAG 基础问答
- 显示工具调用记录
- 显示引用来源
- 知识库统计
"""
import gradio as gr
from typing import List, Dict, Optional
import logging

from core.rag_pipeline import RAGPipeline
from core.agent.agent import AgricultureAgent, AgentContext

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局单例
_pipeline: Optional[RAGPipeline] = None
_agent: Optional[AgricultureAgent] = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


def get_agent() -> AgricultureAgent:
    global _agent
    if _agent is None:
        _agent = AgricultureAgent(rag_pipeline=get_pipeline())
    return _agent


def format_sources(sources: List[Dict]) -> str:
    """格式化引用来源"""
    if not sources:
        return "暂无引用来源"

    parts = []
    for i, src in enumerate(sources, 1):
        filename = src.get("filename", "未知")
        page = src.get("page", "")
        page_str = f" 第{page}页" if page else ""
        score = src.get("score", 0)
        snippet = src.get("snippet", "")

        parts.append(f"**[{i}] {filename}{page_str}** (相关度: {score:.2f})\n> {snippet}")

    return "\n\n".join(parts)


def format_tool_calls(tool_calls: List) -> str:
    """格式化工具调用记录"""
    if not tool_calls:
        return ""

    parts = ["### 工具调用记录\n"]
    for tc in tool_calls:
        name = tc.name
        args = tc.arguments
        result = tc.result
        success = "" if tc.success else ""

        parts.append(f"**{success} {name}**")
        if args:
            args_str = ", ".join(f"{k}={v}" for k, v in args.items())
            parts.append(f"- 参数: {args_str}")
        if result:
            data = result.get("data", result)
            if isinstance(data, dict):
                parts.append(f"- 结果: {data.get('summary', str(data)[:200])}")
            else:
                parts.append(f"- 结果: {str(data)[:200]}")
        parts.append("")

    return "\n".join(parts)


def chat_with_agent(
    message: str,
    history: List,
    mode: str,
):
    """
    聊天处理函数

    Args:
        message: 用户消息
        history: Gradio 聊天历史
        mode: 模式选择 (agent/rag)

    Yields:
        (history, sources_html, tools_html, stats_text)
    """
    if not message.strip():
        yield history, "", "", ""
        return

    # 转换历史格式
    chat_history = []
    for msg in history:
        chat_history.append({"role": "user", "content": msg["content"]})
        # 助手回复会在下一轮添加

    # 添加用户消息到历史
    history = history + [{"role": "user", "content": message}]

    try:
        if mode == "Agent 智能问答":
            # Agent 模式
            agent = get_agent()
            result = agent.process(message, history=chat_history)

            answer = result.answer
            sources_str = format_sources(result.sources)
            tools_str = format_tool_calls(result.tool_calls)

        else:
            # RAG 模式
            pipeline = get_pipeline()
            rag_result = pipeline.query(message, history=chat_history)

            answer = rag_result.get("answer", "")
            sources_str = format_sources(rag_result.get("sources", []))
            tools_str = ""

        # 流式输出效果
        full_response = ""
        for char in answer:
            full_response += char
            # 更新历史中的助手回复
            display_history = history + [{"role": "assistant", "content": full_response}]
            yield display_history, sources_str, tools_str, ""

    except Exception as e:
        logger.error(f"处理失败: {e}")
        error_msg = f"处理失败: {str(e)}"
        history = history + [{"role": "assistant", "content": error_msg}]
        yield history, "", "", ""


def get_stats() -> str:
    """获取知识库统计"""
    try:
        pipeline = get_pipeline()
        doc_count = pipeline.vector_store.count()
        model = pipeline.llm_client.model

        return f"""
**知识库状态**
- 文档数量: {doc_count}
- LLM 模型: {model}
- 混合检索: {'启用' if pipeline.use_hybrid else '关闭'}
- Reranker: {'启用' if pipeline.use_reranker else '关闭'}
"""
    except Exception as e:
        return f"获取统计失败: {e}"


def get_tools_info() -> str:
    """获取可用工具列表"""
    try:
        agent = get_agent()
        tools = agent.tool_registry.get_all_tools()

        parts = ["### 可用工具\n"]
        for tool in tools:
            parts.append(f"- **{tool.name}**: {tool.description}")

        return "\n".join(parts)
    except Exception as e:
        return f"获取工具列表失败: {e}"


# 创建 Gradio 界面
with gr.Blocks(
    title="农业 RAG Agent 智能问答"
) as app:
    gr.Markdown("""
    # 农业气候与资源数据智能问答系统

    基于 RAG + Agent 技术，支持：
    - 农业知识库智能检索问答
    - 天气查询（10个主要农业城市）
    - 农学计算（积温、降水统计、发育期推算）
    """)

    with gr.Row():
        with gr.Column(scale=3):
            # 模式选择
            mode = gr.Radio(
                choices=["Agent 智能问答", "RAG 基础问答"],
                value="Agent 智能问答",
                label="问答模式",
                info="Agent 模式支持工具调用，RAG 模式仅使用知识库检索"
            )

            # 聊天界面
            chatbot = gr.Chatbot(
                label="对话",
                height=450,
                show_copy_button=True,
            )

            # 输入框
            with gr.Row():
                msg = gr.Textbox(
                    label="输入问题",
                    placeholder="例如：四川盆地水稻最佳播种期是什么时候？",
                    scale=4,
                    submit_btn=True,
                )
                clear = gr.ClearButton([msg, chatbot], value="清空对话")

        with gr.Column(scale=1):
            # 右侧信息面板
            gr.Markdown("### 知识库状态")
            stats_box = gr.Markdown(get_stats())
            refresh_stats = gr.Button("刷新状态")
            refresh_stats.click(lambda: get_stats(), outputs=stats_box)

            gr.Markdown("---")
            gr.Markdown(get_tools_info())

    # 来源和工具调用展示
    with gr.Row():
        with gr.Column():
            sources_box = gr.Markdown(
                label="引用来源",
                elem_classes=["sources-box"]
            )
        with gr.Column():
            tools_box = gr.Markdown(
                label="工具调用",
                elem_classes=["tools-box"]
            )

    # 示例问题
    gr.Markdown("### 示例问题")
    gr.Examples(
        examples=[
            "四川盆地水稻最佳播种期是什么时候？",
            "北京今天的天气如何？适合农事活动吗？",
            "计算哈尔滨地区水稻从播种到成熟所需的积温",
            "东北春玉米的适宜播种期和收获期是什么时候？",
            "成都地区的年降水量是多少？",
        ],
        inputs=msg,
    )

    # 处理消息提交
    msg.submit(
        fn=chat_with_agent,
        inputs=[msg, chatbot, mode],
        outputs=[chatbot, sources_box, tools_box, stats_box],
    ).then(lambda: "", outputs=msg)  # 清空输入框


if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
        css="""
        .sources-box { background: #f8f9fa; padding: 10px; border-radius: 8px; }
        .tools-box { background: #e3f2fd; padding: 10px; border-radius: 8px; }
        """
    )
