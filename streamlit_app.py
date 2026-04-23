"""
Streamlit 前端界面
农业 RAG Agent 智能问答系统

启动方式：streamlit run streamlit_app.py
"""
import streamlit as st
from typing import List, Dict, Optional
import logging

from core.rag_pipeline import RAGPipeline
from core.agent.agent import AgricultureAgent, AgentContext

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@st.cache_resource
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


@st.cache_resource
def get_agent() -> AgricultureAgent:
    return AgricultureAgent(rag_pipeline=get_pipeline())


# ── 页面配置 ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="农业智能问答助手",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 自定义样式 ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* 整体 */
    .stApp { background: #fafbfc; }

    /* 隐藏 Streamlit 默认头部和底部 */
    header { visibility: hidden; }
    #stDecoration { display: none; }

    /* 聊天气泡 */
    .stChatMessage {
        background: transparent !important;
        border: none !important;
        padding: 0.5rem 0 !important;
    }

    /* 用户消息 */
    [data-testid="stChatMessageContent"] {
        border-radius: 16px;
        padding: 12px 16px;
    }

    /* 侧边栏 */
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #eef0f2;
    }

    /* 引用卡片 */
    .source-card {
        background: #f7f8fa;
        border-radius: 10px;
        padding: 10px 14px;
        margin: 6px 0;
        border-left: 3px solid #667eea;
        font-size: 13px;
    }

    /* 工具卡片 */
    .tool-card {
        background: #eef2ff;
        border-radius: 10px;
        padding: 10px 14px;
        margin: 6px 0;
        border-left: 3px solid #764ba2;
        font-size: 13px;
    }

    /* 示例按钮 */
    .example-btn {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 20px;
        padding: 8px 16px;
        font-size: 13px;
        cursor: pointer;
        transition: all 0.2s;
    }
    .example-btn:hover {
        background: #f0f0f0;
        border-color: #667eea;
    }
</style>
""", unsafe_allow_html=True)


# ── 初始化 Session State ─────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "mode" not in st.session_state:
    st.session_state.mode = "Agent"

if "sources" not in st.session_state:
    st.session_state.sources = []

if "tool_calls" not in st.session_state:
    st.session_state.tool_calls = []

if "last_sources" not in st.session_state:
    st.session_state.last_sources = ""

if "last_tools" not in st.session_state:
    st.session_state.last_tools = ""


# ── 侧边栏 ──────────────────────────────────────────────────────────────────

with st.sidebar:
    # Logo 和标题
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 10px;">
        <div style="font-size:48px;">🌾</div>
        <h2 style="margin:8px 0 4px; color:#1a1a1a;">农业智能问答助手</h2>
        <p style="color:#888; font-size:13px;">RAG + Agent 技术</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # 模式切换
    st.session_state.mode = st.radio(
        "问答模式",
        options=["Agent", "RAG"],
        index=0,
        horizontal=True,
        help="Agent 模式支持工具调用，RAG 模式仅使用知识库检索",
    )

    st.divider()

    # 系统状态
    st.markdown("#### 📊 系统状态")
    try:
        pipeline = get_pipeline()
        doc_count = pipeline.vector_store.count()
        st.markdown(f"- 🟢 服务运行中\n- 📄 已加载 **{doc_count}** 篇文档")
    except:
        st.markdown("- 🟡 初始化中...")

    st.divider()

    # 支持工具
    st.markdown("#### 🔧 支持工具")
    st.markdown("""
    - 🌤 天气查询
    - 🌡 积温计算
    - 📚 知识库检索
    """)

    st.divider()

    # 引用来源
    st.markdown("#### 📄 参考来源")
    st.markdown(st.session_state.last_sources, unsafe_allow_html=True)

    st.divider()

    # 工具调用
    st.markdown("#### 🔧 工具调用")
    st.markdown(st.session_state.last_tools, unsafe_allow_html=True)

    st.divider()

    # 清空对话
    if st.button("🗑 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_sources = ""
        st.session_state.last_tools = ""
        st.rerun()


# ── 主区域 ───────────────────────────────────────────────────────────────────

# 欢迎语（无消息时显示）
if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center; padding: 80px 20px;">
        <div style="font-size:64px;">🌾</div>
        <h1 style="color:#1a1a1a; margin:16px 0 8px;">农业智能问答助手</h1>
        <p style="color:#888; font-size:16px; max-width:500px; margin:0 auto 30px;">
            我可以帮您解答农业科技、天气查询、积温计算等问题，请输入您的问题开始对话。
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 示例问题
    example_questions = [
        ("🌱", "四川盆地水稻最佳播种期是什么时候？"),
        ("🌤", "北京今天的天气如何？适合农事活动吗？"),
        ("🌡", "计算哈尔滨地区水稻积温需求"),
        ("🌽", "东北春玉米的适宜播种期是什么？"),
        ("🌧", "成都地区年降水量是多少？"),
    ]

    cols = st.columns(len(example_questions))
    for i, (icon, q) in enumerate(example_questions):
        with cols[i]:
            if st.button(f"{icon} {q[:6]}...", key=f"ex_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": q})
                st.rerun()


# ── 渲染对话历史 ─────────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ── 处理用户输入 ─────────────────────────────────────────────────────────────

if prompt := st.chat_input("输入您的问题..."):
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # 生成回答
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                # 构建历史
                chat_history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                ]

                if st.session_state.mode == "Agent":
                    agent = get_agent()
                    result = agent.process(prompt, history=chat_history)
                    answer = result.answer

                    # 格式化引用来源
                    if result.sources:
                        src_parts = []
                        for i, src in enumerate(result.sources[:5], 1):
                            filename = src.get("filename", "未知")
                            page = src.get("page", "")
                            page_str = f" P.{page}" if page else ""
                            score = src.get("score", 0)
                            snippet = src.get("snippet", "")[:80]
                            src_parts.append(
                                f'<div class="source-card">'
                                f'<b>{i}. {filename}{page_str}</b> <code>相关度 {score:.2f}</code><br>'
                                f'{snippet}...</div>'
                            )
                        st.session_state.last_sources = "".join(src_parts)
                    else:
                        st.session_state.last_sources = "暂无引用来源"

                    # 格式化工具调用
                    if result.tool_calls:
                        tc_parts = []
                        for tc in result.tool_calls:
                            icon = "✓" if tc.success else "✗"
                            name = tc.name
                            args_str = ""
                            if tc.arguments:
                                args_str = " | ".join(f"{k}={v}" for k, v in tc.arguments.items())
                            tc_parts.append(
                                f'<div class="tool-card">'
                                f'{icon} <b>{name}</b>'
                                f'{"<br>" + args_str if args_str else ""}</div>'
                            )
                        st.session_state.last_tools = "".join(tc_parts)
                    else:
                        st.session_state.last_tools = "暂无工具调用"

                else:
                    pipeline = get_pipeline()
                    rag_result = pipeline.query(prompt, history=chat_history)
                    answer = rag_result.get("answer", "")

                    if rag_result.get("sources"):
                        src_parts = []
                        for i, src in enumerate(rag_result["sources"][:5], 1):
                            filename = src.get("filename", "未知")
                            page = src.get("page", "")
                            page_str = f" P.{page}" if page else ""
                            score = src.get("score", 0)
                            snippet = src.get("snippet", "")[:80]
                            src_parts.append(
                                f'<div class="source-card">'
                                f'<b>{i}. {filename}{page_str}</b> <code>相关度 {score:.2f}</code><br>'
                                f'{snippet}...</div>'
                            )
                        st.session_state.last_sources = "".join(src_parts)
                    else:
                        st.session_state.last_sources = "暂无引用来源"
                    st.session_state.last_tools = "暂无工具调用"

                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                error_msg = f"处理出错: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
