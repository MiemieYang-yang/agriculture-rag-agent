"""
LLM 调用模块（Qwen3 / 通义千问）
Qwen 系列完全兼容 OpenAI 接口，直接用 openai 库调用

支持：
- 普通同步调用
- 流式输出（streaming）—— Phase 4 工程化时体验更好
- 对话历史管理（多轮对话）
"""
from typing import List, Dict, Optional, Generator


from openai import OpenAI

from core.config import cfg
import logging

logger = logging.getLogger(__name__)

class QwenClient:
    """
    Qwen LLM 客户端

    使用方式:
        client = QwenClient()

        # 单次问答
        reply = client.chat("今天天气怎么样？")

        # 带历史的多轮对话
        history = []
        reply, history = client.chat_with_history("水稻最佳温度？", history)
        reply, history = client.chat_with_history("那玉米呢？", history)  # 记得上轮上下文
    """

    def __init__(self):
        if not cfg.QWEN_API_KEY:
            raise ValueError(
                "未设置 QWEN_API_KEY，请在 .env 文件中配置\n"
                "申请地址：https://dashscope.aliyun.com/（免费额度够用）"
            )
        self.client = OpenAI(
            api_key=cfg.QWEN_API_KEY,
            base_url=cfg.QWEN_BASE_URL,
        )
        self.model = cfg.QWEN_MODEL
        logger.info(f"Qwen 客户端初始化完成，模型: {self.model}")

    # ── 核心调用 ─────────────────────────────────────────────────────────────

    def chat(
            self,
            user_message: str,
            system_prompt: str = cfg.SYSTEM_PROMPT,
            history: Optional[List[Dict]] = None,
            temperature: float = 0.3,  # 农业问答，低温度保证准确性
            max_tokens: int = 2048,
    ) -> str:
        """
        单次调用

        Args:
            user_message: 用户输入
            system_prompt: 系统提示词
            history:       对话历史 [{"role": "user/assistant", "content": "..."}]
            temperature:   生成随机性，0.0~1.0，农业问答建议 0.1~0.3

        Returns:
            模型回复文本
        """
        messages = self._build_messages(user_message, system_prompt, history)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            reply = response.choices[0].message.content
            logger.debug(f"LLM 回复长度: {len(reply)} 字符")
            return reply

        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            raise

    def chat_stream(
            self,
            user_message: str,
            system_prompt: str = cfg.SYSTEM_PROMPT,
            history: Optional[List[Dict]] = None,
            temperature: float = 0.3,
    ) -> Generator[str, None, None]:
        """
        流式调用（Phase 4 工程化时使用）
        边生成边返回，前端实时渲染

        使用方式:
            for chunk in client.chat_stream("问题"):
                print(chunk, end="", flush=True)
        """
        messages = self._build_messages(user_message, system_prompt, history)

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def chat_with_history(
            self,
            user_message: str,
            history: List[Dict],
            system_prompt: str = cfg.SYSTEM_PROMPT,
            **kwargs,
    ) -> tuple[str, List[Dict]]:
        """
        多轮对话接口，自动维护历史

        Returns:
            (reply, updated_history)
        """
        reply = self.chat(user_message, system_prompt, history, **kwargs)

        # 更新历史
        updated_history = history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": reply},
        ]
        return reply, updated_history

    # ── 工具方法 ─────────────────────────────────────────────────────────────

    @staticmethod
    def _build_messages(
            user_message: str,
            system_prompt: str,
            history: Optional[List[Dict]],
    ) -> List[Dict]:
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        return messages