"""
RAGAS 框架评估模块
评估 RAG 系统的检索和生成质量

RAGAS 评估指标：
- Faithfulness: 答案是否忠实于检索内容（是否产生幻觉）
- Answer Relevancy: 答案与问题的相关性
- Context Precision: 检索内容的精确度
- Context Recall: 检索内容的召回率

使用方式:
    evaluator = RAGASEvaluator()
    results = evaluator.evaluate(pipeline)
    print(results)
"""
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
from typing import List, Dict, Optional
from datasets import Dataset
import logging
from core.config import cfg

logger = logging.getLogger(__name__)


class RAGASEvaluator:
    """
    RAGAS 评估器

    使用 RAGAS 框架评估 RAG Pipeline 的性能
    """

    def __init__(
            self,
            test_data: Optional[List[Dict]] = None,
    ):
        # 加载测试数据集
        if test_data is None:
            from evaluation.eval_dataset import EVAL_DATASET
            test_data = EVAL_DATASET

        self.test_data = test_data
        logger.info(f"加载评估数据集: {len(self.test_data)} 条")

    def generate_rag_responses(
            self,
            pipeline,
    ) -> Dataset:
        """
        用 RAG Pipeline 生成答案和检索上下文

        RAGAS 0.4.x 需要的字段:
        - user_input: str (问题)
        - response: str (RAG生成的答案)
        - retrieved_contexts: List[str] (检索到的文档内容列表)
        - reference: str (标准答案)

        Args:
            pipeline: RAGPipeline 实例

        Returns:
            HuggingFace Dataset，可直接用于 RAGAS evaluate()
        """
        logger.info(f"开始生成 RAG 回答，共 {len(self.test_data)} 条问题...")

        results = []
        for i, item in enumerate(self.test_data):
            question = item["question"]
            reference = item["ground_truth"]

            logger.info(f"处理问题 {i + 1}/{len(self.test_data)}: {question[:50]}...")

            # 调用 RAG Pipeline 检索
            retrieved = pipeline._retrieve(question)

            # 提取检索上下文
            retrieved_contexts = [doc["content"] for doc in retrieved]

            # 如果有检索结果，生成答案
            if retrieved_contexts:
                prompt = pipeline._build_prompt(question, retrieved)
                response = pipeline.llm_client.chat(prompt)
            else:
                # 无检索结果时的 fallback
                response = pipeline.llm_client.chat(
                    question,
                    system_prompt=cfg.SYSTEM_PROMPT or "请基于你的专业知识回答。",
                )

            results.append({
                "user_input": question,
                "response": response,
                "retrieved_contexts": retrieved_contexts,
                "reference": reference,
            })

        logger.info(f"RAG 回答生成完成")
        return Dataset.from_list(results)

    def evaluate(
            self,
            pipeline,
            metrics: Optional[List[str]] = None,
    ) -> Dict:
        """
        执行 RAGAS 评估

        Args:
            pipeline: RAG Pipeline 实例
            metrics: 要评估的指标列表，默认全部

        Returns:
            各指标的分数字典，如:
            {
                "faithfulness": 0.85,
                "answer_relevancy": 0.78,
                "context_precision": 0.72,
                "context_recall": 0.68,
            }
        """
        # 导入 RAGAS (适配 0.4.x 版本)
        try:
            from ragas import evaluate
            from ragas.metrics._faithfulness import Faithfulness
            from ragas.metrics._answer_relevance import AnswerRelevancy
            from ragas.metrics._context_precision import ContextPrecision
            from ragas.metrics._context_recall import ContextRecall
        except ImportError:
            logger.error("请先安装 ragas: pip install ragas")
            raise

        # 默认评估所有指标
        if metrics is None:
            metrics_to_use = [
                Faithfulness(),
                AnswerRelevancy(),
                ContextPrecision(),
                ContextRecall(),
            ]
        else:
            # 根据名称选择指标
            metrics_map = {
                "faithfulness": Faithfulness(),
                "answer_relevancy": AnswerRelevancy(),
                "context_precision": ContextPrecision(),
                "context_recall": ContextRecall(),
            }
            metrics_to_use = [metrics_map[m] for m in metrics if m in metrics_map]

        # 准备数据集
        dataset = self.generate_rag_responses(pipeline)

        # 配置 RAGAS 使用的 LLM（适配 Qwen）
        llm = self._get_ragas_llm()
        embeddings = self._get_ragas_embeddings()

        logger.info("开始 RAGAS 评估...")

        # 执行评估
        try:
            # 配置超时和重试（适配网络较慢的情况）
            from ragas.run_config import RunConfig
            run_config = RunConfig(
                timeout=300,      # 5分钟超时
                max_retries=5,    # 最多重试5次
                max_wait=60,      # 重试等待时间
            )

            # RAGAS 0.4.x 版本的调用方式
            result = evaluate(
                dataset,
                metrics=metrics_to_use,
                llm=llm,
                embeddings=embeddings,
                run_config=run_config,
            )

            # 提取平均分数 (使用 _repr_dict)
            scores = {}
            if hasattr(result, '_repr_dict'):
                for metric_name, score in result._repr_dict.items():
                    scores[metric_name] = float(score)

            logger.info("RAGAS 评估完成")
            for metric_name, score in scores.items():
                logger.info(f"  {metric_name}: {score:.4f}")

            return scores

        except Exception as e:
            logger.error(f"RAGAS 评估失败: {e}")
            raise

    def _get_ragas_llm(self):
        """配置 RAGAS 使用的 LLM（适配 Qwen）"""
        from langchain_openai import ChatOpenAI
        from core.config import cfg

        return ChatOpenAI(
            model=cfg.QWEN_MODEL,
            openai_api_key=cfg.QWEN_API_KEY,
            openai_api_base=cfg.QWEN_BASE_URL,
            temperature=0.3,
            request_timeout=120,  # 单次请求超时 2分钟
        )

    def _get_ragas_embeddings(self):
        """配置 RAGAS 使用的 Embeddings"""
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        from core.config import cfg

        # 使用本地 BGE-M3 模型路径
        model_path = os.path.join(Path(__file__).parent.parent, "bge-m3")
        if not os.path.exists(model_path):
            logger.warning(f"本地模型路径不存在: {model_path}，将尝试从网络下载")
            model_path = cfg.BGE_MODEL_NAME

        return HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={"trust_remote_code": True},
        )

    def quick_eval(
            self,
            pipeline,
            sample_size: int = 5,
    ) -> Dict:
        """
        快速评估（使用小样本）

        用于快速验证 Pipeline 是否正常工作

        Args:
            pipeline: RAG Pipeline 实例
            sample_size: 样本数量，默认 5 条

        Returns:
            评估结果
        """
        # 取前 sample_size 条
        small_test = self.test_data[:sample_size]
        logger.info(f"快速评估: 使用 {len(small_test)} 条样本")

        # 创建临时评估器
        temp_evaluator = RAGASEvaluator(test_data=small_test)
        return temp_evaluator.evaluate(pipeline)


if __name__ == "__main__":
    # 切换到项目根目录（确保相对路径正确）
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    # 测试评估器
    from core.rag_pipeline import RAGPipeline

    # 创建 Baseline Pipeline
    pipeline = RAGPipeline(use_hyde=False, use_reranker=False)

    # 快速评估
    evaluator = RAGASEvaluator()
    results = evaluator.quick_eval(pipeline, sample_size=3)

    print("\n评估结果:")
    for metric, score in results.items():
        print(f"  {metric}: {score:.4f}")