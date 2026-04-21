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
import asyncio
from typing import List, Dict, Optional
from datasets import Dataset
import logging

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

        RAGAS 需要的字段:
        - question: str (问题)
        - answer: str (RAG生成的答案)
        - contexts: List[str] (检索到的文档内容列表)
        - ground_truth: str (标准答案)

        Args:
            pipeline: RAGPipeline 实例

        Returns:
            HuggingFace Dataset，可直接用于 RAGAS evaluate()
        """
        logger.info(f"开始生成 RAG 回答，共 {len(self.test_data)} 条问题...")

        results = []
        for i, item in enumerate(self.test_data):
            question = item["question"]
            ground_truth = item["ground_truth"]

            logger.info(f"处理问题 {i + 1}/{len(self.test_data)}: {question[:50]}...")

            # 调用 RAG Pipeline 检索
            retrieved = pipeline._retrieve(question)

            # 提取检索上下文
            contexts = [doc["content"] for doc in retrieved]

            # 如果有检索结果，生成答案
            if contexts:
                prompt = pipeline._build_prompt(question, retrieved)
                answer = pipeline.llm_client.chat(prompt)
            else:
                # 无检索结果时的 fallback
                answer = pipeline.llm_client.chat(
                    question,
                    system_prompt=pipeline.llm_client.system_prompt or "请基于你的专业知识回答。",
                )

            results.append({
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth,
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
        # 导入 RAGAS
        try:
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            )
        except ImportError:
            logger.error("请先安装 ragas: pip install ragas")
            raise

        # 默认评估所有指标
        if metrics is None:
            metrics_to_use = [
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ]
        else:
            # 根据名称选择指标
            metrics_map = {
                "faithfulness": faithfulness,
                "answer_relevancy": answer_relevancy,
                "context_precision": context_precision,
                "context_recall": context_recall,
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
            # RAGAS 0.1.x 版本的调用方式
            result = evaluate(
                dataset,
                metrics=metrics_to_use,
                llm=llm,
                embeddings=embeddings,
            )

            # 提取分数
            scores = {}
            for metric in metrics_to_use:
                metric_name = metric.name
                if metric_name in result:
                    scores[metric_name] = float(result[metric_name])

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
        )

    def _get_ragas_embeddings(self):
        """配置 RAGAS 使用的 Embeddings"""
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from core.config import cfg

        # 使用 BGE-M3 模型
        return HuggingFaceEmbeddings(
            model_name=cfg.BGE_MODEL_NAME,
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