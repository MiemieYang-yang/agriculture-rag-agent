"""
四组对比实验脚本
对比不同检索策略的效果

实验配置：
1. Baseline（无优化）
2. +HyDE
3. +Reranker
4. +HyDE+Reranker

运行方式:
    python evaluation/run_comparison.py

输出:
    evaluation/results/comparison_results.json
    evaluation/results/comparison_report.md
"""
import json
import time
import os
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import logging

# 添加项目根目录到路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.rag_pipeline import RAGPipeline
from evaluation.ragas_evaluator import RAGASEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# 实验配置（包含混合检索）
EXPERIMENT_CONFIGS = [
    {"name": "Baseline", "use_hyde": False, "use_reranker": False, "use_hybrid": False},
    {"name": "+Hybrid", "use_hyde": False, "use_reranker": False, "use_hybrid": True},
    {"name": "+HyDE", "use_hyde": True, "use_reranker": False, "use_hybrid": False},
    {"name": "+Reranker", "use_hyde": False, "use_reranker": True, "use_hybrid": False},
    {"name": "+Hybrid+Reranker", "use_hyde": False, "use_reranker": True, "use_hybrid": True},
    {"name": "+HyDE+Reranker", "use_hyde": True, "use_reranker": True, "use_hybrid": False},
    {"name": "+Hybrid+HyDE+Reranker", "use_hyde": True, "use_reranker": True, "use_hybrid": True},
]


def run_single_experiment(
        config: Dict,
        evaluator: RAGASEvaluator,
        top_k: int = 5,
        rerank_top_k: int = 50,
        vector_top_k: int = 50,
        bm25_top_k: int = 50,
) -> Dict:
    """
    运行单个实验

    Args:
        config: 实验配置 {"name": str, "use_hyde": bool, "use_reranker": bool, "use_hybrid": bool}
        evaluator: RAGAS 评估器
        top_k: 最终返回的文档数
        rerank_top_k: 粗排时检索的文档数
        vector_top_k: 向量检索数量
        bm25_top_k: BM25 检索数量

    Returns:
        实验结果字典
    """
    logger.info(f"\n{'=' * 50}")
    logger.info(f"开始实验: {config['name']}")
    logger.info(f"配置: hyde={config.get('use_hyde', False)}, reranker={config.get('use_reranker', False)}, hybrid={config.get('use_hybrid', False)}")
    logger.info(f"{'=' * 50}\n")

    # 创建 Pipeline（带特定配置）
    pipeline = RAGPipeline(
        use_hyde=config.get("use_hyde", False),
        use_reranker=config.get("use_reranker", False),
        use_hybrid=config.get("use_hybrid", False),
        top_k=top_k,
        rerank_top_k=rerank_top_k,
        vector_top_k=vector_top_k,
        bm25_top_k=bm25_top_k,
    )

    # 记录时间
    start_time = time.time()

    try:
        # 执行评估
        eval_results = evaluator.evaluate(pipeline)
        elapsed_time = time.time() - start_time

        # 提取关键指标
        result = {
            "config": config,
            "metrics": {
                "faithfulness": float(eval_results.get("faithfulness", 0)),
                "answer_relevancy": float(eval_results.get("answer_relevancy", 0)),
                "context_precision": float(eval_results.get("context_precision", 0)),
                "context_recall": float(eval_results.get("context_recall", 0)),
            },
            "elapsed_time": elapsed_time,
            "timestamp": datetime.now().isoformat(),
            "status": "success",
        }

        logger.info(f"实验完成: {config['name']}")
        logger.info(f"耗时: {elapsed_time:.2f}秒")
        for metric, value in result["metrics"].items():
            logger.info(f"  {metric}: {value:.4f}")

    except Exception as e:
        logger.error(f"实验失败: {config['name']}, 错误: {e}")
        result = {
            "config": config,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "status": "failed",
        }

    return result


def run_comparison_experiment(
        output_dir: str = "evaluation/results",
        top_k: int = 5,
        rerank_top_k: int = 50,
        sample_size: Optional[int] = None,
) -> Dict:
    """
    运行四组对比实验

    Args:
        output_dir: 结果输出目录
        top_k: 最终返回的文档数
        rerank_top_k: 粗排时检索的文档数
        sample_size: 样本数量，None 表示使用全部数据

    Returns:
        所有实验结果汇总
    """
    results = {}

    # 创建评估器
    evaluator = RAGASEvaluator()

    # 如果指定了样本数量，使用子集
    if sample_size:
        from evaluation.eval_dataset import EVAL_DATASET
        evaluator = RAGASEvaluator(test_data=EVAL_DATASET[:sample_size])
        logger.info(f"使用样本子集: {sample_size} 条")

    # 运行每组实验
    for config in EXPERIMENT_CONFIGS:
        result = run_single_experiment(
            config=config,
            evaluator=evaluator,
            top_k=top_k,
            rerank_top_k=rerank_top_k,
        )
        results[config["name"]] = result

    # 保存结果
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 保存 JSON 结果
    result_file = output_path / "comparison_results.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"结果已保存: {result_file}")

    # 生成 Markdown 报告
    report_file = output_path / "comparison_report.md"
    generate_comparison_report(results, report_file)
    logger.info(f"报告已生成: {report_file}")

    return results


def generate_comparison_report(results: Dict, output_file: Path):
    """生成 Markdown 格式的对比报告"""
    report = []

    # 标题
    report.append("# Phase 2 检索优化对比实验报告")
    report.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 实验配置
    report.append("\n## 实验配置")
    report.append("- 评估数据集: 50条农业领域问答")
    report.append("- 最终 Top-K: 5")
    report.append("- Reranker 粗排 Top-K: 50")

    # 评估指标说明
    report.append("\n## 评估指标")
    report.append("| 指标 | 说明 |")
    report.append("|------|------|")
    report.append("| Faithfulness | 答案是否忠实于检索内容 |")
    report.append("| Answer Relevancy | 答案与问题的相关性 |")
    report.append("| Context Precision | 检索内容的精确度 |")
    report.append("| Context Recall | 检索内容的召回率 |")

    # 实验结果表格
    report.append("\n## 实验结果对比")
    report.append("\n| 实验 | Faithfulness | Answer Relevancy | Context Precision | Context Recall |")
    report.append("|------|-------------|-----------------|-------------------|---------------|")

    for name, data in results.items():
        if data.get("status") == "success" and "metrics" in data:
            m = data["metrics"]
            report.append(
                f"| {name} | {m['faithfulness']:.4f} | "
                f"{m['answer_relevancy']:.4f} | {m['context_precision']:.4f} | "
                f"{m['context_recall']:.4f} |"
            )
        else:
            error_msg = data.get("error", "Unknown error")
            report.append(f"| {name} | ERROR | ERROR | ERROR | ERROR |")

    # 计算改进幅度
    report.append("\n## 相对基线的改进")
    if "Baseline" in results and results["Baseline"].get("status") == "success":
        baseline = results["Baseline"]["metrics"]
        report.append("\n| 实验 | Answer Relevancy 改进 | Context Recall 改进 |")
        report.append("|------|----------------------|---------------------|")

        for name, data in results.items():
            if name == "Baseline" or data.get("status") != "success":
                continue
            if "metrics" not in data:
                continue

            ar_improve = (data["metrics"]["answer_relevancy"] - baseline["answer_relevancy"]) * 100
            cr_improve = (data["metrics"]["context_recall"] - baseline["context_recall"]) * 100
            report.append(f"| {name} | {ar_improve:+.2f}% | {cr_improve:+.2f}% |")

    # 结论
    report.append("\n## 结论")

    # 找出最佳配置
    successful_results = [
        (name, data["metrics"]["answer_relevancy"] + data["metrics"]["context_recall"])
        for name, data in results.items()
        if data.get("status") == "success" and "metrics" in data
    ]

    if successful_results:
        best_config = max(successful_results, key=lambda x: x[1])
        report.append(f"\n最佳配置: **{best_config[0]}**")
        report.append(f"\n相比 Baseline，Answer Relevancy + Context Recall 总分提升：")

        if "Baseline" in results and results["Baseline"].get("status") == "success":
            baseline_score = (
                    results["Baseline"]["metrics"]["answer_relevancy"] +
                    results["Baseline"]["metrics"]["context_recall"]
            )
            improvement = (best_config[1] - baseline_score) * 50  # 转为百分比
            report.append(f"- 总分从 {baseline_score * 50:.2f}% 提升至 {best_config[1] * 50:.2f}%")
            report.append(f"- 绝对提升 {improvement:.2f}%")

    # 简历描述建议
    report.append("\n## 简历描述建议")
    report.append("\n```")
    report.append("构建 50 条农业领域评估集，引入 HyDE 与 BGE-Reranker 对检索链路优化，")
    report.append("Answer Relevancy 从 X% 提升至 Y%，Context Recall 提升 Z 个百分点，")
    report.append("使用 RAGAS 框架完成量化评估。")
    report.append("```")

    # 写入文件
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report))


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="运行 Phase 2 检索优化对比实验")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="样本数量，不指定则使用全部 50 条数据",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation/results",
        help="结果输出目录",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="最终返回的文档数",
    )
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=50,
        help="Reranker 粗排数量",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Phase 2 检索优化对比实验")
    logger.info("=" * 60)

    results = run_comparison_experiment(
        output_dir=args.output_dir,
        top_k=args.top_k,
        rerank_top_k=args.rerank_top_k,
        sample_size=args.sample_size,
    )

    # 打印最终摘要
    logger.info("\n" + "=" * 60)
    logger.info("实验完成！结果摘要：")
    logger.info("=" * 60)

    for name, data in results.items():
        if data.get("status") == "success":
            m = data["metrics"]
            logger.info(f"\n{name}:")
            logger.info(f"  Faithfulness: {m['faithfulness']:.4f}")
            logger.info(f"  Answer Relevancy: {m['answer_relevancy']:.4f}")
            logger.info(f"  Context Precision: {m['context_precision']:.4f}")
            logger.info(f"  Context Recall: {m['context_recall']:.4f}")
        else:
            logger.info(f"\n{name}: 失败 - {data.get('error', 'Unknown')}")


if __name__ == "__main__":
    main()