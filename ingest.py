"""
文档入库脚本
把 data/raw/ 目录下的所有文档处理后写入 ChromaDB

运行方式:
    python ingest.py
    python ingest.py --dir data/raw --clear   # 清空重建
    python ingest.py --dir data/raw --stats   # 只看统计不入库
"""
import argparse
import sys
from pathlib import Path

import logging

logger = logging.getLogger(__name__)

# 确保项目根目录在 Python 路径中
sys.path.insert(0, str(Path(__file__).parent))

from core.document_processor import DocumentProcessor
from core.embedder import BGEEmbedder
from core.vector_store import VectorStore


def main():
    parser = argparse.ArgumentParser(description="农业知识库文档入库工具")
    parser.add_argument("--dir", default="data/raw", help="文档目录路径")
    parser.add_argument("--clear", action="store_true", help="清空知识库后重建")
    parser.add_argument("--stats", action="store_true", help="只显示统计信息")
    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info("农业 RAG 知识库构建")
    logger.info("=" * 50)

    # 初始化组件（BGEEmbedder 首次加载会下载模型）
    embedder = BGEEmbedder()
    vs = VectorStore(embedder=embedder)
    processor = DocumentProcessor()

    if args.stats:
        logger.info(f"当前知识库文档数: {vs.count()}")
        return

    if args.clear:
        confirm = input("确认清空知识库？(yes/no): ")
        if confirm.lower() != "yes":
            logger.info("取消操作")
            return
        vs.clear()

    # Step 1: 加载文档
    logger.info(f"Step 1/3: 从 {args.dir} 加载文档...")
    raw_docs = processor.load_directory(args.dir)

    if not raw_docs:
        logger.warning(f"未找到任何文档，请检查目录: {args.dir}")
        logger.info("支持格式: .pdf, .txt, .md")
        logger.info("提示：可以先放几个测试文档进去试运行")
        return

    # Step 2: 文档切分
    logger.info("Step 2/3: 文档切分...")
    chunks = processor.split_documents(raw_docs)
    logger.info(f"切分结果：{len(raw_docs)} 篇 → {len(chunks)} 个 chunks")

    # Step 3: 向量化并写入
    logger.info("Step 3/3: 向量化并写入 ChromaDB...")
    written = vs.add_documents(chunks)

    logger.info("=" * 50)
    logger.info(f"入库完成！写入: {written} 条，知识库总量: {vs.count()} 条")
    logger.info("现在可以运行: python main.py 开始问答")


if __name__ == "__main__":
    main()