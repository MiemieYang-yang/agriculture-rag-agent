"""
文档处理模块
支持 PDF、TXT、Markdown 三种格式
核心逻辑：加载 → 清洗 → 按字符切分（带重叠）→ 附加元数据
"""
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import logging
from tqdm import tqdm

from core.config import cfg

logger = logging.getLogger(__name__)
@dataclass
class Document:
    """单个文本块（chunk）的数据结构"""
    content: str  # 文本内容
    metadata: dict = field(default_factory=dict)  # 来源、页码等信息
    doc_id: str = ""  # 唯一 ID，写入 ChromaDB 时用


class DocumentProcessor:
    """
    文档处理器

    使用方式:
        processor = DocumentProcessor()
        docs = processor.load_directory("data/raw")
        chunks = processor.split_documents(docs)
    """

    def __init__(
            self,
            chunk_size: int = cfg.CHUNK_SIZE,
            chunk_overlap: int = cfg.CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ── 公开接口 ─────────────────────────────────────────────────────────────

    def load_directory(self, dir_path: str) -> List[Document]:
        """加载目录下所有支持的文档"""
        dir_path = Path(dir_path)
        if not dir_path.exists():
            raise FileNotFoundError(f"目录不存在: {dir_path}")

        all_docs: List[Document] = []
        files = list(dir_path.rglob("*"))
        supported = {".pdf", ".txt", ".md"}

        for fp in tqdm(files, desc="加载文档"):
            if fp.suffix.lower() not in supported:
                continue
            try:
                docs = self._load_single_file(fp)
                all_docs.extend(docs)
                logger.info(f"加载成功: {fp.name} ({len(docs)} 段)")
            except Exception as e:
                logger.warning(f"加载失败: {fp.name} — {e}")

        logger.info(f"共加载 {len(all_docs)} 段原始文档")
        return all_docs

    def split_documents(self, docs: List[Document]) -> List[Document]:
        """将原始文档切分为固定大小的 chunks"""
        chunks: List[Document] = []
        for doc in docs:
            doc_chunks = self._split_text(doc.content)
            for i, chunk_text in enumerate(doc_chunks):
                chunk = Document(
                    content=chunk_text,
                    metadata={
                        **doc.metadata,
                        "chunk_index": i,
                        "chunk_total": len(doc_chunks),
                    },
                    doc_id=f"{doc.doc_id}_chunk{i}",
                )
                chunks.append(chunk)

        logger.info(f"切分完成，共 {len(chunks)} 个 chunks（chunk_size={self.chunk_size}, overlap={self.chunk_overlap}）")
        return chunks

    # ── 私有方法 ─────────────────────────────────────────────────────────────

    def _load_single_file(self, fp: Path) -> List[Document]:
        suffix = fp.suffix.lower()
        if suffix == ".pdf":
            return self._load_pdf(fp)
        elif suffix in {".txt", ".md"}:
            return self._load_text(fp)
        else:
            raise ValueError(f"不支持的格式: {suffix}")

    def _load_pdf(self, fp: Path) -> List[Document]:
        """逐页加载 PDF，每页作为一个原始 Document"""
        try:
            import pypdf
        except ImportError:
            raise ImportError("请安装 pypdf: pip install pypdf")

        docs = []
        with open(fp, "rb") as f:
            reader = pypdf.PdfReader(f)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                text = self._clean_text(text)
                if len(text) < 50:  # 跳过几乎为空的页
                    continue
                docs.append(Document(
                    content=text,
                    metadata={
                        "source": str(fp),
                        "filename": fp.name,
                        "page": page_num + 1,
                        "file_type": "pdf",
                    },
                    doc_id=f"{fp.stem}_p{page_num + 1}",
                ))
        return docs

    def _load_text(self, fp: Path) -> List[Document]:
        """加载 TXT / Markdown 文件，整个文件作为一个原始 Document"""
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        text = self._clean_text(text)
        if not text.strip():
            return []
        return [Document(
            content=text,
            metadata={
                "source": str(fp),
                "filename": fp.name,
                "file_type": fp.suffix.lstrip("."),
            },
            doc_id=fp.stem,
        )]

    def _clean_text(self, text: str) -> str:
        """基础文本清洗"""
        text = re.sub(r'\s+', ' ', text)  # 合并多余空白
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)  # 去控制字符
        return text.strip()

    def _split_text(self, text: str) -> List[str]:
        """
        按字符数切分，带重叠
        策略：尽量在句号/换行处断开，避免从句子中间截断
        """
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size

            if end >= len(text):
                chunks.append(text[start:])
                break

            # 往前找最近的句子边界（。！？\n）
            boundary = end
            for sep in ['。', '！', '？', '\n', '，', ' ']:
                # 从右向左查找分隔符的位置
                idx = text.rfind(sep, start + self.chunk_overlap, end)
                if idx != -1:
                    boundary = idx + 1
                    break

            chunks.append(text[start:boundary])
            # 下一个 chunk 从 (boundary - overlap) 开始，保证上下文连续
            start = max(start + 1, boundary - self.chunk_overlap)

        return chunks