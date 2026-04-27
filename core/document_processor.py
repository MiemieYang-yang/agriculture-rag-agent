"""
文档处理模块
支持 PDF、TXT、Markdown、Word 四种格式
核心逻辑：加载 → 清洗 → 按标题层级切分（MarkdownHeaderTextSplitter）→ 超长段落二次切分（RecursiveCharacterTextSplitter）
"""
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import logging
from tqdm import tqdm

from core.config import cfg

# LangChain 文本切分器
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# 标题层级切分配置
HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]
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
        supported = {".pdf", ".txt", ".md", ".docx"}

        for fp in tqdm(files, desc="加载文档"):
            if fp.suffix.lower() not in supported:
                continue
            # 过滤 Word 临时文件（以 ~$ 开头）
            if fp.name.startswith("~$"):
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
        """
        将原始文档切分为固定大小的 chunks

        注意：Word (.docx) 和 Markdown (.md) 文档已在加载时完成切分，
        此方法仅处理 PDF 和 TXT 文档的切分。
        """
        chunks: List[Document] = []
        for doc in docs:
            # 检查是否已被切分（docx/md 已在加载时切分）
            if "chunk_index" in doc.metadata:
                # 已经是切分后的 chunk，直接添加
                chunks.append(doc)
            else:
                # PDF/TXT 需要切分
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
        elif suffix == ".md":
            return self._load_markdown(fp)
        elif suffix == ".docx":
            return self._load_docx(fp)
        elif suffix == ".txt":
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
        """基础文本清洗（用于 PDF/TXT，合并多余空白）"""
        text = re.sub(r'\s+', ' ', text)  # 合并多余空白
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)  # 去控制字符
        return text.strip()

    def _clean_markdown(self, text: str) -> str:
        """Markdown 文本清洗（保留换行符，只合并行内多余空格）"""
        # 去控制字符
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
        # 移除 HTML anchor 标签（Word 书签链接）
        text = re.sub(r'<a\s+id="[^"]*">\s*</a>', '', text)
        # 合并行内多余空格（但保留换行）
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            # 合并行内的多余空格
            cleaned_line = re.sub(r'[ \t]+', ' ', line)
            cleaned_lines.append(cleaned_line.rstrip())
        return '\n'.join(cleaned_lines).strip()

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

    # ── Word/Markdown 处理方法 ─────────────────────────────────────────────────

    def _load_docx(self, fp: Path) -> List[Document]:
        """
        加载 Word 文档，转换为 Markdown 后按标题切分。

        步骤：
        1. mammoth 将 .docx 转为 Markdown
        2. 过滤目录行
        3. 清洗文本
        4. 保存转换后的 md 文件（可选）
        5. 按标题层级切分
        """
        try:
            import mammoth
        except ImportError:
            raise ImportError("请安装 mammoth: pip install mammoth")

        # 转换为 Markdown
        markdown_text = self._convert_docx_to_markdown(fp)

        # 过滤目录行
        markdown_text = self._filter_toc_lines(markdown_text)

        # 清洗（保留换行符）
        markdown_text = self._clean_markdown(markdown_text)

        if not markdown_text.strip():
            return []

        # 保存转换后的 md 文件
        saved_md_path = self._save_converted_markdown(markdown_text, fp)
        if saved_md_path:
            logger.info(f"已保存转换后的 Markdown: {saved_md_path}")

        # 构建基础元数据
        base_metadata = {
            "source": str(fp),
            "filename": fp.name,
            "file_type": "docx",
        }

        # 按标题切分
        return self._split_with_headers(markdown_text, base_metadata, fp.stem)

    def _convert_docx_to_markdown(self, fp: Path) -> str:
        """
        使用 mammoth 将 Word 文档转换为 Markdown。

        mammoth 自动处理：
        - Heading 1/2/3/4 → # ## ### ####
        - 页眉页脚自动忽略
        - 表格转换为 Markdown 格式
        """
        try:
            import mammoth
        except ImportError:
            raise ImportError("请安装 mammoth: pip install mammoth")

        with open(fp, "rb") as f:
            result = mammoth.convert_to_markdown(f)
            markdown_text = result.value

            # 记录转换警告
            if result.messages:
                for msg in result.messages:
                    logger.warning(f"mammoth 转换警告 [{fp.name}]: {msg}")

        return markdown_text

    def _save_converted_markdown(self, markdown_text: str, original_fp: Path) -> str:
        """
        保存转换后的 Markdown 文件到 data/processed 目录。
        
        Args:
            markdown_text: 转换后的 Markdown 文本
            original_fp: 原始 docx 文件路径
            
        Returns:
            保存的文件路径，如果保存失败则返回空字符串
        """
        try:
            # 创建 processed 目录
            project_root = Path(__file__).parent.parent
            processed_dir = project_root / "data" / "processed"
            processed_dir.mkdir(parents=True, exist_ok=True)
            
            # 生成 md 文件名（保持原名，只改后缀）
            md_filename = original_fp.stem + ".md"
            md_path = processed_dir / md_filename
            
            # 写入文件
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(markdown_text)
            
            return str(md_path)
        except Exception as e:
            logger.warning(f"保存 Markdown 文件失败: {e}")
            return ""

    def _filter_toc_lines(self, text: str) -> str:
        """
        过滤目录行。

        目录行格式："第一章 概述……3" 或 "1.1 简介....5"
        正则匹配：以文字开头，中间有连续点号，末尾是页码
        """
        lines = text.split('\n')
        filtered_lines = []

        # 目录行正则：文字 + 2个以上点号 + 页码
        toc_pattern = re.compile(r'.+[．.…]{2,}\d+\s*$')

        for line in lines:
            stripped = line.strip()
            # 过滤目录行
            if not toc_pattern.match(stripped):
                filtered_lines.append(line)

        return '\n'.join(filtered_lines)

    def _load_markdown(self, fp: Path) -> List[Document]:
        """加载 Markdown 文件，按标题层级切分"""
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        # 清洗（保留换行符）
        text = self._clean_markdown(text)

        if not text.strip():
            return []

        base_metadata = {
            "source": str(fp),
            "filename": fp.name,
            "file_type": "md",
        }

        return self._split_with_headers(text, base_metadata, fp.stem)

    def _split_with_headers(
        self,
        text: str,
        base_metadata: dict,
        doc_id_prefix: str
    ) -> List[Document]:
        """
        使用 MarkdownHeaderTextSplitter 按标题层级切分。

        处理流程：
        1. 识别 # ## ### #### 四级标题
        2. 每个标题下的内容成为一个 chunk
        3. 标题文本存入 metadata（h1/h2/h3/h4 字段）
        4. 超长 chunk 使用 RecursiveCharacterTextSplitter 二次切分
        """
        # 初始化 MarkdownHeaderTextSplitter
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=HEADERS_TO_SPLIT_ON,
            strip_headers=False,  # 标题保留在 content 中
            return_each_line=False,
        )

        # 按标题切分
        try:
            md_header_splits = markdown_splitter.split_text(text)
        except Exception as e:
            logger.warning(f"MarkdownHeaderTextSplitter 切分失败: {e}，回退到普通切分")
            # 回退到普通切分
            return self._fallback_split(text, base_metadata, doc_id_prefix)

        # 如果没有识别到标题，返回单个文档
        if not md_header_splits:
            return [Document(
                content=text,
                metadata={**base_metadata, "chunk_index": 0, "chunk_total": 1},
                doc_id=f"{doc_id_prefix}_chunk0",
            )]

        documents = []
        chunk_index = 0

        for split_doc in md_header_splits:
            content = split_doc.page_content

            # 提取标题层级信息
            header_info = {
                "h1": split_doc.metadata.get("h1"),
                "h2": split_doc.metadata.get("h2"),
                "h3": split_doc.metadata.get("h3"),
                "h4": split_doc.metadata.get("h4"),
            }

            # 检查是否需要二次切分
            if len(content) > self.chunk_size:
                # 二次切分
                sub_docs = self._split_oversized_chunk(content, base_metadata, header_info)
                for sub_doc in sub_docs:
                    sub_doc.metadata["chunk_index"] = chunk_index
                    sub_doc.doc_id = f"{doc_id_prefix}_chunk{chunk_index}"
                    documents.append(sub_doc)
                    chunk_index += 1
            else:
                # 直接构建 chunk
                metadata = self._build_chunk_metadata(base_metadata, header_info, chunk_index, 0)
                documents.append(Document(
                    content=content,
                    metadata=metadata,
                    doc_id=f"{doc_id_prefix}_chunk{chunk_index}",
                ))
                chunk_index += 1

        # 更新 chunk_total
        total = len(documents)
        for doc in documents:
            doc.metadata["chunk_total"] = total

        return documents

    def _split_oversized_chunk(
        self,
        content: str,
        base_metadata: dict,
        header_info: dict
    ) -> List[Document]:
        """
        对超长 chunk 进行二次切分，保护表格不被切分。

        逻辑：
        1. 检测是否包含表格
        2. 如果有表格，先分离表格和普通文本
        3. 普通文本使用 RecursiveCharacterTextSplitter 切分
        4. 表格作为独立 chunk 保留
        """
        documents = []

        # 检测表格
        if self._contains_table(content):
            # 分离表格和普通文本
            tables, non_table_parts = self._extract_table_blocks(content)

            # 处理普通文本
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],
                length_function=len,
            )

            for text_part in non_table_parts:
                if text_part.strip():
                    splits = text_splitter.split_text(text_part)
                    for split in splits:
                        metadata = self._build_chunk_metadata(base_metadata, header_info, 0, 0)
                        metadata["contains_table"] = False
                        documents.append(Document(
                            content=split,
                            metadata=metadata,
                            doc_id="",  # 由调用者设置
                        ))

            # 表格作为独立 chunk（不切分）
            for table in tables:
                if table.strip():
                    metadata = self._build_chunk_metadata(base_metadata, header_info, 0, 0)
                    metadata["contains_table"] = True
                    documents.append(Document(
                        content=table,
                        metadata=metadata,
                        doc_id="",  # 由调用者设置
                    ))
        else:
            # 无表格，直接切分
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],
                length_function=len,
            )

            splits = text_splitter.split_text(content)
            for split in splits:
                metadata = self._build_chunk_metadata(base_metadata, header_info, 0, 0)
                documents.append(Document(
                    content=split,
                    metadata=metadata,
                    doc_id="",  # 由调用者设置
                ))

        return documents

    def _contains_table(self, text: str) -> bool:
        """
        检测文本是否包含 Markdown 表格。

        Markdown 表格格式：
        | Header 1 | Header 2 |
        |----------|----------|
        | Cell 1   | Cell 2   |
        """
        # 检测以 | 开头且包含多个 | 的行
        table_line_pattern = re.compile(r'^\|.+\|', re.MULTILINE)
        return bool(table_line_pattern.search(text))

    def _extract_table_blocks(self, text: str) -> Tuple[List[str], List[str]]:
        """
        分离表格块和普通文本块。

        返回：(tables, non_table_text_parts)
        """
        tables = []
        non_table_parts = []

        lines = text.split('\n')
        current_block = []
        in_table = False

        for line in lines:
            # 判断是否为表格行
            is_table_line = line.strip().startswith('|') and '|' in line[1:] if line.strip() else False

            if is_table_line:
                if not in_table:
                    # 结束当前非表格块
                    if current_block:
                        non_table_parts.append('\n'.join(current_block))
                        current_block = []
                    in_table = True
                current_block.append(line)
            else:
                if in_table:
                    # 结束当前表格块
                    if current_block:
                        tables.append('\n'.join(current_block))
                        current_block = []
                    in_table = False
                current_block.append(line)

        # 处理剩余块
        if current_block:
            if in_table:
                tables.append('\n'.join(current_block))
            else:
                non_table_parts.append('\n'.join(current_block))

        return tables, non_table_parts

    def _build_chunk_metadata(
        self,
        base: dict,
        header_info: dict,
        chunk_index: int,
        total: int
    ) -> dict:
        """
        构建 chunk 元数据，包含标题层级信息。
        """
        metadata = {**base}  # 复制基础元数据

        # 添加标题层级（仅非空值，清理格式标记）
        for key in ["h1", "h2", "h3", "h4"]:
            value = header_info.get(key)
            if value is not None:
                # 清理 Markdown 格式标记（加粗、斜体等）
                cleaned_value = self._clean_header_text(value)
                metadata[key] = cleaned_value

        metadata["chunk_index"] = chunk_index
        metadata["chunk_total"] = total

        return metadata

    def _clean_header_text(self, text: str) -> str:
        """清理标题文本中的 Markdown 格式标记"""
        # 移除加粗标记 __text__ 或 **text**
        text = re.sub(r'^__(.+)__$', r'\1', text)
        text = re.sub(r'^\*\*(.+)\*\*$', r'\1', text)
        # 移除斜体标记 _text_ 或 *text*
        text = re.sub(r'^_(.+)_$', r'\1', text)
        text = re.sub(r'^\*(.+)\*$', r'\1', text)
        # 清理多余空格
        text = text.strip()
        return text

    def _fallback_split(
        self,
        text: str,
        base_metadata: dict,
        doc_id_prefix: str
    ) -> List[Document]:
        """
        回退切分方法，当 MarkdownHeaderTextSplitter 失败时使用。
        """
        chunks_text = self._split_text(text)
        documents = []

        for i, chunk_text in enumerate(chunks_text):
            documents.append(Document(
                content=chunk_text,
                metadata={
                    **base_metadata,
                    "chunk_index": i,
                    "chunk_total": len(chunks_text),
                },
                doc_id=f"{doc_id_prefix}_chunk{i}",
            ))

        return documents