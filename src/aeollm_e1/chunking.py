from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

TOKEN_RE = re.compile(
    r"https?://\S+|\d+(?:[.,]\d+)*|[A-Za-z][A-Za-z0-9]*(?:[-'][A-Za-z0-9]+)*|"
    r"[\u3400-\u9fff]|[^\s]"
)
NUMBERED_HEADING_RE = re.compile(
    r"^(?:第[一二三四五六七八九十百0-9]+[章节篇部分]|"
    r"[一二三四五六七八九十]+[、.]|"
    r"\d+(?:\.\d+){0,3}[、.：:)）]?\s*)"
)
NAMED_HEADING_RE = re.compile(
    r"^(?:摘要|引言|前言|绪论|结论|总结|讨论|建议|参考文献|附录|目录|"
    r"abstract|introduction|conclusion|references|appendix)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ChunkingConfig:
    min_chunk_tokens: int = 64
    target_chunk_tokens: int = 384
    max_chunk_tokens: int = 512
    max_chunks: int | None = 96
    heading_max_tokens: int = 40

    def validate(self) -> None:
        if not 0 < self.min_chunk_tokens <= self.target_chunk_tokens <= self.max_chunk_tokens:
            raise ValueError("expected 0 < min <= target <= max chunk tokens")
        if self.max_chunks is not None and self.max_chunks <= 0:
            raise ValueError("max_chunks must be positive or None")
        if self.heading_max_tokens <= 0:
            raise ValueError("heading_max_tokens must be positive")


@dataclass
class SourceBlock:
    block_id: int
    type: str
    text: str
    style: str = ""
    heading_reason: str = ""
    table_rows: list[str] | None = None


@dataclass
class Chunk:
    chunk_id: int
    type: str
    text: str
    token_count: int
    source_block_ids: list[int]
    source_types: list[str]
    split_index: int = 0
    split_count: int = 1
    overflow_merged: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def token_spans(text: str) -> list[tuple[int, int]]:
    return [match.span() for match in TOKEN_RE.finditer(text)]


def count_tokens(text: str) -> int:
    return len(token_spans(text))


def _paragraph_heading_reason(paragraph: Paragraph, text: str, config: ChunkingConfig) -> str:
    style = (getattr(paragraph.style, "name", "") or "").strip().lower()
    if style.startswith(("heading", "title", "标题")):
        return "style"
    properties = paragraph._p.pPr
    if properties is not None and properties.outlineLvl is not None:
        return "outline_level"
    if count_tokens(text) > config.heading_max_tokens:
        return ""
    stripped = text.strip().rstrip("：:")
    if NAMED_HEADING_RE.fullmatch(stripped):
        return "named_heading"
    if NUMBERED_HEADING_RE.match(stripped) and not re.search(r"[。！？!?]$", stripped):
        return "numbered_heading"
    return ""


def _paragraph_is_list(paragraph: Paragraph) -> bool:
    style = (getattr(paragraph.style, "name", "") or "").lower()
    if "list" in style or "列表" in style:
        return True
    properties = paragraph._p.pPr
    return properties is not None and properties.numPr is not None


def _normalize_cell(text: str) -> str:
    parts = [part.strip() for part in str(text).replace("\r", "\n").split("\n") if part.strip()]
    return " <br> ".join(parts).replace("\t", " ")


def _table_rows(table: Table) -> list[str]:
    rows: list[str] = []
    seen: set[object] = set()
    for row in table.rows:
        cells: list[str] = []
        for cell in row.cells:
            key = cell._tc
            if key in seen:
                cells.append("")
            else:
                seen.add(key)
                cells.append(_normalize_cell(cell.text))
        rows.append(" | ".join(cells).rstrip())
    return rows


def read_docx_blocks(path: Path, config: ChunkingConfig | None = None) -> list[SourceBlock]:
    config = config or ChunkingConfig()
    config.validate()
    document = Document(str(path))
    blocks: list[SourceBlock] = []
    for item in document.iter_inner_content():
        if isinstance(item, Paragraph):
            text = item.text.strip()
            if not text:
                continue
            is_list = _paragraph_is_list(item)
            reason = _paragraph_heading_reason(item, text, config)
            if is_list and reason == "numbered_heading":
                reason = ""
            block_type = "heading" if reason else ("list" if is_list else "paragraph")
            blocks.append(
                SourceBlock(
                    block_id=len(blocks),
                    type=block_type,
                    text=text,
                    style=(getattr(item.style, "name", "") or ""),
                    heading_reason=reason,
                )
            )
        elif isinstance(item, Table):
            rows = _table_rows(item)
            if any(row.strip(" |") for row in rows):
                blocks.append(
                    SourceBlock(
                        block_id=len(blocks),
                        type="table",
                        text="\n".join(rows),
                        table_rows=rows,
                    )
                )
    return blocks


def _balanced_text_slices(text: str, max_tokens: int, target_tokens: int | None = None) -> list[str]:
    spans = token_spans(text)
    if len(spans) <= max_tokens:
        return [text.strip()]
    target_tokens = target_tokens or max_tokens
    part_count = max(math.ceil(len(spans) / max_tokens), round(len(spans) / target_tokens))
    tokens_per_part = math.ceil(len(spans) / part_count)
    result: list[str] = []
    start_char = 0
    for start_token in range(0, len(spans), tokens_per_part):
        end_token = min(start_token + tokens_per_part, len(spans))
        end_char = spans[end_token][0] if end_token < len(spans) else len(text)
        value = text[start_char:end_char].strip()
        if value:
            result.append(value)
        start_char = end_char
    return result


def _split_table(block: SourceBlock, config: ChunkingConfig) -> list[str]:
    rows = block.table_rows or [block.text]
    chunks: list[str] = []
    pending: list[str] = []
    pending_tokens = count_tokens("[TABLE]\n")
    for row in rows:
        row_parts = _balanced_text_slices(
            row,
            max(1, config.max_chunk_tokens - 2),
            max(1, config.target_chunk_tokens - 2),
        )
        for part in row_parts:
            part_tokens = count_tokens(part)
            if pending and pending_tokens + part_tokens > config.max_chunk_tokens:
                chunks.append("[TABLE]\n" + "\n".join(pending))
                pending = []
                pending_tokens = count_tokens("[TABLE]\n")
            pending.append(part)
            pending_tokens += part_tokens
    if pending:
        chunks.append("[TABLE]\n" + "\n".join(pending))
    return chunks


def _atomic_chunks(blocks: list[SourceBlock], config: ChunkingConfig) -> list[Chunk]:
    chunks: list[Chunk] = []
    for block in blocks:
        if block.type == "heading":
            parts = [block.text]
        elif block.type == "table":
            parts = _split_table(block, config)
        else:
            parts = _balanced_text_slices(
                block.text, config.max_chunk_tokens, config.target_chunk_tokens
            )
        for index, part in enumerate(parts):
            chunks.append(
                Chunk(
                    chunk_id=-1,
                    type=block.type,
                    text=part,
                    token_count=count_tokens(part),
                    source_block_ids=[block.block_id],
                    source_types=[block.type],
                    split_index=index,
                    split_count=len(parts),
                )
            )
    return chunks


def _merge_pair(left: Chunk, right: Chunk, *, overflow: bool = False) -> Chunk:
    separator = "\n\n" if left.type != "table" else "\n"
    text = left.text + separator + right.text
    types = list(dict.fromkeys([*left.source_types, *right.source_types]))
    return Chunk(
        chunk_id=-1,
        type=left.type if left.type == right.type else "mixed",
        text=text,
        token_count=count_tokens(text),
        source_block_ids=[*left.source_block_ids, *right.source_block_ids],
        source_types=types,
        overflow_merged=overflow or left.overflow_merged or right.overflow_merged,
    )


def _merge_short_neighbors(chunks: list[Chunk], config: ChunkingConfig) -> list[Chunk]:
    result: list[Chunk] = []
    for chunk in chunks:
        if not result:
            result.append(chunk)
            continue
        previous = result[-1]
        compatible = previous.type == chunk.type and chunk.type in {"paragraph", "list"}
        is_short_pair = min(previous.token_count, chunk.token_count) < config.min_chunk_tokens
        if compatible and is_short_pair and previous.token_count + chunk.token_count <= config.max_chunk_tokens:
            result[-1] = _merge_pair(previous, chunk)
        else:
            result.append(chunk)
    return result


def _enforce_chunk_budget(chunks: list[Chunk], config: ChunkingConfig) -> list[Chunk]:
    """Preserve all content by adjacent compaction; never select or rank chunks."""
    if config.max_chunks is None:
        return list(chunks)
    result = list(chunks)
    while len(result) > config.max_chunks:
        candidates = []
        for index, (left, right) in enumerate(zip(result, result[1:])):
            combined_tokens = count_tokens(left.text + "\n\n" + right.text)
            size_penalty = 10_000_000 if combined_tokens > config.max_chunk_tokens else 0
            structural_penalty = 0
            if left.type == "heading" or right.type == "heading":
                structural_penalty += 1_000_000
            if left.type != right.type:
                structural_penalty += 100_000
            candidates.append((size_penalty + structural_penalty + combined_tokens, index))
        _, index = min(candidates)
        result[index : index + 2] = [_merge_pair(result[index], result[index + 1], overflow=True)]
    return result


def chunk_blocks(blocks: list[SourceBlock], config: ChunkingConfig | None = None) -> list[Chunk]:
    config = config or ChunkingConfig()
    config.validate()
    chunks = _merge_short_neighbors(_atomic_chunks(blocks, config), config)
    chunks = _enforce_chunk_budget(chunks, config)
    for index, chunk in enumerate(chunks):
        chunk.chunk_id = index
    return chunks


def chunk_docx(path: Path, config: ChunkingConfig | None = None) -> tuple[list[SourceBlock], list[Chunk]]:
    config = config or ChunkingConfig()
    blocks = read_docx_blocks(path, config)
    return blocks, chunk_blocks(blocks, config)
