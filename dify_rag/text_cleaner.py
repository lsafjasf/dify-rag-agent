"""
Dify 中文文档文本清洗模块
======================
用于 RAG Agent 的文本预处理流水线。

处理步骤:
  1. 文件发现 —— 遍历目录, 收集 .mdx / .json 文件
  2. Frontmatter 提取 —— 解析 YAML 头, 分离元数据与正文
  3. JSX/MDX 组件剥离 —— 移除 <Info>、<Card> 等组件标签, 保留内部文字
  4. HTML 标签清理 —— 去除 <div>、<a> 等 HTML 标签
  5. Markdown 整理 —— 保留标题结构, 清理图片/链接语法
  6. 空白规范化 —— 合并多余空行, 统一中英文空格
  7. 文本分块 —— 使用 LangChain splitter 切分为 RAG 可用片段
  8. 输出 Document —— 生成带元数据的 LangChain Document 列表
"""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 需要移除标签但保留内部文字的 MDX 组件 (配对标签)
MDX_PAIRED_COMPONENTS = [
    "Info", "Warning", "Note", "Tip", "Check",
    "Card", "CardGroup",
    "Frame",
    "Steps", "Step",
    "Accordion", "AccordionGroup",
    "Columns",
    "Tabs", "Tab",
]

# 纯装饰性组件 (自闭合, 无文字内容, 完全删除)
MDX_SELF_CLOSING_COMPONENTS = [
    "Frame",  # <Frame /> 也是自闭合的
]

# MDX 组件中需要保留的关键属性 —— 这些属性携带语义信息,
# 在剥离标签之前会被提取并注入到内容中
# 格式: 组件名 → [(属性名, 前缀)]
MDX_SEMANTIC_ATTRS: dict[str, list[tuple[str, str]]] = {
    "Card": [("title", "**"), ("href", "(")],   # title → **title**, href → (href)
    "Step": [("title", "**")],                   # title → **title**
    "Frame": [("caption", "")],                  # caption → caption
}

# href 属性的右括号 (在格式化时补充)
_HREF_ATTRS = {"href"}

# HTML 标签 —— 剥离标签, 保留内部文字
HTML_TAGS = [
    "div", "span", "p", "h1", "h2", "h3", "h4", "h5", "h6",
    "strong", "em", "b", "i", "u", "code", "pre",
    "a", "img", "br", "hr",
    "ul", "ol", "li",
    "table", "thead", "tbody", "tr", "th", "td",
    "section", "article", "header", "footer", "nav",
]

# AI 翻译声明行模式
AI_TRANSLATION_PATTERN = re.compile(
    r"> 本文档由 AI 自动翻译。如有任何不准确之处，请参考 \[英文原版\]\(.*?\)\s*"
)

# Markdown 图片: ![alt](url)
MD_IMAGE_PATTERN = re.compile(r"!\[.*?\]\(.*?\)")

# Markdown 链接: [text](url) → text (url)
# 同时保留链接文字和 URL 路径 —— URL 中包含大量专业知识
# (API 端点路径、参数名、层级结构等), 对 RAG 检索价值很高。
MD_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

# --------------------------------------------------------
# 数据结构
# --------------------------------------------------------


@dataclass
class CleanConfig:
    """文本清洗配置"""

    # 是否保留 Markdown 标题 (##, ### 等)
    keep_headers: bool = True

    # 是否移除 AI 翻译声明
    remove_translation_notice: bool = True

    # 最小行长度 (短于此值的行视为碎片, 会被移除)
    # 设为 1 以避免误删代码块中的单字符行 (如 JSON 的 { })
    min_line_length: int = 1

    # 连续空行合并阈值 (超过 N 个空行合并为 1 个)
    max_consecutive_newlines: int = 2

    # 分块大小 (字符数, 中文约 2 字符 = 1 token)
    chunk_size: int = 800

    # 分块重叠大小
    chunk_overlap: int = 150

    # 文档来源根目录 (写入 metadata)
    source_root: str = ""

    # 需要处理的文件后缀
    extensions: tuple[str, ...] = (".mdx", ".md", ".json")

    # 是否保留代码块
    keep_code_blocks: bool = True


@dataclass
class CleanedDocument:
    """清洗后的单篇文档"""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    source_path: str = ""


# --------------------------------------------------------
# 核心清洗逻辑
# --------------------------------------------------------


def discover_files(root: str | Path, extensions: tuple[str, ...] | None = None) -> list[Path]:
    """递归发现所有文档文件。

    Args:
        root: 文档根目录。
        extensions: 目标文件后缀, 默认 (".mdx", ".md")。

    Returns:
        按路径排序的文件列表。
    """
    if extensions is None:
        extensions = (".mdx", ".md")

    root = Path(root)
    files: list[Path] = []
    for ext in extensions:
        files.extend(root.rglob(f"*{ext}"))
    return sorted(files)


def extract_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """从 MDX 文本中提取 YAML frontmatter。

    首行必须是 `---`, 否则整个文本视为正文。

    Args:
        text: 原始 MDX 文本。

    Returns:
        (metadata_dict, body_text) 元组。metadata 可能为空 dict。
    """
    text = text.strip()
    if not text.startswith("---"):
        return {}, text

    # 找第二个 ---
    rest = text[3:]  # 跳过首个 ---
    end_idx = rest.find("\n---")
    if end_idx == -1:
        # 第二个 --- 在同一行? 极少数情况
        end_idx = rest.find("---")
        if end_idx == -1:
            return {}, text

    fm_text = rest[:end_idx].strip()
    body = rest[end_idx + 4:].strip()  # 跳过 \n---

    metadata: dict[str, Any] = {}
    if fm_text:
        try:
            parsed = yaml.safe_load(fm_text)
            if isinstance(parsed, dict):
                metadata = parsed
        except yaml.YAMLError:
            warnings.warn(f"YAML frontmatter 解析失败, 将跳过", RuntimeWarning)

    return metadata, body


def remove_translation_notice(text: str) -> str:
    """移除 AI 翻译声明行。"""
    return AI_TRANSLATION_PATTERN.sub("", text)


def _preserve_component_attrs(text: str) -> str:
    """在剥离 MDX 标签之前, 提取关键属性的语义信息注入到内容中。

    处理的组件及其属性:
      Card:  title → **title**, href → (href)
      Step:  title → **title**
      Frame: caption → caption

    例如:
      <Card title="API 参考" href="/zh/api">内容</Card>
      → **API 参考** (/zh/api)：内容

    经过此预处理后, strip_mdx_tags 仍会剥离剩余的标签壳,
    但关键信息已经以 Markdown 格式嵌入到了文本内容中。
    """
    def _replacer(match: re.Match) -> str:
        tag_name = match.group(1)
        inner = match.group(2).strip()
        full_tag = match.group(0)

        attr_specs = MDX_SEMANTIC_ATTRS.get(tag_name, [])
        parts: list[str] = []
        for attr_name, prefix in attr_specs:
            # 匹配 attr="value" 或 attr='value'
            m = re.search(rf'{attr_name}="([^"]*)"', full_tag)
            if not m:
                m = re.search(rf"{attr_name}='([^']*)'", full_tag)
            if not m:
                continue
            value = m.group(1)
            if attr_name in _HREF_ATTRS:
                parts.append(f"{prefix}{value})")
            elif prefix:
                parts.append(f"{prefix}{value}{prefix}")
            else:
                parts.append(f"{value}")

        if not parts:
            return inner  # 没有可提取的属性, 仅返回内部文字

        # 组装输出: **title** (href)：inner 或 **title**：inner 等
        prefix_str = " ".join(parts)
        if inner:
            return f"{prefix_str}：{inner}"
        return prefix_str

    # 对每个有语义属性的组件类型进行替换
    for component_name in MDX_SEMANTIC_ATTRS:
        pattern = re.compile(
            rf"<({component_name})(?:\s[^>]*)?>(.*?)</\1>",
            re.DOTALL,
        )
        text = pattern.sub(_replacer, text)

    return text


def strip_mdx_tags(text: str) -> str:
    """剥离 MDX/JSX 组件标签, 保留内部文字。

    策略:
    1. 反复从内向外剥离配对标签 (无嵌套的最内层优先)。
    2. 最多迭代 20 轮, 防止死循环。
    3. 移除残留的自闭合标签。

    Args:
        text: 包含 MDX 组件的文本。

    Returns:
        剥离后的纯文本。
    """
    all_components = "|".join(MDX_PAIRED_COMPONENTS)

    # 配对标签模式: <Tag props...>content</Tag>
    # 要求内部 content 不包含同类型标签 (即匹配最内层)
    paired_pattern = re.compile(
        rf"<({all_components})(?:\s[^>]*)?>\s*(.*?)\s*</\1>",
        re.DOTALL,
    )

    max_iterations = 20
    for _ in range(max_iterations):
        new_text = paired_pattern.sub(r"\2", text)
        if new_text == text:
            break
        text = new_text

    # 自闭合标签: <Tag ... />
    self_closing_pattern = re.compile(
        rf"<({all_components})(?:\s[^>]*)?\s*/\s*>",
    )
    text = self_closing_pattern.sub("", text)

    return text


def strip_html_tags(text: str) -> str:
    """剥离 HTML 标签, 保留内部文字。

    同时处理:
    - `<tag>` / `</tag>` / `<tag ...>`
    - `<tag />` 自闭合
    - HTML 注释 `<!-- ... -->`

    Args:
        text: 含 HTML 的文本。

    Returns:
        剥离后的纯文本。
    """
    # 注释
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

    # 所有 HTML 标签 (配对或自闭合)
    text = re.sub(r"</?[a-zA-Z][a-zA-Z0-9]*(?:\s[^>]*)?\s*/?>", "", text)

    return text


def clean_markdown(text: str, keep_headers: bool = True) -> str:
    """清理 Markdown 标记, 使文本更适合 embedding。

    - 图片：完全移除 `![alt](url)`
    - 链接：保留文字 `[text](url)` → `text`
    - 粗体/斜体：保留文字
    - 行内代码：保留代码文字
    - 代码块：可选保留内容
    - 标题：可选保留 `#` 标记

    Args:
        text: Markdown 文本。
        keep_headers: 是否保留 # 标题前缀。

    Returns:
        清洗后的文本。
    """
    # 图片
    text = MD_IMAGE_PATTERN.sub("", text)

    # 链接 → 保留链接文字 + URL 路径
    text = MD_LINK_PATTERN.sub(r"\1 (\2)", text)

    # 粗体/斜体: **text** / *text* → text
    # 注意: 只处理 * 包裹的强调, 不用 _ 因为 _ 在技术文档中频繁出现于
    # 变量名 (user_file, voice_and_tone) 和 URL 路径中, 误删风险极高。
    text = re.sub(r"\*{1,2}([^*\n]+?)\*{1,2}", r"\1", text)

    # 下划线斜体: 仅匹配空白符包围的 _word_ 模式,
    # 避免误伤 snake_case 标识符 (user_file 等)。
    text = re.sub(r"(?<=\s)_([^_\s]+?)_(?=\s|$|[.,!?;:，。！？；：)])", r"\1", text)
    # 行首的 _word_  (没有前置空白, 单独处理)
    text = re.sub(r"^_([^_\s]+?)_(?=\s|$|[.,!?;:，。！？；：)])", r"\1", text, flags=re.MULTILINE)

    # 删除线 ~~text~~ → text
    text = re.sub(r"~~([^~]+?)~~", r"\1", text)

    # 行内代码 `code` → code
    text = re.sub(r"`([^`]+?)`", r"\1", text)

    # 代码块围栏标记: ```language / ``` → 移除标记, 保留块内内容
    text = re.sub(r"^```.*$", "", text, flags=re.MULTILINE)

    # 清理残留的孤立反引号 (紧贴行首或行尾)
    text = re.sub(r"^`+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*`+$", "", text, flags=re.MULTILINE)

    # 如果保留标题, 不变; 否则去掉 #
    if not keep_headers:
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    return text


def normalize_whitespace(
    text: str,
    min_line_length: int = 1,
    max_consecutive_newlines: int = 2,
) -> str:
    """规范化空白字符。

    - 合并连续空行 (保留最多 N 个)
    - 移除过短的无意义行
    - 统一行首尾空白
    - 规范化中文全角空格

    Args:
        text: 待规范文本。
        min_line_length: 少于该字符数的行会被删除。
        max_consecutive_newlines: 最多允许的连续换行数。

    Returns:
        规范化后的文本。
    """
    # 合并连续空行
    pattern = r"\n{" + str(max_consecutive_newlines + 1) + r",}"
    text = re.sub(pattern, "\n" * max_consecutive_newlines, text)

    # 逐行处理
    lines = text.split("\n")
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()

        # 保留空行 (作为段落分隔)
        if not stripped:
            cleaned.append("")
            continue

        # 过滤过短的无意义行 (但不是标题)
        if len(stripped) < min_line_length and not stripped.startswith("#"):
            continue

        # 规范化中文全角空格 → 半角
        stripped = stripped.replace("　", " ")

        cleaned.append(stripped)

    # 重新组合, 去除首尾空行
    result = "\n".join(cleaned).strip()

    return result


def normalize_chinese_punctuation(text: str) -> str:
    """规范化中文标点 — 统一全角/半角混用。

    - 中文上下文中的英文逗号 → 中文逗号 (可选)
    - 清理多余空格 (中文字间不应有无意义空格)

    Args:
        text: 含中文的文本。

    Returns:
        规范化后的文本。
    """
    # 移除中文字符之间的空格 (中文书写通常无空格)
    # "这 是 一个 例子" → "这是一个例子"
    text = re.sub(r"(?<=[一-鿿㐀-䶿])\s+(?=[一-鿿㐀-䶿])", "", text)

    # 中文标点后面多余的英文空格
    # "你好。 世界" → "你好。世界"
    text = re.sub(
        r"(?<=[。，；！？、：])\s+",
        "",
        text,
    )

    return text


# --------------------------------------------------------
# 文档分块 (LangChain 集成)
# --------------------------------------------------------


def chunk_documents(
    documents: list[CleanedDocument],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[dict[str, Any]]:
    """使用 LangChain RecursiveCharacterTextSplitter 切分文档。

    Args:
        documents: 清洗后的文档列表。
        chunk_size: 每块最大字符数。
        chunk_overlap: 块之间重叠字符数。

    Returns:
        [{"content": str, "metadata": dict}, ...]
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", ". ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )

    chunks: list[dict[str, Any]] = []
    for doc in documents:
        if not doc.content.strip():
            continue
        split_texts = splitter.split_text(doc.content)
        # 先过滤空字符串（百炼 API 不接受空内容），再计算准确的 chunk_total
        valid_texts = [t.strip() for t in split_texts if t.strip()]
        for i, chunk_text in enumerate(valid_texts):
            chunks.append({
                "content": chunk_text,
                "metadata": {
                    **doc.metadata,
                    "source": doc.source_path,
                    "chunk_index": i,
                    "chunk_total": len(valid_texts),
                },
            })
    return chunks


def to_langchain_documents(
    chunks: list[dict[str, Any]],
) -> list[Any]:
    """将 chunk dict 列表转换为 LangChain Document 对象。

    需要: pip install langchain-core

    Args:
        chunks: chunk_documents() 的输出。

    Returns:
        list[langchain_core.documents.Document]
    """
    try:
        from langchain_core.documents import Document

        return [
            Document(page_content=c["content"], metadata=c["metadata"])
            for c in chunks
            if c["content"].strip()  # 二次过滤，防止空内容进入向量库
        ]
    except ImportError:
        warnings.warn(
            "langchain-core 未安装, 返回原始 dict 列表。"
            "建议: pip install langchain-core",
            RuntimeWarning,
        )
        return chunks


# --------------------------------------------------------
# 主流水线
# --------------------------------------------------------


def clean_text(text: str, config: CleanConfig | None = None) -> str:
    """对单篇文档文本执行完整清洗流水线。

    Args:
        text: 原始 MDX 文本。
        config: 清洗配置, 为 None 则使用默认值。

    Returns:
        清洗后的纯文本。
    """
    if config is None:
        config = CleanConfig()

    # Step 1: 移除翻译声明
    if config.remove_translation_notice:
        text = remove_translation_notice(text)

    # Step 2: 提取 MDX 组件关键属性 (Card title/href, Step title, Frame caption)
    text = _preserve_component_attrs(text)

    # Step 3: 剥离 MDX 组件标签
    text = strip_mdx_tags(text)

    # Step 4: 剥离 HTML 标签
    text = strip_html_tags(text)

    # Step 5: 清理 Markdown 标记
    text = clean_markdown(text, keep_headers=config.keep_headers)

    # Step 6: 规范化空白
    text = normalize_whitespace(
        text,
        min_line_length=config.min_line_length,
        max_consecutive_newlines=config.max_consecutive_newlines,
    )

    # Step 7: 中文标点规范化
    text = normalize_chinese_punctuation(text)

    return text.strip()


# --------------------------------------------------------
# OpenAPI JSON 解析
# --------------------------------------------------------


def _parse_openapi_json(file_path: Path) -> str:
    """将 OpenAPI 规范 JSON 文件转换为可检索的文本。

    提取: API 标题、描述、每个端点的 HTTP 方法、路径、摘要、描述、
          参数名/类型/说明、请求体 schema 属性。

    Args:
        file_path: OpenAPI JSON 文件路径。

    Returns:
        格式化的纯文本，适合 embedding。
    """
    import json

    with open(file_path, encoding="utf-8") as f:
        spec = json.load(f)

    lines: list[str] = []
    info = spec.get("info", {})
    title = info.get("title", file_path.stem)
    description = info.get("description", "")

    lines.append(f"## {title}")
    if description:
        lines.append(description)

    paths = spec.get("paths", {})
    if not paths:
        return "\n".join(lines)

    lines.append("")
    lines.append("## API 端点列表")
    lines.append("")

    for path_url, methods in paths.items():
        if not isinstance(methods, dict):
            continue
        for method, detail in methods.items():
            if not isinstance(detail, dict):
                continue
            method_upper = method.upper()
            summary = detail.get("summary", "")
            desc = detail.get("description", "")
            tags = detail.get("tags", [])

            # 端点标题行
            tag_str = f" [{', '.join(tags)}]" if tags else ""
            lines.append(f"### {method_upper} {path_url}{tag_str}")
            if summary:
                lines.append(f"**{summary}**")
            if desc:
                lines.append(desc)

            # 参数
            params = detail.get("parameters", [])
            if params:
                lines.append("")
                lines.append("参数:")
                for p in params:
                    p_name = p.get("name", "?")
                    p_in = p.get("in", "")
                    p_required = " (必填)" if p.get("required") else ""
                    p_desc = p.get("description", "")
                    p_type = p.get("schema", {}).get("type", "") if "schema" in p else ""
                    type_str = f" [{p_type}]" if p_type else ""
                    lines.append(f"  - `{p_name}` ({p_in}){p_required}{type_str}: {p_desc}")

            # 请求体 schema 属性
            req_body = detail.get("requestBody", {})
            if req_body and "content" in req_body:
                for content_type, content_spec in req_body["content"].items():
                    schema = content_spec.get("schema", {})
                    props = schema.get("properties", {})
                    required_fields = schema.get("required", [])
                    if props:
                        lines.append("")
                        lines.append(f"请求体 ({content_type}):")
                        for prop_name, prop_spec in props.items():
                            prop_type = prop_spec.get("type", "")
                            prop_desc = prop_spec.get("description", "")
                            req_mark = " (必填)" if prop_name in required_fields else ""
                            lines.append(f"  - `{prop_name}` [{prop_type}]{req_mark}: {prop_desc}")

            # 响应状态码
            responses = detail.get("responses", {})
            if responses:
                resp_codes = []
                for code, resp_spec in responses.items():
                    resp_desc = resp_spec.get("description", "") if isinstance(resp_spec, dict) else ""
                    resp_codes.append(f"`{code}` {resp_desc}")
                if resp_codes:
                    lines.append("")
                    lines.append(f"响应: {', '.join(resp_codes)}")

            lines.append("")

    return "\n".join(lines)


def clean_file(file_path: str | Path, config: CleanConfig | None = None) -> CleanedDocument:
    """读取并清洗单个文件。

    Args:
        file_path: 文件路径。
        config: 清洗配置。

    Returns:
        CleanedDocument 对象。
    """
    file_path = Path(file_path)
    raw = file_path.read_text(encoding="utf-8")

    # JSON 文件走 OpenAPI 解析通道
    if file_path.suffix.lower() == ".json":
        cleaned_body = _parse_openapi_json(file_path)
        metadata: dict[str, Any] = {
            "file_name": file_path.name,
            "file_stem": file_path.stem,
            "relative_path": str(file_path),
            "format": "openapi_json",
        }
        if config and config.source_root:
            try:
                metadata["relative_path"] = str(
                    file_path.relative_to(config.source_root)
                )
            except ValueError:
                pass
        return CleanedDocument(
            content=cleaned_body,
            metadata=metadata,
            source_path=str(file_path),
        )

    # 提取 frontmatter
    metadata, body = extract_frontmatter(raw)

    # 清洗正文
    cleaned_body = clean_text(body, config)

    # 注入文件元信息
    metadata["file_name"] = file_path.name
    metadata["file_stem"] = file_path.stem
    metadata["relative_path"] = str(file_path)

    if config and config.source_root:
        try:
            metadata["relative_path"] = str(
                file_path.relative_to(config.source_root)
            )
        except ValueError:
            pass

    return CleanedDocument(
        content=cleaned_body,
        metadata=metadata,
        source_path=str(file_path),
    )


def clean_directory(
    root_dir: str | Path,
    config: CleanConfig | None = None,
) -> list[CleanedDocument]:
    """清洗整个目录下的所有文档。

    Args:
        root_dir: 文档根目录。
        config: 清洗配置。

    Returns:
        CleanedDocument 列表。
    """
    if config is None:
        config = CleanConfig()

    config.source_root = str(root_dir)
    files = discover_files(str(root_dir), config.extensions)
    results: list[CleanedDocument] = []

    for fp in files:
        try:
            doc = clean_file(fp, config)
            if doc.content.strip():
                results.append(doc)
        except Exception as exc:
            warnings.warn(f"处理文件失败: {fp} —— {exc}", RuntimeWarning)

    return results


# --------------------------------------------------------
# 便捷入口
# --------------------------------------------------------


def build_rag_documents(
    docs_dir: str | Path,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    as_langchain: bool = True,
) -> list[Any]:
    """一站式入口: 目录 → 清洗 → 分块 → LangChain Document。

    Args:
        docs_dir: Dify 中文文档根目录。
        chunk_size: 分块大小。
        chunk_overlap: 重叠大小。
        as_langchain: 是否返回 LangChain Document 对象。

    Returns:
        LangChain Document 列表, 或 dict 列表。
    """
    config = CleanConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    print(f"发现 {len(discover_files(str(docs_dir)))} 个文件")
    documents = clean_directory(docs_dir, config)
    print(f"成功清洗 {len(documents)} 篇文档")

    chunks = chunk_documents(documents, chunk_size, chunk_overlap)
    print(f"生成 {len(chunks)} 个文本块")

    if as_langchain:
        chunks = to_langchain_documents(chunks)
        print(f"输出 {len(chunks)} 个 LangChain Document")

    return chunks


# --------------------------------------------------------
# CLI
# --------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python text_cleaner.py <docs_dir> [--json out.json] [--dry]")
        sys.exit(1)

    docs_dir = sys.argv[1]
    out_json = None
    dry_run = False

    for arg in sys.argv[2:]:
        if arg == "--dry":
            dry_run = True
        elif arg.startswith("--json") and "=" in arg:
            out_json = arg.split("=", 1)[1]
        elif arg == "--json" or arg == "-o":
            idx = sys.argv.index(arg)
            if idx + 1 < len(sys.argv):
                out_json = sys.argv[idx + 1]

    chunks = build_rag_documents(docs_dir, as_langchain=False)

    if dry_run:
        # 打印前 3 条样本
        for i, c in enumerate(chunks[:3]):
            print(f"\n{'='*60}")
            print(f"Chunk #{i} | source: {c['metadata'].get('source', '?')}")
            print(f"{'='*60}")
            print(c["content"][:500])

    if out_json:
        out_path = Path(out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(chunks, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"\n已输出至: {out_json}")
