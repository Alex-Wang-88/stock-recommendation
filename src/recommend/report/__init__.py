"""输出层：候选清单 + 因子贡献 + 风控提示 + 数据时点。"""

from .render import (
    DISPLAY_COLUMNS,
    PCT_COLUMNS,
    markdown_table,
    render_markdown,
    to_display,
)

__all__ = ["DISPLAY_COLUMNS", "PCT_COLUMNS", "markdown_table", "render_markdown", "to_display"]
