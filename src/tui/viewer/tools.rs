use crate::tool_format;

use super::ledger::{LedgerRow, NameCol, TimestampCol, push_row, render_truncation_indicator};
use super::markdown::render_markdown_to_lines;
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ToolOutputKind {
    ToolCall,
    ToolResult,
}

impl ToolOutputKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::ToolCall => "call",
            Self::ToolResult => "result",
        }
    }
}
pub(super) fn make_tool_output_id(
    entry_index: usize,
    parent_id: Option<&str>,
    block_index: usize,
    kind: ToolOutputKind,
    raw_id: Option<&str>,
) -> ToolOutputId {
    let parent = parent_id.unwrap_or("top");
    let raw = raw_id.unwrap_or("none");
    ToolOutputId(format!(
        "entry:{entry_index}:parent:{parent}:block:{block_index}:kind:{}:id:{raw}",
        kind.as_str()
    ))
}

pub(super) fn make_tool_summary_output_id(
    entry_index: usize,
    parent_id: Option<&str>,
) -> ToolOutputId {
    let parent = parent_id.unwrap_or("top");
    ToolOutputId(format!("entry:{entry_index}:parent:{parent}:kind:summary"))
}
/// Extract text content from tool result for markdown rendering.
/// Returns Some(text) if content is a string or array of text blocks.
/// Returns None for JSON structures that should be pretty-printed instead.
pub(super) fn extract_tool_result_text(content: Option<&serde_json::Value>) -> Option<String> {
    match content {
        Some(serde_json::Value::String(s)) => Some(s.clone()),
        Some(serde_json::Value::Array(arr)) => {
            // Handle array of content blocks (e.g., [{type: "text", text: "..."}])
            let texts: Vec<&str> = arr
                .iter()
                .filter_map(|item| item.get("text").and_then(|t| t.as_str()))
                .collect();
            if !texts.is_empty() {
                Some(texts.join("\n\n"))
            } else {
                None // Array without text blocks - render as JSON
            }
        }
        _ => None, // Objects, null, etc. - render as JSON
    }
}

/// Format tool result content to a string for display (non-text content)
pub(super) fn format_tool_result_content(content: Option<&serde_json::Value>) -> String {
    match content {
        Some(value) => {
            if let Ok(formatted) = serde_json::to_string_pretty(value) {
                formatted
            } else {
                "<invalid content>".to_string()
            }
        }
        None => "<no content>".to_string(),
    }
}

/// Pick the display text for a tool result: prefer extracted text content,
/// fall back to a JSON pretty-print for objects, null, or text-less arrays.
pub(super) fn tool_result_display_text(content: Option<&serde_json::Value>) -> String {
    extract_tool_result_text(content).unwrap_or_else(|| format_tool_result_content(content))
}

/// Descriptor for one tool-call rendering pass shared by entry and
/// summary-expansion render paths.
pub(super) struct ToolCallRenderSpec<'a> {
    pub name: &'a str,
    pub input: &'a serde_json::Value,
    pub label: &'a str,
    pub label_color: (u8, u8, u8),
    pub dimmed: bool,
    pub content_width: usize,
    pub timestamp: Option<&'a str>,
    pub tool_display: ToolDisplayMode,
    pub tool_output_id: &'a ToolOutputId,
    pub expanded: bool,
}

/// Descriptor for one tool-result rendering pass shared by entry and
/// summary-expansion render paths.
pub(super) struct ToolResultRenderSpec<'a> {
    pub text: &'a str,
    pub content_width: usize,
    pub timestamp: Option<&'a str>,
    pub tool_display: ToolDisplayMode,
    pub tool_output_id: &'a ToolOutputId,
    pub expanded: bool,
}
/// Render a formatted tool call with proper styling
pub(super) fn render_tool_call(lines: &mut Vec<RenderedLine>, spec: &ToolCallRenderSpec<'_>) {
    let ToolCallRenderSpec {
        name,
        input,
        label,
        label_color,
        dimmed,
        content_width,
        timestamp,
        tool_display,
        tool_output_id,
        expanded,
    } = *spec;
    let formatted = tool_format::format_tool_call(name, input, content_width);

    let header_content = vec![(
        formatted.header.clone(),
        LineStyle {
            fg: Some(th().tool_text),
            dimmed,
            ..Default::default()
        },
    )];
    push_row(
        lines,
        LedgerRow {
            timestamp: match timestamp {
                Some(ts) => TimestampCol::Stamp(ts),
                None => TimestampCol::Disabled,
            },
            name: NameCol::Label {
                text: label,
                color: label_color,
                bold: false,
                dimmed,
            },
            separator_dimmed: dimmed,
            tool_output_id: Some(tool_output_id),
            clickable: false,
        },
        header_content,
    );

    // Render the body if present, with empty line separator
    if let Some(body) = formatted.body {
        let show_timing = timestamp.is_some();

        // Empty line between header and body
        push_row(
            lines,
            LedgerRow {
                timestamp: if show_timing {
                    TimestampCol::Pad
                } else {
                    TimestampCol::Disabled
                },
                name: NameCol::BlankPlain,
                separator_dimmed: dimmed,
                tool_output_id: Some(tool_output_id),
                clickable: false,
            },
            Vec::new(),
        );

        if tool_display == ToolDisplayMode::Truncated && !expanded {
            let body_lines: Vec<&str> = body.lines().collect();
            let total = body_lines.len();
            if total > TRUNCATED_BODY_LINES {
                let truncated = body_lines[..TRUNCATED_BODY_LINES].join("\n");
                render_tool_body(
                    lines,
                    &truncated,
                    dimmed,
                    show_timing,
                    Some(tool_output_id),
                    true,
                );
                render_truncation_indicator(
                    lines,
                    total - TRUNCATED_BODY_LINES,
                    dimmed,
                    show_timing,
                    Some(tool_output_id),
                );
            } else {
                render_tool_body(lines, &body, dimmed, show_timing, None, false);
            }
        } else {
            let id = (tool_display == ToolDisplayMode::Truncated).then_some(tool_output_id);
            render_tool_body(lines, &body, dimmed, show_timing, id, id.is_some());
        }
    }
}

/// Render tool body with diff-aware coloring
fn render_tool_body(
    lines: &mut Vec<RenderedLine>,
    text: &str,
    dimmed: bool,
    show_timing: bool,
    tool_output_id: Option<&ToolOutputId>,
    clickable: bool,
) {
    for line in text.lines() {
        let style = if line.starts_with("+ ") {
            LineStyle {
                fg: Some(th().diff_add),
                dimmed,
                ..Default::default()
            }
        } else if line.starts_with("- ") {
            LineStyle {
                fg: Some(th().diff_remove),
                dimmed,
                ..Default::default()
            }
        } else {
            LineStyle {
                dimmed: true,
                ..Default::default()
            }
        };
        push_row(
            lines,
            LedgerRow {
                timestamp: if show_timing {
                    TimestampCol::Pad
                } else {
                    TimestampCol::Disabled
                },
                name: NameCol::BlankPlain,
                separator_dimmed: dimmed,
                tool_output_id,
                clickable,
            },
            vec![(line.to_string(), style)],
        );
    }
}

/// Render tool result with arrow indicator and markdown
pub(super) fn render_tool_result(lines: &mut Vec<RenderedLine>, spec: &ToolResultRenderSpec<'_>) {
    let ToolResultRenderSpec {
        text,
        content_width,
        timestamp,
        tool_display,
        tool_output_id,
        expanded,
    } = *spec;
    // Fence plain text tool results to prevent markdown misinterpretation.
    // If the result already contains fenced code blocks, assume it's intentional markdown.
    let text = if text.contains("```") {
        text.to_string()
    } else {
        format!("```text\n{}\n```", text)
    };
    // Render markdown
    let styled_lines = render_markdown_to_lines(&text, content_width);

    let total = styled_lines.len();
    let limit = if tool_display == ToolDisplayMode::Truncated
        && !expanded
        && total > TRUNCATED_RESULT_LINES
    {
        TRUNCATED_RESULT_LINES
    } else {
        total
    };

    for (i, styled_line) in styled_lines.iter().take(limit).enumerate() {
        let timestamp_col = match (i, timestamp) {
            (0, Some(ts)) => TimestampCol::Stamp(ts),
            (_, Some(_)) => TimestampCol::Pad,
            (_, None) => TimestampCol::Disabled,
        };
        let name_col = if i == 0 {
            NameCol::Label {
                text: "↳ Result",
                color: th().tool_text,
                bold: false,
                dimmed: false,
            }
        } else {
            NameCol::BlankPlain
        };
        let content: Vec<_> = styled_line
            .spans
            .iter()
            .map(|(t, s)| (t.clone(), s.clone()))
            .collect();
        let clickable = tool_display == ToolDisplayMode::Truncated && (expanded || limit < total);
        let id = clickable.then_some(tool_output_id);
        push_row(
            lines,
            LedgerRow {
                timestamp: timestamp_col,
                name: name_col,
                separator_dimmed: false,
                tool_output_id: id,
                clickable,
            },
            content,
        );
    }

    if limit < total {
        render_truncation_indicator(
            lines,
            total - limit,
            false,
            timestamp.is_some(),
            Some(tool_output_id),
        );
    }
}
