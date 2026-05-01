use crate::tool_format;

use super::ledger::{push_line, render_truncation_indicator};
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
/// Render a formatted tool call with proper styling
#[allow(clippy::too_many_arguments)]
pub(super) fn render_tool_call(
    lines: &mut Vec<RenderedLine>,
    name: &str,
    input: &serde_json::Value,
    label: &str,
    label_color: (u8, u8, u8),
    dimmed: bool,
    content_width: usize,
    timestamp: Option<&str>,
    tool_display: ToolDisplayMode,
    tool_output_id: &ToolOutputId,
    expanded: bool,
) {
    let formatted = tool_format::format_tool_call(name, input, content_width);

    let mut spans = Vec::new();

    // Timestamp prefix (only on first line if provided)
    if let Some(ts) = timestamp {
        spans.push((
            format!(" {} ", ts),
            LineStyle {
                fg: Some((140, 140, 140)),
                dimmed: false,
                bold: false,
                italic: false,
            },
        ));
    }

    // Name column
    spans.push((
        format!("{:>width$}", label, width = NAME_WIDTH),
        LineStyle {
            fg: Some(label_color),
            bold: false,
            dimmed,
            italic: false,
        },
    ));

    // Separator
    spans.push((
        " │ ".to_string(),
        LineStyle {
            fg: Some(th().border),
            dimmed,
            ..Default::default()
        },
    ));

    // Print the header in subtle gray
    spans.push((
        formatted.header.clone(),
        LineStyle {
            fg: Some(th().tool_text),
            dimmed,
            ..Default::default()
        },
    ));

    push_line(lines, spans, Some(tool_output_id), false);

    // Render the body if present, with empty line separator
    if let Some(body) = formatted.body {
        let show_timing = timestamp.is_some();

        // Empty line between header and body
        let mut empty_spans = Vec::new();
        if show_timing {
            empty_spans.push((" ".repeat(TIMESTAMP_WIDTH), LineStyle::default()));
        }
        empty_spans.push((" ".repeat(NAME_WIDTH), LineStyle::default()));
        empty_spans.push((
            " │ ".to_string(),
            LineStyle {
                fg: Some(th().border),
                dimmed,
                ..Default::default()
            },
        ));
        push_line(lines, empty_spans, Some(tool_output_id), false);

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
        let mut spans = Vec::new();

        // Timing alignment padding (if timing is enabled)
        if show_timing {
            spans.push((" ".repeat(TIMESTAMP_WIDTH), LineStyle::default()));
        }

        // Empty name column
        spans.push((" ".repeat(NAME_WIDTH), LineStyle::default()));

        // Separator
        spans.push((
            " │ ".to_string(),
            LineStyle {
                fg: Some(th().border),
                dimmed,
                ..Default::default()
            },
        ));

        // Content with diff coloring
        if line.starts_with("+ ") {
            spans.push((
                line.to_string(),
                LineStyle {
                    fg: Some(th().diff_add),
                    dimmed,
                    ..Default::default()
                },
            ));
        } else if line.starts_with("- ") {
            spans.push((
                line.to_string(),
                LineStyle {
                    fg: Some(th().diff_remove),
                    dimmed,
                    ..Default::default()
                },
            ));
        } else {
            spans.push((
                line.to_string(),
                LineStyle {
                    dimmed: true,
                    ..Default::default()
                },
            ));
        }

        push_line(lines, spans, tool_output_id, clickable);
    }
}

/// Render tool result with arrow indicator and markdown
pub(super) fn render_tool_result(
    lines: &mut Vec<RenderedLine>,
    text: &str,
    content_width: usize,
    timestamp: Option<&str>,
    tool_display: ToolDisplayMode,
    tool_output_id: &ToolOutputId,
    expanded: bool,
) {
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
        let mut spans = Vec::new();

        // Timestamp prefix (only on first line if provided)
        if i == 0 {
            if let Some(ts) = timestamp {
                spans.push((
                    format!(" {} ", ts),
                    LineStyle {
                        fg: Some((140, 140, 140)),
                        dimmed: false,
                        bold: false,
                        italic: false,
                    },
                ));
            }
        } else if timestamp.is_some() {
            // Pad continuation lines to align with timestamped first line
            spans.push((" ".repeat(TIMESTAMP_WIDTH), LineStyle::default()));
        }

        // First line gets the label, rest are empty
        if i == 0 {
            spans.push((
                format!("{:>width$}", "↳ Result", width = NAME_WIDTH),
                LineStyle {
                    fg: Some(th().tool_text),
                    ..Default::default()
                },
            ));
        } else {
            spans.push((" ".repeat(NAME_WIDTH), LineStyle::default()));
        }

        // Separator
        spans.push((
            " │ ".to_string(),
            LineStyle {
                fg: Some(th().border),
                ..Default::default()
            },
        ));

        // Content spans from markdown rendering
        for (text, style) in &styled_line.spans {
            spans.push((text.clone(), style.clone()));
        }

        let clickable = tool_display == ToolDisplayMode::Truncated && (expanded || limit < total);
        let id = clickable.then_some(tool_output_id);
        push_line(lines, spans, id, clickable);
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
