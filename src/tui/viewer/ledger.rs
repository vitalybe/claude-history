use super::markdown::StyledLine;
use super::{LineStyle, NAME_WIDTH, RenderedLine, TIMESTAMP_WIDTH, ToolOutputId, th};

/// Render ledger block with styled markdown lines
pub(super) fn render_ledger_block_styled(
    lines: &mut Vec<RenderedLine>,
    name: &str,
    color: (u8, u8, u8),
    bold: bool,
    styled_lines: Vec<StyledLine>,
    timestamp: Option<&str>,
) {
    for (i, styled_line) in styled_lines.iter().enumerate() {
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

        // Name column (right-aligned, only on first line)
        let name_text = if i == 0 {
            format!("{:>width$}", name, width = NAME_WIDTH)
        } else {
            " ".repeat(NAME_WIDTH)
        };

        spans.push((
            name_text,
            LineStyle {
                fg: Some(color),
                bold,
                dimmed: false,
                italic: false,
            },
        ));

        // Separator
        spans.push((
            " │ ".to_string(),
            LineStyle {
                fg: Some(th().border),
                ..Default::default()
            },
        ));

        // Content spans
        if styled_line.spans.is_empty() {
            // Empty line - just push name and separator
        } else {
            for (text, style) in &styled_line.spans {
                spans.push((text.clone(), style.clone()));
            }
        }

        lines.push(RenderedLine::new(spans));
    }

    // If no lines, still output at least the name
    if styled_lines.is_empty() {
        let mut spans = Vec::new();

        // Timestamp prefix if provided
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

        spans.push((
            format!("{:>width$}", name, width = NAME_WIDTH),
            LineStyle {
                fg: Some(color),
                bold,
                dimmed: false,
                italic: false,
            },
        ));
        spans.push((
            " │ ".to_string(),
            LineStyle {
                fg: Some(th().border),
                ..Default::default()
            },
        ));
        lines.push(RenderedLine::new(spans));
    }
}

pub(super) fn push_line(
    lines: &mut Vec<RenderedLine>,
    spans: Vec<(String, LineStyle)>,
    tool_output_id: Option<&ToolOutputId>,
    clickable: bool,
) {
    if let Some(id) = tool_output_id {
        lines.push(RenderedLine::tool_output(spans, id.clone(), clickable));
    } else {
        lines.push(RenderedLine::new(spans));
    }
}

/// Render a truncation indicator line like "(N more lines...)"
pub(super) fn render_truncation_indicator(
    lines: &mut Vec<RenderedLine>,
    remaining: usize,
    dimmed: bool,
    show_timing: bool,
    tool_output_id: Option<&ToolOutputId>,
) {
    let mut spans = Vec::new();

    if show_timing {
        spans.push((" ".repeat(TIMESTAMP_WIDTH), LineStyle::default()));
    }

    spans.push((" ".repeat(NAME_WIDTH), LineStyle::default()));
    spans.push((
        " │ ".to_string(),
        LineStyle {
            fg: Some(th().border),
            dimmed,
            ..Default::default()
        },
    ));
    spans.push((
        format!("({} more lines...)", remaining),
        LineStyle {
            dimmed: true,
            ..Default::default()
        },
    ));

    push_line(lines, spans, tool_output_id, tool_output_id.is_some());
}

/// Render ledger block with styled markdown lines (dimmed for subagents)
pub(super) fn render_ledger_block_styled_dimmed(
    lines: &mut Vec<RenderedLine>,
    name: &str,
    color: (u8, u8, u8),
    styled_lines: Vec<StyledLine>,
    show_timing: bool,
) {
    for (i, styled_line) in styled_lines.iter().enumerate() {
        let mut spans = Vec::new();

        if show_timing {
            spans.push((" ".repeat(TIMESTAMP_WIDTH), LineStyle::default()));
        }

        let name_text = if i == 0 {
            format!("{:>width$}", name, width = NAME_WIDTH)
        } else {
            " ".repeat(NAME_WIDTH)
        };

        spans.push((
            name_text,
            LineStyle {
                fg: Some(color),
                bold: false,
                dimmed: true,
                italic: false,
            },
        ));

        spans.push((
            " │ ".to_string(),
            LineStyle {
                fg: Some(th().border),
                dimmed: true,
                ..Default::default()
            },
        ));

        for (text, mut style) in styled_line.spans.iter().cloned() {
            style.dimmed = true;
            spans.push((text, style));
        }

        lines.push(RenderedLine::new(spans));
    }

    if styled_lines.is_empty() {
        let mut spans = Vec::new();
        if show_timing {
            spans.push((" ".repeat(TIMESTAMP_WIDTH), LineStyle::default()));
        }
        spans.push((
            format!("{:>width$}", name, width = NAME_WIDTH),
            LineStyle {
                fg: Some(color),
                bold: false,
                dimmed: true,
                italic: false,
            },
        ));
        spans.push((
            " │ ".to_string(),
            LineStyle {
                fg: Some(th().border),
                dimmed: true,
                ..Default::default()
            },
        ));
        lines.push(RenderedLine::new(spans));
    }
}

/// Render ledger block with plain text (dimmed for subagents)
pub(super) fn render_ledger_block_plain_dimmed(
    lines: &mut Vec<RenderedLine>,
    name: &str,
    color: (u8, u8, u8),
    text: &str,
    show_timing: bool,
) {
    for (i, line_text) in text.lines().enumerate() {
        let mut spans = Vec::new();

        if show_timing {
            spans.push((" ".repeat(TIMESTAMP_WIDTH), LineStyle::default()));
        }

        let name_text = if i == 0 {
            format!("{:>width$}", name, width = NAME_WIDTH)
        } else {
            " ".repeat(NAME_WIDTH)
        };

        spans.push((
            name_text,
            LineStyle {
                fg: Some(color),
                bold: false,
                dimmed: true,
                italic: false,
            },
        ));

        spans.push((
            " │ ".to_string(),
            LineStyle {
                fg: Some(th().border),
                dimmed: true,
                ..Default::default()
            },
        ));

        spans.push((
            line_text.to_string(),
            LineStyle {
                dimmed: true,
                ..Default::default()
            },
        ));

        lines.push(RenderedLine::new(spans));
    }
}

/// Render continuation lines (dimmed for subagents)
pub(super) fn render_continuation_dimmed(
    lines: &mut Vec<RenderedLine>,
    text: &str,
    show_timing: bool,
    tool_output_id: Option<&ToolOutputId>,
) {
    for line_text in text.lines() {
        let mut spans = Vec::new();

        if show_timing {
            spans.push((" ".repeat(TIMESTAMP_WIDTH), LineStyle::default()));
        }

        spans.push((
            " ".repeat(NAME_WIDTH),
            LineStyle {
                dimmed: true,
                ..Default::default()
            },
        ));
        spans.push((
            " │ ".to_string(),
            LineStyle {
                fg: Some(th().border),
                dimmed: true,
                ..Default::default()
            },
        ));
        spans.push((
            line_text.to_string(),
            LineStyle {
                dimmed: true,
                ..Default::default()
            },
        ));

        push_line(lines, spans, tool_output_id, tool_output_id.is_some());
    }
}
