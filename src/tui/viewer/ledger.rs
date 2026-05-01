use super::markdown::StyledLine;
use super::{LineStyle, NAME_WIDTH, RenderedLine, TIMESTAMP_WIDTH, ToolOutputId, th};

/// The timestamp column for a single ledger row.
pub(super) enum TimestampCol<'a> {
    /// Timing disabled — no column at all.
    Disabled,
    /// Timing enabled, but this row has no timestamp text (continuation).
    Pad,
    /// Timing enabled, and this row carries the timestamp text (e.g. "12:34").
    Stamp(&'a str),
}

/// The name column for a single ledger row.
pub(super) enum NameCol<'a> {
    /// First row of a block: a right-aligned label.
    Label {
        text: &'a str,
        color: (u8, u8, u8),
        bold: bool,
        dimmed: bool,
    },
    /// Continuation row: blank name, default style.
    BlankPlain,
    /// Continuation row: blank name, `dimmed: true` (no fg).
    BlankDim,
    /// Continuation row: blank name carrying the label color, `dimmed: true`.
    BlankColoredDim { color: (u8, u8, u8) },
}

/// Description of one ledger row's structural columns.
pub(super) struct LedgerRow<'a> {
    pub timestamp: TimestampCol<'a>,
    pub name: NameCol<'a>,
    /// Whether the " │ " separator span renders dimmed.
    pub separator_dimmed: bool,
    /// Optional tool-output id attached to the resulting `RenderedLine`.
    pub tool_output_id: Option<&'a ToolOutputId>,
    pub clickable: bool,
}

/// Low-level ledger writer: assembles the timestamp / name / separator
/// columns according to `row` and appends `content` spans after them.
///
/// All ledger rows in the viewer go through this single entry point so
/// that timestamp width, name alignment, separator styling, and tool
/// output id / clickable propagation stay consistent.
pub(super) fn push_row(
    lines: &mut Vec<RenderedLine>,
    row: LedgerRow<'_>,
    content: Vec<(String, LineStyle)>,
) {
    let mut spans = Vec::with_capacity(3 + content.len());

    match row.timestamp {
        TimestampCol::Disabled => {}
        TimestampCol::Pad => {
            spans.push((" ".repeat(TIMESTAMP_WIDTH), LineStyle::default()));
        }
        TimestampCol::Stamp(ts) => {
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
    }

    match row.name {
        NameCol::Label {
            text,
            color,
            bold,
            dimmed,
        } => {
            spans.push((
                format!("{:>width$}", text, width = NAME_WIDTH),
                LineStyle {
                    fg: Some(color),
                    bold,
                    dimmed,
                    italic: false,
                },
            ));
        }
        NameCol::BlankPlain => {
            spans.push((" ".repeat(NAME_WIDTH), LineStyle::default()));
        }
        NameCol::BlankDim => {
            spans.push((
                " ".repeat(NAME_WIDTH),
                LineStyle {
                    dimmed: true,
                    ..Default::default()
                },
            ));
        }
        NameCol::BlankColoredDim { color } => {
            spans.push((
                " ".repeat(NAME_WIDTH),
                LineStyle {
                    fg: Some(color),
                    dimmed: true,
                    ..Default::default()
                },
            ));
        }
    }

    spans.push((
        " │ ".to_string(),
        LineStyle {
            fg: Some(th().border),
            dimmed: row.separator_dimmed,
            ..Default::default()
        },
    ));

    spans.extend(content);

    let line = match row.tool_output_id {
        Some(id) => RenderedLine::tool_output(spans, id.clone(), row.clickable),
        None => RenderedLine::new(spans),
    };
    lines.push(line);
}

/// Timestamp slot for the `i`-th row of a block whose first row carries
/// `timestamp` (or `None` when timing is disabled or no timestamp).
fn block_timestamp<'a>(timestamp: Option<&'a str>, i: usize) -> TimestampCol<'a> {
    match (i, timestamp) {
        (0, Some(ts)) => TimestampCol::Stamp(ts),
        (_, Some(_)) => TimestampCol::Pad,
        (_, None) => TimestampCol::Disabled,
    }
}

fn show_timing_col(show_timing: bool) -> TimestampCol<'static> {
    if show_timing {
        TimestampCol::Pad
    } else {
        TimestampCol::Disabled
    }
}

/// Render ledger block with styled markdown lines
pub(super) fn render_ledger_block_styled(
    lines: &mut Vec<RenderedLine>,
    name: &str,
    color: (u8, u8, u8),
    bold: bool,
    styled_lines: Vec<StyledLine>,
    timestamp: Option<&str>,
) {
    if styled_lines.is_empty() {
        push_row(
            lines,
            LedgerRow {
                timestamp: block_timestamp(timestamp, 0),
                name: NameCol::Label {
                    text: name,
                    color,
                    bold,
                    dimmed: false,
                },
                separator_dimmed: false,
                tool_output_id: None,
                clickable: false,
            },
            Vec::new(),
        );
        return;
    }

    for (i, styled_line) in styled_lines.iter().enumerate() {
        let name_col = if i == 0 {
            NameCol::Label {
                text: name,
                color,
                bold,
                dimmed: false,
            }
        } else {
            NameCol::BlankPlain
        };
        let content = styled_line
            .spans
            .iter()
            .map(|(t, s)| (t.clone(), s.clone()))
            .collect();
        push_row(
            lines,
            LedgerRow {
                timestamp: block_timestamp(timestamp, i),
                name: name_col,
                separator_dimmed: false,
                tool_output_id: None,
                clickable: false,
            },
            content,
        );
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
    let content = vec![(
        format!("({} more lines...)", remaining),
        LineStyle {
            dimmed: true,
            ..Default::default()
        },
    )];
    push_row(
        lines,
        LedgerRow {
            timestamp: show_timing_col(show_timing),
            name: NameCol::BlankPlain,
            separator_dimmed: dimmed,
            tool_output_id,
            clickable: tool_output_id.is_some(),
        },
        content,
    );
}

/// Render ledger block with styled markdown lines (dimmed for subagents)
pub(super) fn render_ledger_block_styled_dimmed(
    lines: &mut Vec<RenderedLine>,
    name: &str,
    color: (u8, u8, u8),
    styled_lines: Vec<StyledLine>,
    show_timing: bool,
) {
    if styled_lines.is_empty() {
        push_row(
            lines,
            LedgerRow {
                timestamp: show_timing_col(show_timing),
                name: NameCol::Label {
                    text: name,
                    color,
                    bold: false,
                    dimmed: true,
                },
                separator_dimmed: true,
                tool_output_id: None,
                clickable: false,
            },
            Vec::new(),
        );
        return;
    }

    for (i, styled_line) in styled_lines.iter().enumerate() {
        let name_col = if i == 0 {
            NameCol::Label {
                text: name,
                color,
                bold: false,
                dimmed: true,
            }
        } else {
            NameCol::BlankColoredDim { color }
        };
        let content = styled_line
            .spans
            .iter()
            .cloned()
            .map(|(text, mut style)| {
                style.dimmed = true;
                (text, style)
            })
            .collect();
        push_row(
            lines,
            LedgerRow {
                timestamp: show_timing_col(show_timing),
                name: name_col,
                separator_dimmed: true,
                tool_output_id: None,
                clickable: false,
            },
            content,
        );
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
        let name_col = if i == 0 {
            NameCol::Label {
                text: name,
                color,
                bold: false,
                dimmed: true,
            }
        } else {
            NameCol::BlankColoredDim { color }
        };
        let content = vec![(
            line_text.to_string(),
            LineStyle {
                dimmed: true,
                ..Default::default()
            },
        )];
        push_row(
            lines,
            LedgerRow {
                timestamp: show_timing_col(show_timing),
                name: name_col,
                separator_dimmed: true,
                tool_output_id: None,
                clickable: false,
            },
            content,
        );
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
        let content = vec![(
            line_text.to_string(),
            LineStyle {
                dimmed: true,
                ..Default::default()
            },
        )];
        push_row(
            lines,
            LedgerRow {
                timestamp: show_timing_col(show_timing),
                name: NameCol::BlankDim,
                separator_dimmed: true,
                tool_output_id,
                clickable: tool_output_id.is_some(),
            },
            content,
        );
    }
}
