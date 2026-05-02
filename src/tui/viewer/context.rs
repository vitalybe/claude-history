//! Render-context helpers shared across viewer entry/summary/tool rendering.

use std::borrow::Cow;

use super::entry::subagent_label;
use super::ledger::{
    render_continuation_dimmed, render_ledger_block_plain_dimmed, render_truncation_indicator,
};
use super::timing::TimingSlot;
use super::{
    RenderOptions, RenderedLine, TRUNCATED_RESULT_LINES, ToolDisplayMode, ToolOutputId, th,
};

/// Resolve the assistant-side label for the current entry: the nested
/// arrow form for subagent messages, otherwise the literal "Claude".
pub(super) fn assistant_label(parent_id: Option<&str>) -> Cow<'static, str> {
    match parent_id {
        Some(p) => Cow::Owned(subagent_label(p)),
        None => Cow::Borrowed("Claude"),
    }
}

/// Render the dimmed body of a subagent tool result.
///
/// In truncated tool-display mode this emits at most `TRUNCATED_RESULT_LINES`
/// of the result followed by a clickable "(N more lines...)" indicator;
/// otherwise it renders the full result as a continuation block. Used by
/// both the user-message subagent branch and the agent-progress user
/// branch.
pub(super) fn render_dimmed_tool_result_body(
    lines: &mut Vec<RenderedLine>,
    options: &RenderOptions,
    output_id: &ToolOutputId,
    expanded: bool,
    content_str: &str,
    timing: TimingSlot<'_>,
) {
    let truncated_mode = options.tool_display == ToolDisplayMode::Truncated;
    if truncated_mode && !expanded {
        let content_lines: Vec<&str> = content_str.lines().collect();
        let total = content_lines.len();
        if total > TRUNCATED_RESULT_LINES {
            let truncated = content_lines[..TRUNCATED_RESULT_LINES].join("\n");
            render_continuation_dimmed(lines, &truncated, timing, Some(output_id));
            render_truncation_indicator(
                lines,
                total - TRUNCATED_RESULT_LINES,
                true,
                timing,
                Some(output_id),
            );
        } else {
            render_continuation_dimmed(lines, content_str, timing, None);
        }
    } else {
        let id = truncated_mode.then_some(output_id);
        render_continuation_dimmed(lines, content_str, timing, id);
    }
}

/// Render the "  ↳ Tool │ <Result>" header that introduces a dimmed
/// subagent tool result block.
pub(super) fn render_subagent_tool_result_header(
    lines: &mut Vec<RenderedLine>,
    timing: TimingSlot<'_>,
) {
    render_ledger_block_plain_dimmed(lines, "  ↳ Tool", th().accent_dim, "<Result>", timing);
}
