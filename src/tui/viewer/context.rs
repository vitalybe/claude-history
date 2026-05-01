//! Render-context types shared across viewer entry/summary/tool rendering.
//!
//! These small helpers replace the hand-written "consume timestamp or
//! pad" branches and the repeated `parent_id.map(subagent_label)` calls
//! that previously lived inline in `entry.rs` and `summary.rs`.

use std::borrow::Cow;

use super::entry::subagent_label;
use super::ledger::{
    render_continuation_dimmed, render_ledger_block_plain_dimmed, render_truncation_indicator,
};
use super::{
    RenderOptions, RenderedLine, TRUNCATED_RESULT_LINES, ToolDisplayMode, ToolOutputId, th,
};

/// Padding placeholder used when timing is enabled but no timestamp is
/// available for a row. The ledger writer ignores its contents and
/// renders `TIMESTAMP_WIDTH` spaces; the value only acts as a `Some`
/// marker that "the timing column is present."
const TIMING_PAD: &str = "     ";

/// Per-block timing cursor.
///
/// Top-level entries hand the first row their timestamp once; later
/// rows align under it with blank padding when timing is enabled, or
/// emit no timing slot at all when it is disabled. `RowTiming` owns
/// that state so callers no longer reproduce the four-armed consume-or-
/// pad branch at every row site.
pub(super) struct RowTiming<'a> {
    show_timing: bool,
    pending: Option<&'a str>,
    consumed: bool,
}

impl<'a> RowTiming<'a> {
    pub(super) fn new(show_timing: bool, timestamp: Option<&'a str>) -> Self {
        Self {
            show_timing,
            pending: if show_timing { timestamp } else { None },
            consumed: false,
        }
    }

    /// Slot for a row that may carry the first-row timestamp. The first
    /// call returns the entry's original timestamp slot — `Some(ts)`
    /// when present, or `None` when timing is disabled or the timestamp
    /// is missing/invalid (which renders without a timing column for
    /// that block, matching pre-refactor behavior). Subsequent calls
    /// return the alignment pad when timing is enabled or `None` when
    /// disabled.
    pub(super) fn consume(&mut self) -> Option<&'a str> {
        if !self.consumed {
            self.consumed = true;
            return self.pending.take();
        }
        self.pad()
    }

    /// Slot for a `render_ledger_block_styled`-style text block.
    ///
    /// Returns `Some(ts)` once for the first text block and `None`
    /// thereafter — even when timing is enabled. This mirrors the
    /// pre-refactor behavior of multi-text-block messages, where every
    /// text block past the first rendered with the timing column
    /// entirely disabled.
    pub(super) fn take_once(&mut self) -> Option<&'a str> {
        if !self.consumed {
            self.consumed = true;
            return self.pending.take();
        }
        None
    }

    /// Padding-only slot — never marks the timestamp as consumed and
    /// never returns the original timestamp. Use this for nested rows
    /// whose entry already had its timestamp (or lack thereof) handled
    /// by an earlier block.
    pub(super) fn pad(&self) -> Option<&'a str> {
        if self.show_timing {
            Some(TIMING_PAD)
        } else {
            None
        }
    }

    pub(super) fn show_timing(&self) -> bool {
        self.show_timing
    }
}

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
) {
    let truncated_mode = options.tool_display == ToolDisplayMode::Truncated;
    if truncated_mode && !expanded {
        let content_lines: Vec<&str> = content_str.lines().collect();
        let total = content_lines.len();
        if total > TRUNCATED_RESULT_LINES {
            let truncated = content_lines[..TRUNCATED_RESULT_LINES].join("\n");
            render_continuation_dimmed(lines, &truncated, options.show_timing, Some(output_id));
            render_truncation_indicator(
                lines,
                total - TRUNCATED_RESULT_LINES,
                true,
                options.show_timing,
                Some(output_id),
            );
        } else {
            render_continuation_dimmed(lines, content_str, options.show_timing, None);
        }
    } else {
        let id = truncated_mode.then_some(output_id);
        render_continuation_dimmed(lines, content_str, options.show_timing, id);
    }
}

/// Render the "  ↳ Tool │ <Result>" header that introduces a dimmed
/// subagent tool result block.
pub(super) fn render_subagent_tool_result_header(lines: &mut Vec<RenderedLine>, show_timing: bool) {
    render_ledger_block_plain_dimmed(lines, "  ↳ Tool", th().accent_dim, "<Result>", show_timing);
}
