//! Conversation viewer rendering for TUI display.
//!
//! This module renders conversation JSONL files to `Vec<RenderedLine>` for display
//! in the TUI viewer. It produces styled spans that ratatui can render directly,
//! without using ANSI escape codes.

use crate::claude::LogEntry;
use std::collections::BTreeSet;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::tui::theme::{self, Theme};

mod commands;
mod entry;
mod ledger;
mod markdown;
mod output;
mod summary;
mod tools;

pub use output::{LineStyle, RenderedLine};

use entry::{render_entry, subagent_label};
use summary::{
    PendingToolSummary, flush_tool_summary, tool_only_assistant_summary,
    user_entry_is_only_tool_results,
};
use tools::make_tool_summary_output_id;

/// Width of the focus gutter indicator (▌ + space)
pub const GUTTER_WIDTH: usize = 2;

const NAME_WIDTH: usize = 9;
/// Width of timestamp prefix when timing is enabled (space + HH:MM + space)
const TIMESTAMP_WIDTH: usize = 7;

/// Get the current theme (cached after first detection)
fn th() -> &'static Theme {
    theme::detect_theme()
}

/// Maximum body lines shown in truncated tool call mode
const TRUNCATED_BODY_LINES: usize = 3;
/// Maximum result lines shown in truncated tool result mode
const TRUNCATED_RESULT_LINES: usize = 4;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ToolOutputId(pub String);

/// Controls how tool calls and results are displayed
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ToolDisplayMode {
    #[default]
    Hidden,
    Truncated,
    Full,
}

impl ToolDisplayMode {
    /// Cycle to the next mode: Summary → Truncated → Full → Summary
    pub fn next(self) -> Self {
        match self {
            Self::Hidden => Self::Truncated,
            Self::Truncated => Self::Full,
            Self::Full => Self::Hidden,
        }
    }

    pub fn is_summary(self) -> bool {
        matches!(self, Self::Hidden)
    }

    /// Whether full or truncated tool details should be rendered
    pub fn shows_details(self) -> bool {
        !matches!(self, Self::Hidden)
    }

    /// Whether tools should be included in exported text
    pub fn is_visible(self) -> bool {
        self.shows_details()
    }

    /// Fixed-width label for the status bar (3 chars each)
    pub fn status_label(self) -> &'static str {
        match self {
            Self::Hidden => "sum",
            Self::Truncated => "trn",
            Self::Full => "all",
        }
    }
}

/// Options for rendering a conversation
pub struct RenderOptions {
    pub tool_display: ToolDisplayMode,
    pub show_thinking: bool,
    pub show_timing: bool,
    pub content_width: usize,
    pub expanded_tool_outputs: BTreeSet<ToolOutputId>,
}

/// Tracks the line range of a single message (User or Assistant entry) in the rendered output
#[derive(Clone, Debug)]
pub struct MessageRange {
    /// Index of the JSONL entry (line number in the file, 0-based, counting only parsed entries)
    pub entry_index: usize,
    /// Start line in rendered output (inclusive)
    pub start_line: usize,
    /// End line in rendered output (exclusive, excludes trailing blank)
    pub end_line: usize,
}

/// Result of rendering a conversation
pub struct RenderedConversation {
    pub lines: Vec<RenderedLine>,
    pub messages: Vec<MessageRange>,
}

/// Format an ISO 8601 timestamp to HH:MM local time
fn format_timestamp(iso_timestamp: &str) -> Option<String> {
    use chrono::{DateTime, Local};
    // Parse RFC 3339 timestamp (handles timezone offsets) and convert to local time
    DateTime::parse_from_rfc3339(iso_timestamp)
        .ok()
        .map(|dt| dt.with_timezone(&Local).format("%H:%M").to_string())
}

#[derive(Debug)]
pub struct RenderableEntry {
    pub entry_index: usize,
    entry: LogEntry,
}

pub fn parse_conversation_file(file_path: &Path) -> std::io::Result<Vec<RenderableEntry>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    let mut entries = Vec::new();
    let mut entry_index: usize = 0;

    for line_result in reader.lines() {
        let line = line_result?;
        if line.trim().is_empty() {
            continue;
        }

        if let Ok(entry) = serde_json::from_str::<LogEntry>(&line) {
            if !matches!(entry, LogEntry::FileHistorySnapshot { .. }) {
                entries.push(RenderableEntry { entry_index, entry });
            }
            entry_index += 1;
        }
    }

    Ok(entries)
}

pub fn render_parsed_conversation(
    entries: &[RenderableEntry],
    options: &RenderOptions,
) -> RenderedConversation {
    let mut lines = Vec::new();
    let mut messages = Vec::new();
    let mut pending_tool_summary: Option<PendingToolSummary> = None;

    for (parsed_idx, parsed) in entries.iter().enumerate() {
        let entry_index = parsed.entry_index;
        let entry = &parsed.entry;

        if options.tool_display.is_summary() {
            if let Some((parent_id, timestamp, summary)) =
                tool_only_assistant_summary(entry, options)
            {
                match &mut pending_tool_summary {
                    Some(pending) if pending.parent_id.as_deref() == parent_id => {
                        pending.last_parsed_idx = parsed_idx;
                        pending.summary.merge(summary);
                    }
                    _ => {
                        flush_tool_summary(
                            &mut lines,
                            &mut messages,
                            &mut pending_tool_summary,
                            entries,
                            options,
                        );
                        pending_tool_summary = Some(PendingToolSummary {
                            id: make_tool_summary_output_id(entry_index, parent_id),
                            first_entry_index: entry_index,
                            first_parsed_idx: parsed_idx,
                            last_parsed_idx: parsed_idx,
                            parent_id: parent_id.map(str::to_string),
                            timestamp: timestamp.map(str::to_string),
                            summary,
                        });
                    }
                }
                continue;
            }

            if user_entry_is_only_tool_results(entry, options) {
                if let Some(pending) = &mut pending_tool_summary {
                    pending.last_parsed_idx = parsed_idx;
                }
                continue;
            }
        }

        flush_tool_summary(
            &mut lines,
            &mut messages,
            &mut pending_tool_summary,
            entries,
            options,
        );

        let is_message = matches!(entry, LogEntry::User { .. } | LogEntry::Assistant { .. })
            || matches!(entry, LogEntry::Progress { data, .. }
                if options.show_thinking && crate::claude::parse_agent_progress(data).is_some());
        let start_line = lines.len();
        render_entry(&mut lines, entry_index, entry, options);
        let end_line = lines.len();

        if is_message && end_line > start_line {
            let effective_end =
                if end_line > 0 && lines.get(end_line - 1).is_some_and(|l| l.spans.is_empty()) {
                    end_line - 1
                } else {
                    end_line
                };
            if effective_end > start_line {
                messages.push(MessageRange {
                    entry_index,
                    start_line,
                    end_line: effective_end,
                });
            }
        }
    }

    flush_tool_summary(
        &mut lines,
        &mut messages,
        &mut pending_tool_summary,
        entries,
        options,
    );

    // Collapse consecutive empty lines into single empty lines.
    // Multiple render functions each add trailing empty lines, which can
    // result in double blanks when a tool result has empty output.
    // After dedup, remap message ranges to account for removed lines.
    let mut removed = vec![false; lines.len()];
    let mut i = 1;
    while i < lines.len() {
        if lines[i].spans.is_empty() && lines[i - 1].spans.is_empty() {
            removed[i] = true;
            i += 1;
        } else {
            i += 1;
        }
    }

    // Build index mapping: old line index -> new line index
    let mut new_index = Vec::with_capacity(lines.len());
    let mut offset = 0usize;
    for (idx, &is_removed) in removed.iter().enumerate() {
        if is_removed {
            new_index.push(idx - offset); // won't be used, but fill for completeness
            offset += 1;
        } else {
            new_index.push(idx - offset);
        }
    }
    let total_after = lines.len() - offset;

    // Remove the marked lines
    {
        let mut write = 0;
        for (read, &is_removed) in removed.iter().enumerate() {
            if !is_removed {
                if write != read {
                    lines.swap(write, read);
                }
                write += 1;
            }
        }
        lines.truncate(total_after);
    }

    // Remap message ranges
    for msg in &mut messages {
        msg.start_line = new_index[msg.start_line];
        // end_line is exclusive, so map the last included line and add 1
        if msg.end_line > 0 && msg.end_line <= new_index.len() {
            // Find the new index of the last non-removed line before end_line
            let mut last = msg.end_line - 1;
            while last > msg.start_line && removed[last] {
                last -= 1;
            }
            msg.end_line = new_index[last] + 1;
        } else if msg.end_line == new_index.len() {
            msg.end_line = total_after;
        }
        // Clamp
        msg.end_line = msg.end_line.min(total_after);
        msg.start_line = msg.start_line.min(msg.end_line);
    }

    // Remove empty ranges
    messages.retain(|m| m.start_line < m.end_line);

    RenderedConversation { lines, messages }
}

/// Render a conversation file to lines for display in the TUI viewer
pub fn render_conversation(
    file_path: &Path,
    options: &RenderOptions,
) -> std::io::Result<RenderedConversation> {
    let entries = parse_conversation_file(file_path)?;
    Ok(render_parsed_conversation(&entries, options))
}

#[cfg(test)]
mod tests {
    use super::markdown::render_markdown_to_lines;
    use super::tools::{ToolOutputKind, make_tool_output_id, render_tool_call};
    use super::*;

    /// Helper to render markdown and extract just the content text (without styling)
    fn render_to_text(input: &str, width: usize) -> String {
        let lines = render_markdown_to_lines(input, width);
        lines
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|(text, _)| text.as_str())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn test_plain_text() {
        let result = render_to_text("Hello world", 80);
        assert_eq!(result.trim(), "Hello world");
    }

    #[test]
    fn test_heading() {
        let result = render_to_text("# Heading 1", 80);
        assert!(result.contains("Heading 1"));
    }

    #[test]
    fn test_heading_with_paragraph() {
        let result = render_to_text("# Heading\n\nSome text", 80);
        let lines: Vec<&str> = result.lines().collect();
        // Should have: heading, blank, text
        assert_eq!(lines.len(), 3, "Expected 3 lines, got:\n{}", result);
        assert!(lines[0].contains("Heading"));
        assert_eq!(lines[1], "");
        assert_eq!(lines[2], "Some text");
    }

    #[test]
    fn test_paragraph_with_list() {
        let result = render_to_text("Some intro:\n\n- Item 1\n- Item 2", 80);
        let lines: Vec<&str> = result.lines().collect();
        // Should have: para, blank, item1, item2
        assert_eq!(lines.len(), 4, "Expected 4 lines, got:\n{}", result);
        assert_eq!(lines[0], "Some intro:");
        assert_eq!(lines[1], "");
        assert!(lines[2].contains("- Item 1"));
        assert!(lines[3].contains("- Item 2"));
    }

    #[test]
    fn test_numbered_list_with_bold() {
        // This is the bug case: numbered list item starting with bold text
        let result = render_to_text("1. **Task 10:** description\n2. **Task 11:** more", 80);
        let lines: Vec<&str> = result.lines().collect();
        // Should have: item1, item2 (NO blank lines between number and content)
        assert_eq!(lines.len(), 2, "Expected 2 lines, got:\n{}", result);
        assert!(
            lines[0].starts_with("1. "),
            "Line should start with '1. ': {:?}",
            lines[0]
        );
        assert!(
            lines[0].contains("Task 10"),
            "Line should contain 'Task 10': {:?}",
            lines[0]
        );
        assert!(
            lines[1].starts_with("2. "),
            "Line should start with '2. ': {:?}",
            lines[1]
        );
        assert!(
            lines[1].contains("Task 11"),
            "Line should contain 'Task 11': {:?}",
            lines[1]
        );
    }

    #[test]
    fn test_numbered_list_no_extra_blank_lines() {
        let input = "## Changes\n\n1. **First change:**\n   - details\n2. **Second change:**\n   - more details";
        let result = render_to_text(input, 80);
        let lines: Vec<&str> = result.lines().collect();

        // Verify no blank lines between "1." and "First change"
        let line1_idx = lines
            .iter()
            .position(|l| l.starts_with("1. "))
            .expect("Should find '1. '");
        assert!(
            lines[line1_idx].contains("First change"),
            "First item should be on same line as '1. '"
        );

        // Verify no blank lines between "2." and "Second change"
        let line2_idx = lines
            .iter()
            .position(|l| l.starts_with("2. "))
            .expect("Should find '2. '");
        assert!(
            lines[line2_idx].contains("Second change"),
            "Second item should be on same line as '2. '"
        );
    }

    #[test]
    fn test_consecutive_list_items_no_blanks() {
        let result = render_to_text("- First\n- Second\n- Third", 80);
        let lines: Vec<&str> = result.lines().collect();
        // Should be exactly 3 lines, no blanks between items
        assert_eq!(
            lines.len(),
            3,
            "Expected 3 lines with no blanks, got:\n{}",
            result
        );
        assert!(lines[0].contains("- First"));
        assert!(lines[1].contains("- Second"));
        assert!(lines[2].contains("- Third"));
    }

    #[test]
    fn test_nested_list() {
        let result = render_to_text("- Item 1\n  - Nested 1\n  - Nested 2\n- Item 2", 80);
        let lines: Vec<&str> = result.lines().collect();
        // Should have: item1, nested1, nested2, item2 (no extra blanks)
        assert_eq!(lines.len(), 4, "Expected 4 lines, got:\n{}", result);
        assert!(lines[0].contains("- Item 1"));
        assert!(lines[1].contains("- Nested 1"));
        assert!(lines[2].contains("- Nested 2"));
        assert!(lines[3].contains("- Item 2"));
    }

    #[test]
    fn test_code_block() {
        let result = render_to_text("Text\n\n```rust\nlet x = 1;\n```\n\nMore text", 80);
        let lines: Vec<&str> = result.lines().collect();
        // TUI strips fence markers (signaled via color instead).
        assert!(!result.contains("```"));
        assert!(result.contains("let x = 1;"));

        // Check for proper spacing
        let text_idx = lines.iter().position(|l| l == &"Text").unwrap();
        let more_idx = lines.iter().position(|l| l == &"More text").unwrap();
        assert_eq!(lines[text_idx + 1], "", "Should have blank line after Text");
        assert_eq!(
            lines[more_idx - 1],
            "",
            "Should have blank line before More text"
        );
    }

    #[test]
    fn test_block_quote() {
        let result = render_to_text("Text\n\n> Quote here", 80);
        let lines: Vec<&str> = result.lines().collect();
        // Block quote renders with quote prefix on one line, blank, then content
        // This is due to how the markdown parser handles block quotes
        assert_eq!(lines[0], "Text");
        assert_eq!(lines[1], ""); // blank before quote
        assert!(lines[2].starts_with("> "), "Should have quote prefix");
        // Content may be on same line or next line depending on parser
        let has_content =
            lines[2].contains("Quote here") || (lines.len() > 4 && lines[4].contains("Quote here"));
        assert!(has_content, "Should contain quote content");
    }

    #[test]
    fn test_horizontal_rule() {
        let result = render_to_text("Before\n\n---\n\nAfter", 80);
        let lines: Vec<&str> = result.lines().collect();
        // Should have proper spacing around rule
        let before_idx = lines.iter().position(|l| l == &"Before").unwrap();
        let after_idx = lines.iter().position(|l| l == &"After").unwrap();
        // Rule should be on its own with blanks around it
        assert_eq!(
            lines[before_idx + 1],
            "",
            "Should have blank line after Before"
        );
        assert!(lines[before_idx + 2].contains("─"), "Should have rule");
        assert_eq!(
            lines[after_idx - 1],
            "",
            "Should have blank line before After"
        );
    }

    #[test]
    fn test_multiple_paragraphs() {
        let result = render_to_text(
            "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.",
            80,
        );
        let lines: Vec<&str> = result.lines().collect();
        // Should have: p1, blank, p2, blank, p3
        assert_eq!(lines.len(), 5, "Expected 5 lines, got:\n{}", result);
        assert_eq!(lines[0], "First paragraph.");
        assert_eq!(lines[1], "");
        assert_eq!(lines[2], "Second paragraph.");
        assert_eq!(lines[3], "");
        assert_eq!(lines[4], "Third paragraph.");
    }

    #[test]
    fn test_list_with_multiline_items() {
        let input = "1. First item\n   with continuation\n2. Second item\n   also continued";
        let result = render_to_text(input, 80);
        let lines: Vec<&str> = result.lines().collect();

        // First item should start with "1. "
        assert!(lines[0].starts_with("1. "), "First line: {:?}", lines[0]);
        // Soft breaks join continuation to the same paragraph, so all first-item
        // text may appear on a single line at wide widths
        let first_item_text = lines.join(" ");
        assert!(
            first_item_text.contains("First item"),
            "Should contain first item text"
        );
        assert!(
            first_item_text.contains("with continuation"),
            "Should contain continuation"
        );

        // Second item should start with "2. "
        let line2_idx = lines
            .iter()
            .position(|l| l.starts_with("2. "))
            .expect("Should find '2. '");
        assert!(line2_idx >= 1, "Second item should appear after first");
    }

    #[test]
    fn test_no_trailing_blank_lines() {
        let result = render_to_text("Text\n\n## Heading\n\nParagraph", 80);
        // Should not end with blank lines
        assert!(
            !result.ends_with("\n\n"),
            "Should not have trailing blank lines: {:?}",
            result
        );
    }

    #[test]
    fn test_inline_code() {
        let result = render_to_text("Use `code` here", 80);
        assert!(result.contains("code"));
    }

    #[test]
    fn test_bold_and_italic() {
        let result = render_to_text("**bold** and *italic* text", 80);
        // Just verify it renders without panicking and contains the text
        assert!(result.contains("bold"));
        assert!(result.contains("italic"));
    }

    #[test]
    fn test_table_basic() {
        let input = "| A | B |\n|---|---|\n| 1 | 2 |";
        let result = render_to_text(input, 80);
        eprintln!("Table output:\n{}", result);
        assert!(result.contains('┌'), "Expected top-left corner");
        assert!(result.contains('│'), "Expected vertical border");
        assert!(result.contains('└'), "Expected bottom-left corner");
        assert!(result.contains(" A "), "Expected cell A");
        assert!(result.contains(" B "), "Expected cell B");
        assert!(result.contains(" 1 "), "Expected cell 1");
        assert!(result.contains(" 2 "), "Expected cell 2");
    }

    #[test]
    fn test_table_column_widths() {
        let input = "| Short | Longer text |\n|---|---|\n| A | B |";
        let result = render_to_text(input, 80);
        eprintln!("Table output:\n{}", result);
        assert!(result.contains("Short"), "Expected Short");
        assert!(result.contains("Longer text"), "Expected Longer text");
        // Columns should be sized to fit longest content
        let lines: Vec<&str> = result.lines().collect();
        // All border lines should be same width
        let border_widths: Vec<usize> = lines
            .iter()
            .filter(|l| l.starts_with('┌') || l.starts_with('├') || l.starts_with('└'))
            .map(|l| l.chars().count())
            .collect();
        assert!(
            border_widths.windows(2).all(|w| w[0] == w[1]),
            "Border lines should be same width: {:?}",
            border_widths
        );
    }

    #[test]
    fn test_table_multiple_rows() {
        let input = "| H1 | H2 | H3 |\n|----|----|----|\n| A | B | C |\n| D | E | F |";
        let result = render_to_text(input, 80);
        eprintln!("Table output:\n{}", result);
        assert!(result.contains('├'), "Expected row separators");
        assert!(result.contains('┼'), "Expected cross junctions");
    }

    fn line_text(line: &RenderedLine) -> String {
        line.spans.iter().map(|(text, _)| text.as_str()).collect()
    }

    fn rendered_text(conversation: &RenderedConversation) -> String {
        conversation
            .lines
            .iter()
            .map(line_text)
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn test_render_options(tool_display: ToolDisplayMode) -> RenderOptions {
        RenderOptions {
            tool_display,
            show_thinking: false,
            show_timing: false,
            content_width: 80,
            expanded_tool_outputs: BTreeSet::new(),
        }
    }

    fn tool_summary_entries() -> Vec<RenderableEntry> {
        vec![
            RenderableEntry {
                entry_index: 0,
                entry: serde_json::from_str(
                    r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"Grep","input":{"pattern":"one"}}]}}"#,
                )
                .unwrap(),
            },
            RenderableEntry {
                entry_index: 1,
                entry: serde_json::from_str(
                    r#"{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"grep result"}]}}"#,
                )
                .unwrap(),
            },
            RenderableEntry {
                entry_index: 2,
                entry: serde_json::from_str(
                    r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"toolu_2","name":"Read","input":{"file_path":"src/main.rs"}},{"type":"tool_use","id":"toolu_3","name":"Bash","input":{"command":"cargo test"}}]}}"#,
                )
                .unwrap(),
            },
            RenderableEntry {
                entry_index: 3,
                entry: serde_json::from_str(
                    r#"{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_3","content":"bash result"}]}}"#,
                )
                .unwrap(),
            },
        ]
    }

    #[test]
    fn hidden_tool_mode_renders_activity_summary() {
        let entry = RenderableEntry {
            entry_index: 0,
            entry: serde_json::from_str(
                r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"Grep","input":{"pattern":"one"}},{"type":"tool_use","id":"toolu_2","name":"Grep","input":{"pattern":"two"}},{"type":"tool_use","id":"toolu_3","name":"Read","input":{"file_path":"src/main.rs"}}]}}"#,
            )
            .unwrap(),
        };
        let rendered =
            render_parsed_conversation(&[entry], &test_render_options(ToolDisplayMode::Hidden));
        let text = rendered_text(&rendered);

        assert!(text.contains("Searched for 2 patterns"));
        assert!(text.contains("read 1 file"));
        assert!(!text.contains("Grep:"));
        assert!(!text.contains("Read:"));
    }

    #[test]
    fn hidden_tool_mode_coalesces_tool_only_entries_across_results() {
        let entries = tool_summary_entries();
        let rendered =
            render_parsed_conversation(&entries, &test_render_options(ToolDisplayMode::Hidden));
        let text = rendered_text(&rendered);

        assert!(text.contains("Searched for 1 pattern, read 1 file, ran 1 shell command"));
        assert_eq!(text.matches("Claude").count(), 1);
        assert!(!text.contains("Result"));
    }

    #[test]
    fn expanded_tool_summary_renders_truncated_details() {
        let entries = tool_summary_entries();
        let mut options = test_render_options(ToolDisplayMode::Hidden);
        options
            .expanded_tool_outputs
            .insert(make_tool_summary_output_id(0, None));
        let rendered = render_parsed_conversation(&entries, &options);
        let text = rendered_text(&rendered);

        assert!(!text.contains("Searched for 1 pattern, read 1 file, ran 1 shell command"));
        assert!(text.contains("Grep: \"one\" in ."));
        assert!(text.contains("Read: src/main.rs"));
        assert!(text.contains("Bash: cargo test"));
        assert!(text.contains("↳ Result"));
        assert!(text.contains("bash result"));
        assert!(rendered.lines.iter().any(|line| {
            line.clickable
                && line.tool_output_id.as_ref() == Some(&make_tool_summary_output_id(0, None))
        }));
    }

    #[test]
    fn hidden_tool_mode_status_label_is_summary() {
        assert_eq!(ToolDisplayMode::Hidden.status_label(), "sum");
    }

    #[test]
    fn tool_output_ids_use_stable_literal_format() {
        assert_eq!(
            make_tool_output_id(0, None, 0, ToolOutputKind::ToolCall, Some("toolu_1")).0,
            "entry:0:parent:top:block:0:kind:call:id:toolu_1"
        );
        assert_eq!(
            make_tool_output_id(
                1,
                Some("toolu_parent"),
                2,
                ToolOutputKind::ToolResult,
                Some("toolu_2"),
            )
            .0,
            "entry:1:parent:toolu_parent:block:2:kind:result:id:toolu_2"
        );
        assert_eq!(
            make_tool_summary_output_id(3, Some("toolu_parent")).0,
            "entry:3:parent:toolu_parent:kind:summary"
        );
    }

    #[test]
    fn parse_conversation_file_preserves_entry_indices() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("conversation.jsonl");
        std::fs::write(
            &path,
            concat!(
                "\n",
                r#"{"type":"user","message":{"role":"user","content":"first"}}"#,
                "\n",
                "not json\n",
                r#"{"type":"file-history-snapshot","messageId":"m1","snapshot":{},"isSnapshotUpdate":false}"#,
                "\n",
                r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"second"}]}}"#,
                "\n",
            ),
        )
        .unwrap();

        let entries = parse_conversation_file(&path).unwrap();

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].entry_index, 0);
        assert_eq!(entries[1].entry_index, 2);
    }

    #[test]
    fn show_thinking_controls_subagent_entries() {
        let entries = vec![
            RenderableEntry {
                entry_index: 0,
                entry: serde_json::from_str(
                    r#"{"type":"assistant","parent_tool_use_id":"toolu_parent","message":{"role":"assistant","content":[{"type":"text","text":"subagent text"}]}}"#,
                )
                .unwrap(),
            },
            RenderableEntry {
                entry_index: 1,
                entry: serde_json::from_str(
                    r#"{"type":"progress","data":{"type":"agent_progress","agentId":"agent-abcdef123456","message":{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"agent progress text"}]}}}}"#,
                )
                .unwrap(),
            },
        ];
        let hidden =
            render_parsed_conversation(&entries, &test_render_options(ToolDisplayMode::Hidden));
        assert!(!rendered_text(&hidden).contains("subagent text"));
        assert!(!rendered_text(&hidden).contains("agent progress text"));

        let mut options = test_render_options(ToolDisplayMode::Hidden);
        options.show_thinking = true;
        let shown = render_parsed_conversation(&entries, &options);
        let text = rendered_text(&shown);
        assert!(text.contains("subagent text"));
        assert!(text.contains("agent progress text"));
    }

    #[test]
    fn tool_call_metadata_tracks_truncated_and_expanded_state() {
        let input = serde_json::json!({"command":"one\ntwo\nthree\nfour\nfive"});
        let output_id = make_tool_output_id(0, None, 0, ToolOutputKind::ToolCall, Some("toolu_1"));
        let mut lines = Vec::new();
        render_tool_call(
            &mut lines,
            "Bash",
            &input,
            "Claude",
            th().accent_dim,
            false,
            80,
            None,
            ToolDisplayMode::Truncated,
            &output_id,
            false,
        );
        assert!(
            lines
                .iter()
                .any(|line| line_text(line).contains("more lines"))
        );
        assert!(
            lines
                .iter()
                .any(|line| line.clickable && line.tool_output_id.as_ref() == Some(&output_id))
        );
        assert!(!lines.iter().any(|line| line_text(line).contains("five")));

        let mut expanded = Vec::new();
        render_tool_call(
            &mut expanded,
            "Bash",
            &input,
            "Claude",
            th().accent_dim,
            false,
            80,
            None,
            ToolDisplayMode::Truncated,
            &output_id,
            true,
        );
        assert!(
            !expanded
                .iter()
                .any(|line| line_text(line).contains("more lines"))
        );
        assert!(expanded.iter().any(|line| line_text(line).contains("five")));
    }

    #[test]
    fn test_format_timestamp() {
        // UTC timestamp with Z suffix
        let ts = "2026-02-04T19:46:38.440Z";
        let result = format_timestamp(ts);
        assert!(result.is_some(), "Should parse UTC timestamp");
        let formatted = result.unwrap();
        // Should be HH:MM format (local time)
        assert_eq!(formatted.len(), 5, "Should be HH:MM format: {}", formatted);
        assert!(
            formatted.contains(':'),
            "Should contain colon: {}",
            formatted
        );

        // Timestamp with timezone offset
        let ts2 = "2026-02-04T14:46:38-05:00";
        let result2 = format_timestamp(ts2);
        assert!(result2.is_some(), "Should parse timestamp with offset");
    }

    // -----------------------------------------------------------------
    // Render regression harness
    //
    // These tests pin the observable behavior of the rendering pipeline
    // (text, span styles, clickability, tool output IDs, message ranges)
    // so subsequent refactors of the viewer can detect drift.
    // -----------------------------------------------------------------

    fn line_style_at<'a>(line: &'a RenderedLine, text: &str) -> &'a LineStyle {
        &line
            .spans
            .iter()
            .find(|(t, _)| t == text)
            .unwrap_or_else(|| panic!("span {:?} not found in line {:?}", text, line_text(line)))
            .1
    }

    fn user_entry(entry_index: usize, text: &str, timestamp: Option<&str>) -> RenderableEntry {
        let ts_field = match timestamp {
            Some(t) => format!(r#","timestamp":"{}""#, t),
            None => String::new(),
        };
        let json = format!(
            r#"{{"type":"user","message":{{"role":"user","content":"{}"}}{}}}"#,
            text, ts_field
        );
        RenderableEntry {
            entry_index,
            entry: serde_json::from_str(&json).unwrap(),
        }
    }

    fn assistant_text_entry(
        entry_index: usize,
        text: &str,
        timestamp: Option<&str>,
    ) -> RenderableEntry {
        let ts_field = match timestamp {
            Some(t) => format!(r#","timestamp":"{}""#, t),
            None => String::new(),
        };
        let json = format!(
            r#"{{"type":"assistant","message":{{"role":"assistant","content":[{{"type":"text","text":"{}"}}]}}{}}}"#,
            text, ts_field
        );
        RenderableEntry {
            entry_index,
            entry: serde_json::from_str(&json).unwrap(),
        }
    }

    #[test]
    fn message_ranges_track_user_and_assistant_entries() {
        let entries = vec![
            user_entry(0, "Hello", None),
            assistant_text_entry(1, "Hi there", None),
        ];
        let rendered =
            render_parsed_conversation(&entries, &test_render_options(ToolDisplayMode::Hidden));

        // Two messages, each with one content line and one trailing blank.
        assert_eq!(rendered.messages.len(), 2);

        let user = &rendered.messages[0];
        assert_eq!(user.entry_index, 0);
        assert_eq!(user.start_line, 0);
        assert_eq!(user.end_line, 1, "user range excludes trailing blank");

        let assistant = &rendered.messages[1];
        assert_eq!(assistant.entry_index, 1);
        assert_eq!(assistant.start_line, 2);
        assert_eq!(assistant.end_line, 3);

        // Lines: [user, blank, assistant, blank]
        assert_eq!(rendered.lines.len(), 4);
        assert!(rendered.lines[1].spans.is_empty());
        assert!(rendered.lines[3].spans.is_empty());
    }

    #[test]
    fn message_ranges_skip_non_message_entries() {
        // A summary-only entry produces no rendered output and no MessageRange.
        let entries = vec![
            RenderableEntry {
                entry_index: 0,
                entry: serde_json::from_str(r#"{"type":"summary","summary":"ignored"}"#).unwrap(),
            },
            user_entry(1, "Hello", None),
        ];
        let rendered =
            render_parsed_conversation(&entries, &test_render_options(ToolDisplayMode::Hidden));

        assert_eq!(rendered.messages.len(), 1);
        assert_eq!(rendered.messages[0].entry_index, 1);
        assert_eq!(rendered.messages[0].start_line, 0);
    }

    #[test]
    fn timing_enabled_renders_timestamp_prefix_span() {
        let entries = vec![user_entry(0, "Hello", Some("2026-02-04T12:34:56Z"))];
        let mut options = test_render_options(ToolDisplayMode::Hidden);
        options.show_timing = true;

        let rendered = render_parsed_conversation(&entries, &options);
        let first = &rendered.lines[0];
        let ts_span = &first.spans[0].0;

        assert_eq!(
            ts_span.len(),
            TIMESTAMP_WIDTH,
            "timestamp prefix span width: {:?}",
            ts_span
        );
        assert!(
            ts_span.starts_with(' ') && ts_span.ends_with(' ') && ts_span.contains(':'),
            "timestamp prefix should be ' HH:MM ', got {:?}",
            ts_span
        );
    }

    #[test]
    fn timing_disabled_omits_timestamp_prefix_span() {
        let entries = vec![user_entry(0, "Hello", Some("2026-02-04T12:34:56Z"))];
        let rendered =
            render_parsed_conversation(&entries, &test_render_options(ToolDisplayMode::Hidden));
        let first = &rendered.lines[0];

        // First span is the right-aligned name column, not a timestamp.
        assert_eq!(first.spans[0].0.trim(), "You");
    }

    #[test]
    fn invalid_timestamp_skips_timestamp_prefix() {
        // Even with show_timing=true, a non-RFC3339 timestamp produces no time prefix.
        let entries = vec![user_entry(0, "Hello", Some("not-a-timestamp"))];
        let mut options = test_render_options(ToolDisplayMode::Hidden);
        options.show_timing = true;

        let rendered = render_parsed_conversation(&entries, &options);
        let first = &rendered.lines[0];
        assert_eq!(
            first.spans[0].0.trim(),
            "You",
            "first span should be name column, not timestamp"
        );
    }

    #[test]
    fn assistant_continuation_line_aligns_under_timestamp() {
        // Multi-line assistant text should pad continuation lines to the
        // timestamp width so the name column stays aligned.
        let entries = vec![assistant_text_entry(
            0,
            "line one\\n\\nline two",
            Some("2026-02-04T12:34:56Z"),
        )];
        let mut options = test_render_options(ToolDisplayMode::Hidden);
        options.show_timing = true;

        let rendered = render_parsed_conversation(&entries, &options);
        // Expect at least: header line + blank-paragraph + content line.
        let timestamp_span = &rendered.lines[0].spans[0].0;
        assert_eq!(timestamp_span.len(), TIMESTAMP_WIDTH);

        // Find a continuation line (one whose first span is whitespace of TIMESTAMP_WIDTH).
        let has_padded_continuation = rendered.lines.iter().skip(1).any(|line| {
            line.spans
                .first()
                .is_some_and(|(t, _)| t.len() == TIMESTAMP_WIDTH && t.trim().is_empty())
        });
        assert!(
            has_padded_continuation,
            "expected a continuation line padded to TIMESTAMP_WIDTH"
        );
    }

    #[test]
    fn user_label_uses_text_primary_bold() {
        let entries = vec![user_entry(0, "Hello", None)];
        let rendered =
            render_parsed_conversation(&entries, &test_render_options(ToolDisplayMode::Hidden));
        let line = &rendered.lines[0];
        let name_text = format!("{:>width$}", "You", width = NAME_WIDTH);
        let style = line_style_at(line, &name_text);

        assert_eq!(style.fg, Some(th().text_primary));
        assert!(style.bold);
        assert!(!style.dimmed);
    }

    #[test]
    fn assistant_label_uses_accent_bold() {
        let entries = vec![assistant_text_entry(0, "Hi", None)];
        let rendered =
            render_parsed_conversation(&entries, &test_render_options(ToolDisplayMode::Hidden));
        let line = &rendered.lines[0];
        let name_text = format!("{:>width$}", "Claude", width = NAME_WIDTH);
        let style = line_style_at(line, &name_text);

        assert_eq!(style.fg, Some(th().accent));
        assert!(style.bold);
    }

    #[test]
    fn subagent_assistant_uses_nested_label_when_thinking_shown() {
        let entries = vec![RenderableEntry {
            entry_index: 0,
            entry: serde_json::from_str(
                r#"{"type":"assistant","parent_tool_use_id":"toolu_parent_abc","message":{"role":"assistant","content":[{"type":"text","text":"sub text"}]}}"#,
            )
            .unwrap(),
        }];
        let mut options = test_render_options(ToolDisplayMode::Hidden);
        options.show_thinking = true;

        let rendered = render_parsed_conversation(&entries, &options);
        let text = rendered_text(&rendered);
        assert!(text.contains("sub text"));
        assert!(
            text.contains('↳'),
            "subagent rows should use the nested-label arrow: {}",
            text
        );
    }

    #[test]
    fn truncated_tool_call_header_carries_expected_tool_output_id() {
        let entries = vec![RenderableEntry {
            entry_index: 7,
            entry: serde_json::from_str(
                r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"toolu_xyz","name":"Bash","input":{"command":"ls"}}]}}"#,
            )
            .unwrap(),
        }];
        let rendered =
            render_parsed_conversation(&entries, &test_render_options(ToolDisplayMode::Truncated));

        let expected = make_tool_output_id(7, None, 0, ToolOutputKind::ToolCall, Some("toolu_xyz"));
        assert_eq!(
            expected.0,
            "entry:7:parent:top:block:0:kind:call:id:toolu_xyz"
        );
        assert!(
            rendered
                .lines
                .iter()
                .any(|line| line.tool_output_id.as_ref() == Some(&expected)),
            "expected at least one rendered line tagged with the tool output id"
        );
    }

    #[test]
    fn full_tool_mode_lines_are_not_clickable() {
        let entries = vec![RenderableEntry {
            entry_index: 0,
            entry: serde_json::from_str(
                r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"Bash","input":{"command":"one\ntwo\nthree\nfour\nfive"}}]}}"#,
            )
            .unwrap(),
        }];
        let rendered =
            render_parsed_conversation(&entries, &test_render_options(ToolDisplayMode::Full));

        assert!(
            rendered.lines.iter().all(|line| !line.clickable),
            "Full tool display mode should not produce clickable lines"
        );
        // Body should be fully visible — no truncation indicator.
        let text = rendered_text(&rendered);
        assert!(text.contains("five"));
        assert!(!text.contains("more lines"));
    }

    #[test]
    fn tool_result_string_content_renders_as_text() {
        let entries = vec![
            RenderableEntry {
                entry_index: 0,
                entry: serde_json::from_str(
                    r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"Bash","input":{"command":"echo hi"}}]}}"#,
                )
                .unwrap(),
            },
            RenderableEntry {
                entry_index: 1,
                entry: serde_json::from_str(
                    r#"{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"hello-world-output"}]}}"#,
                )
                .unwrap(),
            },
        ];
        let rendered =
            render_parsed_conversation(&entries, &test_render_options(ToolDisplayMode::Truncated));

        let text = rendered_text(&rendered);
        assert!(
            text.contains("hello-world-output"),
            "tool result string content should render verbatim: {}",
            text
        );
    }

    #[test]
    fn fixture_file_round_trip_renders_user_and_assistant() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("fixture.jsonl");
        std::fs::write(
            &path,
            concat!(
                r#"{"type":"user","message":{"role":"user","content":"hello"}}"#,
                "\n",
                r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"world"}]}}"#,
                "\n",
            ),
        )
        .unwrap();

        let rendered =
            render_conversation(&path, &test_render_options(ToolDisplayMode::Hidden)).unwrap();
        let text = rendered_text(&rendered);
        assert!(text.contains("hello"));
        assert!(text.contains("world"));
        assert_eq!(rendered.messages.len(), 2);
    }

    #[test]
    fn consecutive_blank_lines_collapse_and_remap_ranges() {
        // Two adjacent user messages each emit a trailing blank; the dedup pass
        // collapses any double-blank that would arise from this sequence.
        let entries = vec![user_entry(0, "first", None), user_entry(1, "second", None)];
        let rendered =
            render_parsed_conversation(&entries, &test_render_options(ToolDisplayMode::Hidden));

        // No two consecutive empty lines should remain.
        for pair in rendered.lines.windows(2) {
            assert!(
                !(pair[0].spans.is_empty() && pair[1].spans.is_empty()),
                "consecutive empty lines should be collapsed"
            );
        }

        // Both user message ranges should still target valid, distinct lines.
        assert_eq!(rendered.messages.len(), 2);
        assert!(rendered.messages[0].end_line <= rendered.messages[1].start_line);
        assert!(rendered.messages[0].start_line < rendered.messages[0].end_line);
        assert!(rendered.messages[1].start_line < rendered.messages[1].end_line);
    }
}
