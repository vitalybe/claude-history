//! Conversation viewer rendering for TUI display.
//!
//! This module renders conversation JSONL files to `Vec<RenderedLine>` for display
//! in the TUI viewer. It produces styled spans that ratatui can render directly,
//! without using ANSI escape codes.

use crate::claude::{self, AssistantMessage, ContentBlock, LogEntry, UserContent};
use crate::tool_format;
use crate::tui::app::{LineStyle, RenderedLine};
use std::collections::BTreeSet;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::tui::theme::{self, Theme};

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ToolOutputKind {
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
    let mut pending_tool_summary: Option<(
        usize,
        Option<String>,
        Option<String>,
        ToolActivitySummary,
    )> = None;

    for parsed in entries {
        let entry_index = parsed.entry_index;
        let entry = &parsed.entry;

        if options.tool_display.is_summary() {
            if let Some((parent_id, timestamp, summary)) =
                tool_only_assistant_summary(entry, options)
            {
                match &mut pending_tool_summary {
                    Some((_, pending_parent, _, pending_summary))
                        if pending_parent.as_deref() == parent_id =>
                    {
                        pending_summary.merge(summary);
                    }
                    _ => {
                        flush_tool_summary(
                            &mut lines,
                            &mut messages,
                            &mut pending_tool_summary,
                            options,
                        );
                        pending_tool_summary = Some((
                            entry_index,
                            parent_id.map(str::to_string),
                            timestamp.map(str::to_string),
                            summary,
                        ));
                    }
                }
                continue;
            }

            if user_entry_is_only_tool_results(entry, options) {
                continue;
            }
        }

        flush_tool_summary(
            &mut lines,
            &mut messages,
            &mut pending_tool_summary,
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

fn make_tool_output_id(
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

#[derive(Default)]
struct ToolActivitySummary {
    searched_patterns: usize,
    searched_file_patterns: usize,
    read_files: usize,
    shell_commands: usize,
    edited_files: usize,
    wrote_files: usize,
    agents: usize,
    fetched_urls: usize,
    web_searches: usize,
    other_tools: usize,
}

impl ToolActivitySummary {
    fn add_call(&mut self, name: &str) {
        match name {
            "Bash" => self.shell_commands += 1,
            "Read" => self.read_files += 1,
            "Grep" => self.searched_patterns += 1,
            "Glob" => self.searched_file_patterns += 1,
            "Edit" => self.edited_files += 1,
            "Write" => self.wrote_files += 1,
            "Task" => self.agents += 1,
            "WebFetch" => self.fetched_urls += 1,
            "WebSearch" => self.web_searches += 1,
            _ => self.other_tools += 1,
        }
    }

    fn merge(&mut self, other: Self) {
        self.searched_patterns += other.searched_patterns;
        self.searched_file_patterns += other.searched_file_patterns;
        self.read_files += other.read_files;
        self.shell_commands += other.shell_commands;
        self.edited_files += other.edited_files;
        self.wrote_files += other.wrote_files;
        self.agents += other.agents;
        self.fetched_urls += other.fetched_urls;
        self.web_searches += other.web_searches;
        self.other_tools += other.other_tools;
    }

    fn is_empty(&self) -> bool {
        self.searched_patterns
            + self.searched_file_patterns
            + self.read_files
            + self.shell_commands
            + self.edited_files
            + self.wrote_files
            + self.agents
            + self.fetched_urls
            + self.web_searches
            + self.other_tools
            == 0
    }

    fn sentence(&self) -> String {
        let mut parts = Vec::new();
        push_summary_item(
            &mut parts,
            self.searched_patterns,
            "Searched for",
            "pattern",
        );
        push_summary_item(
            &mut parts,
            self.searched_file_patterns,
            "Searched for",
            "file pattern",
        );
        push_summary_item(&mut parts, self.read_files, "read", "file");
        push_summary_item(&mut parts, self.shell_commands, "ran", "shell command");
        push_summary_item(&mut parts, self.edited_files, "edited", "file");
        push_summary_item(&mut parts, self.wrote_files, "wrote", "file");
        push_summary_item(&mut parts, self.agents, "started", "agent");
        push_summary_item(&mut parts, self.fetched_urls, "fetched", "URL");
        push_summary_item(&mut parts, self.web_searches, "searched", "web");
        push_summary_item(&mut parts, self.other_tools, "called", "tool");
        capitalize_first(parts.join(", "))
    }
}

fn capitalize_first(text: String) -> String {
    let mut chars = text.chars();
    let Some(first) = chars.next() else {
        return text;
    };
    first.to_uppercase().chain(chars).collect()
}

fn push_summary_item(parts: &mut Vec<String>, count: usize, verb: &str, noun: &str) {
    if count == 0 {
        return;
    }
    let suffix = if count == 1 { "" } else { "s" };
    parts.push(format!("{verb} {count} {noun}{suffix}"));
}

fn render_tool_activity_summary(
    lines: &mut Vec<RenderedLine>,
    label: &str,
    label_color: (u8, u8, u8),
    dimmed: bool,
    timestamp: Option<&str>,
    summary: &ToolActivitySummary,
) {
    if summary.is_empty() {
        return;
    }

    let mut spans = Vec::new();
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
        format!("{:>width$}", label, width = NAME_WIDTH),
        LineStyle {
            fg: Some(label_color),
            bold: false,
            dimmed,
            italic: false,
        },
    ));
    spans.push((
        " │ ".to_string(),
        LineStyle {
            fg: Some(th().border),
            dimmed,
            ..Default::default()
        },
    ));
    spans.push((
        summary.sentence(),
        LineStyle {
            fg: Some(th().tool_text),
            dimmed: true,
            ..Default::default()
        },
    ));
    lines.push(RenderedLine::new(spans));
}

fn summarize_tool_calls(blocks: &[ContentBlock]) -> ToolActivitySummary {
    let mut summary = ToolActivitySummary::default();
    for block in blocks {
        if let ContentBlock::ToolUse { name, .. } = block {
            summary.add_call(name);
        }
    }
    summary
}

fn assistant_blocks_are_tool_only(blocks: &[ContentBlock]) -> bool {
    blocks
        .iter()
        .all(|block| matches!(block, ContentBlock::ToolUse { .. }))
}

fn tool_only_assistant_summary<'a>(
    entry: &'a LogEntry,
    options: &RenderOptions,
) -> Option<(Option<&'a str>, Option<&'a str>, ToolActivitySummary)> {
    let LogEntry::Assistant {
        message,
        timestamp,
        parent_tool_use_id,
        ..
    } = entry
    else {
        return None;
    };

    if parent_tool_use_id.is_some() && !options.show_thinking {
        return None;
    }
    if message.content.is_empty() || !assistant_blocks_are_tool_only(&message.content) {
        return None;
    }

    let summary = summarize_tool_calls(&message.content);
    (!summary.is_empty()).then_some((parent_tool_use_id.as_deref(), timestamp.as_deref(), summary))
}

fn user_entry_is_only_tool_results(entry: &LogEntry, options: &RenderOptions) -> bool {
    let LogEntry::User {
        message,
        parent_tool_use_id,
        ..
    } = entry
    else {
        return false;
    };

    if parent_tool_use_id.is_some() && !options.show_thinking {
        return false;
    }

    let UserContent::Blocks(blocks) = &message.content else {
        return false;
    };
    !blocks.is_empty()
        && blocks
            .iter()
            .all(|block| matches!(block, ContentBlock::ToolResult { .. }))
}

fn flush_tool_summary(
    lines: &mut Vec<RenderedLine>,
    messages: &mut Vec<MessageRange>,
    pending: &mut Option<(usize, Option<String>, Option<String>, ToolActivitySummary)>,
    options: &RenderOptions,
) {
    let Some((entry_index, parent_id, timestamp, summary)) = pending.take() else {
        return;
    };

    let start_line = lines.len();
    let label = parent_id
        .as_deref()
        .map(subagent_label)
        .unwrap_or_else(|| "Claude".to_string());
    let ts = if options.show_timing {
        timestamp.as_deref().and_then(format_timestamp)
    } else {
        None
    };
    render_tool_activity_summary(
        lines,
        &label,
        th().accent_dim,
        parent_id.is_some(),
        ts.as_deref(),
        &summary,
    );
    let end_line = lines.len();
    if end_line > start_line {
        messages.push(MessageRange {
            entry_index,
            start_line,
            end_line,
        });
        lines.push(RenderedLine::new(vec![]));
    }
}

fn render_entry(
    lines: &mut Vec<RenderedLine>,
    entry_index: usize,
    entry: &LogEntry,
    options: &RenderOptions,
) {
    match entry {
        LogEntry::Summary { .. }
        | LogEntry::FileHistorySnapshot { .. }
        | LogEntry::System { .. }
        | LogEntry::CustomTitle { .. }
        | LogEntry::AgentName { .. } => {}
        LogEntry::Progress { data, .. } => {
            // Handle agent_progress entries (only when show_thinking is enabled)
            if options.show_thinking
                && let Some(agent_progress) = crate::claude::parse_agent_progress(data)
            {
                render_agent_message(lines, entry_index, &agent_progress, options);
            }
        }
        LogEntry::User {
            message,
            timestamp,
            parent_tool_use_id,
            ..
        } => {
            // Subagent messages: show nested when show_thinking, skip otherwise
            if parent_tool_use_id.is_some() && !options.show_thinking {
                return;
            }
            let ts = if options.show_timing {
                timestamp.as_deref().and_then(format_timestamp)
            } else {
                None
            };
            render_user_message(
                lines,
                message,
                options,
                ts.as_deref(),
                parent_tool_use_id.as_deref(),
                entry_index,
            );
        }
        LogEntry::Assistant {
            message,
            timestamp,
            parent_tool_use_id,
            ..
        } => {
            // Subagent messages: show nested when show_thinking, skip otherwise
            if parent_tool_use_id.is_some() && !options.show_thinking {
                return;
            }
            let ts = if options.show_timing {
                timestamp.as_deref().and_then(format_timestamp)
            } else {
                None
            };
            render_assistant_message(
                lines,
                message,
                options,
                ts.as_deref(),
                parent_tool_use_id.as_deref(),
                entry_index,
            );
        }
    }
}

fn render_user_message(
    lines: &mut Vec<RenderedLine>,
    message: &crate::claude::UserMessage,
    options: &RenderOptions,
    timestamp: Option<&str>,
    parent_id: Option<&str>,
    entry_index: usize,
) {
    let mut printed = false;
    let mut ts_remaining = timestamp;
    let nested_label = parent_id.map(subagent_label);

    // Detect if this is a skill invocation message
    let is_skill = match &message.content {
        UserContent::String(s) => s.trim().starts_with("Base directory for this skill:"),
        UserContent::Blocks(blocks) => blocks.iter().any(|block| {
            matches!(block, ContentBlock::Text { text } if text.trim().starts_with("Base directory for this skill:"))
        }),
    };

    // Extract text from user message, collecting all text blocks
    let text = match &message.content {
        UserContent::String(s) => process_command_message(s),
        UserContent::Blocks(blocks) => {
            let texts: Vec<String> = blocks
                .iter()
                .filter_map(|block| {
                    if let ContentBlock::Text { text } = block {
                        process_command_message(text)
                    } else {
                        None
                    }
                })
                .collect();
            if texts.is_empty() {
                None
            } else {
                Some(texts.join("\n\n"))
            }
        }
    };

    if let Some(text) = text {
        let md_lines = render_markdown_to_lines(&text, options.content_width);
        if let Some(ref label) = nested_label {
            render_ledger_block_styled_dimmed(
                lines,
                label,
                th().text_primary,
                md_lines,
                options.show_timing,
            );
        } else if is_skill {
            render_ledger_block_styled_dimmed(
                lines,
                "You",
                th().text_primary,
                md_lines,
                options.show_timing,
            );
        } else {
            render_ledger_block_styled(
                lines,
                "You",
                th().text_primary,
                true,
                md_lines,
                ts_remaining,
            );
        }
        printed = true;
        ts_remaining = None;
    }

    // Tool results (if enabled)
    if options.tool_display.shows_details()
        && let UserContent::Blocks(blocks) = &message.content
    {
        for (block_idx, block) in blocks.iter().enumerate() {
            if let ContentBlock::ToolResult {
                content,
                tool_use_id,
                ..
            } = block
            {
                let output_id = make_tool_output_id(
                    entry_index,
                    parent_id,
                    block_idx,
                    ToolOutputKind::ToolResult,
                    Some(tool_use_id),
                );
                let expanded = options.expanded_tool_outputs.contains(&output_id);
                if nested_label.is_some() {
                    // Dimmed tool result for subagent
                    let content_str = format_tool_result_content(content.as_ref());
                    render_ledger_block_plain_dimmed(
                        lines,
                        "  ↳ Tool",
                        th().accent_dim,
                        "<Result>",
                        options.show_timing,
                    );
                    if options.tool_display == ToolDisplayMode::Truncated && !expanded {
                        let content_lines: Vec<&str> = content_str.lines().collect();
                        let total = content_lines.len();
                        if total > TRUNCATED_RESULT_LINES + 1 {
                            let truncated = content_lines[..TRUNCATED_RESULT_LINES].join("\n");
                            render_continuation_dimmed(
                                lines,
                                &truncated,
                                options.show_timing,
                                Some(&output_id),
                            );
                            render_truncation_indicator(
                                lines,
                                total - TRUNCATED_RESULT_LINES,
                                true,
                                options.show_timing,
                                Some(&output_id),
                            );
                        } else {
                            render_continuation_dimmed(
                                lines,
                                &content_str,
                                options.show_timing,
                                None,
                            );
                        }
                    } else {
                        let id = (options.tool_display == ToolDisplayMode::Truncated)
                            .then_some(&output_id);
                        render_continuation_dimmed(lines, &content_str, options.show_timing, id);
                    }
                } else {
                    let content_str = match extract_tool_result_text(content.as_ref()) {
                        Some(text) => text,
                        None => format_tool_result_content(content.as_ref()),
                    };
                    // Pass timestamp to first tool result if no text block consumed it
                    let ts = if ts_remaining.is_some() {
                        let t = ts_remaining;
                        ts_remaining = None;
                        t
                    } else if options.show_timing {
                        Some("     ")
                    } else {
                        None
                    };
                    render_tool_result(
                        lines,
                        &content_str,
                        options.content_width,
                        ts,
                        options.tool_display,
                        &output_id,
                        expanded,
                    );
                }
                printed = true;
            }
        }
    }

    if printed {
        lines.push(RenderedLine::new(vec![])); // Empty line after message
    }
}

/// Extract text content from tool result for markdown rendering.
/// Returns Some(text) if content is a string or array of text blocks.
/// Returns None for JSON structures that should be pretty-printed instead.
fn extract_tool_result_text(content: Option<&serde_json::Value>) -> Option<String> {
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
fn format_tool_result_content(content: Option<&serde_json::Value>) -> String {
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

fn render_assistant_message(
    lines: &mut Vec<RenderedLine>,
    message: &AssistantMessage,
    options: &RenderOptions,
    timestamp: Option<&str>,
    parent_id: Option<&str>,
    entry_index: usize,
) {
    let mut printed = false;
    let mut ts_remaining = timestamp;
    let nested_label = parent_id.map(subagent_label);

    // Text blocks
    for block in &message.content {
        if let ContentBlock::Text { text } = block {
            if text.trim().is_empty() {
                continue;
            }
            let md_lines = render_markdown_to_lines(text, options.content_width);
            if let Some(ref label) = nested_label {
                render_ledger_block_styled_dimmed(
                    lines,
                    label,
                    th().accent,
                    md_lines,
                    options.show_timing,
                );
            } else {
                render_ledger_block_styled(
                    lines,
                    "Claude",
                    th().accent,
                    true,
                    md_lines,
                    ts_remaining,
                );
            }
            printed = true;
            // After first block consumes the timestamp, use blank padding for alignment
            if ts_remaining.is_some() {
                ts_remaining = None;
            }
        }
    }

    if options.tool_display.is_summary() {
        let summary = summarize_tool_calls(&message.content);
        if !summary.is_empty() {
            let label = nested_label.as_deref().unwrap_or("Claude");
            let ts = if ts_remaining.is_some() {
                let t = ts_remaining;
                ts_remaining = None;
                t
            } else if options.show_timing {
                Some("     ")
            } else {
                None
            };
            render_tool_activity_summary(
                lines,
                label,
                th().accent_dim,
                nested_label.is_some(),
                ts,
                &summary,
            );
            printed = true;
        }
    }

    // Tool calls (if enabled)
    if options.tool_display.shows_details() {
        for (block_idx, block) in message.content.iter().enumerate() {
            if let ContentBlock::ToolUse { id, name, input } = block {
                let output_id = make_tool_output_id(
                    entry_index,
                    parent_id,
                    block_idx,
                    ToolOutputKind::ToolCall,
                    Some(id),
                );
                let expanded = options.expanded_tool_outputs.contains(&output_id);
                if let Some(ref label) = nested_label {
                    let align_ts = if options.show_timing {
                        Some("     ")
                    } else {
                        None
                    };
                    render_tool_call(
                        lines,
                        name,
                        input,
                        label,
                        th().accent_dim,
                        true,
                        options.content_width,
                        align_ts,
                        options.tool_display,
                        &output_id,
                        expanded,
                    );
                } else {
                    // Pass timestamp to first tool call if no text block consumed it
                    let ts = if ts_remaining.is_some() {
                        let t = ts_remaining;
                        ts_remaining = None;
                        t
                    } else if options.show_timing {
                        Some("     ")
                    } else {
                        None
                    };
                    render_tool_call(
                        lines,
                        name,
                        input,
                        "Claude",
                        th().accent_dim,
                        false,
                        options.content_width,
                        ts,
                        options.tool_display,
                        &output_id,
                        expanded,
                    );
                }
                printed = true;
            }
        }
    }

    // Thinking blocks (if enabled, skip for subagents)
    if options.show_thinking && nested_label.is_none() {
        for block in &message.content {
            if let ContentBlock::Thinking { thinking, .. } = block {
                if thinking.is_empty() {
                    continue;
                }
                let md_lines = render_markdown_to_lines(thinking, options.content_width);
                let styled_lines = apply_thinking_style(md_lines);
                // Pass timestamp if no previous block consumed it
                let ts = if ts_remaining.is_some() {
                    let t = ts_remaining;
                    ts_remaining = None;
                    t
                } else if options.show_timing {
                    Some("     ")
                } else {
                    None
                };
                render_ledger_block_styled(
                    lines,
                    "Thinking",
                    th().accent_dim,
                    false,
                    styled_lines,
                    ts,
                );
                printed = true;
            }
        }
    }

    if printed {
        lines.push(RenderedLine::new(vec![]));
    }
}

/// A line with styled spans from markdown rendering
struct StyledLine {
    spans: Vec<(String, LineStyle)>,
}

/// Render markdown text to styled lines for TUI display
fn render_markdown_to_lines(input: &str, max_width: usize) -> Vec<StyledLine> {
    let doc = crate::markdown::layout::LayoutEngine::render(input, max_width);
    doc.lines
        .into_iter()
        // Drop fence-only lines — TUI signals code blocks via color instead.
        // Empty lines are kept (they're blank spacers, not fences).
        .filter(|line| line.runs.is_empty() || line.runs.iter().any(|r| !r.attrs.code_fence))
        .map(|line| StyledLine {
            spans: line
                .runs
                .into_iter()
                .filter(|run| !run.attrs.link_url && !run.attrs.heading_marker)
                .map(|run| (run.text, attrs_to_line_style(&run.attrs)))
                .collect(),
        })
        .collect()
}

fn attrs_to_line_style(attrs: &crate::markdown::layout::Attrs) -> LineStyle {
    let fg = if let Some(rgb) = attrs.fg {
        Some(rgb)
    } else if attrs.code_block_lang.is_some() || attrs.code {
        Some(th().code_color)
    } else if attrs.quote {
        Some(th().green)
    } else if attrs.link {
        Some(th().blue)
    } else if attrs.heading {
        Some(th().heading)
    } else {
        None
    };
    LineStyle {
        bold: attrs.bold || attrs.heading,
        italic: attrs.italic,
        dimmed: attrs.dimmed || attrs.strikethrough,
        fg,
    }
}

/// Apply italic and dimmed styling to thinking block content
fn apply_thinking_style(styled_lines: Vec<StyledLine>) -> Vec<StyledLine> {
    styled_lines
        .into_iter()
        .map(|line| StyledLine {
            spans: line
                .spans
                .into_iter()
                .map(|(text, mut style)| {
                    style.italic = true;
                    style.fg = Some(th().thinking_text);
                    (text, style)
                })
                .collect(),
        })
        .collect()
}

/// Render ledger block with styled markdown lines
fn render_ledger_block_styled(
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

fn push_line(
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
fn render_truncation_indicator(
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

/// Render a formatted tool call with proper styling
#[allow(clippy::too_many_arguments)]
fn render_tool_call(
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
            if total > TRUNCATED_BODY_LINES + 1 {
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
fn render_tool_result(
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
        && total > TRUNCATED_RESULT_LINES + 1
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

/// Get a truncated agent ID for display (max 7 characters)
fn short_agent_id(agent_id: &str) -> &str {
    &agent_id[..agent_id.len().min(7)]
}

/// Create a label for subagent entries from a parent_tool_use_id.
fn subagent_label(parent_tool_use_id: &str) -> String {
    format!("↳{}", claude::short_parent_id(parent_tool_use_id))
}

/// Render agent (subagent) progress message
fn render_agent_message(
    lines: &mut Vec<RenderedLine>,
    entry_index: usize,
    agent_progress: &crate::claude::AgentProgressData,
    options: &RenderOptions,
) {
    use crate::claude::{AgentContent, ContentBlock};

    let agent_id = &agent_progress.agent_id;
    let short_id = short_agent_id(agent_id);
    let msg = &agent_progress.message;
    let mut printed = false;

    match msg.message_type.as_str() {
        "user" => {
            let AgentContent::Blocks(blocks) = &msg.message.content;

            // Aggregate text blocks and render together
            let texts: Vec<&str> = blocks
                .iter()
                .filter_map(|b| {
                    if let ContentBlock::Text { text } = b {
                        Some(text.as_str())
                    } else {
                        None
                    }
                })
                .collect();

            if !texts.is_empty() {
                let combined = texts.join("\n\n");
                let md_lines = render_markdown_to_lines(&combined, options.content_width);
                let name = format!("↳{}", short_id);
                render_ledger_block_styled_dimmed(
                    lines,
                    &name,
                    th().text_primary,
                    md_lines,
                    options.show_timing,
                );
                printed = true;
            }

            // Tool results
            if options.tool_display.shows_details() {
                for (block_idx, block) in blocks.iter().enumerate() {
                    if let ContentBlock::ToolResult {
                        content,
                        tool_use_id,
                        ..
                    } = block
                    {
                        let output_id = make_tool_output_id(
                            entry_index,
                            Some(agent_id),
                            block_idx,
                            ToolOutputKind::ToolResult,
                            Some(tool_use_id),
                        );
                        let expanded = options.expanded_tool_outputs.contains(&output_id);
                        render_ledger_block_plain_dimmed(
                            lines,
                            "  ↳ Tool",
                            th().accent_dim,
                            "<Result>",
                            options.show_timing,
                        );
                        let content_str = format_tool_result_content(content.as_ref());
                        if options.tool_display == ToolDisplayMode::Truncated && !expanded {
                            let content_lines: Vec<&str> = content_str.lines().collect();
                            let total = content_lines.len();
                            if total > TRUNCATED_RESULT_LINES + 1 {
                                let truncated = content_lines[..TRUNCATED_RESULT_LINES].join("\n");
                                render_continuation_dimmed(
                                    lines,
                                    &truncated,
                                    options.show_timing,
                                    Some(&output_id),
                                );
                                render_truncation_indicator(
                                    lines,
                                    total - TRUNCATED_RESULT_LINES,
                                    true,
                                    options.show_timing,
                                    Some(&output_id),
                                );
                            } else {
                                render_continuation_dimmed(
                                    lines,
                                    &content_str,
                                    options.show_timing,
                                    None,
                                );
                            }
                        } else {
                            let id = (options.tool_display == ToolDisplayMode::Truncated)
                                .then_some(&output_id);
                            render_continuation_dimmed(
                                lines,
                                &content_str,
                                options.show_timing,
                                id,
                            );
                        }
                        printed = true;
                    }
                }
            }
        }
        "assistant" => {
            let AgentContent::Blocks(blocks) = &msg.message.content;

            // Aggregate text blocks and render together
            let texts: Vec<&str> = blocks
                .iter()
                .filter_map(|b| {
                    if let ContentBlock::Text { text } = b {
                        Some(text.as_str())
                    } else {
                        None
                    }
                })
                .collect();

            if !texts.is_empty() {
                let combined = texts.join("\n\n");
                let md_lines = render_markdown_to_lines(&combined, options.content_width);
                let name = format!("↳{}", short_id);
                render_ledger_block_styled_dimmed(
                    lines,
                    &name,
                    th().accent,
                    md_lines,
                    options.show_timing,
                );
                printed = true;
            }

            if options.tool_display.is_summary() {
                let summary = summarize_tool_calls(blocks);
                let name = format!("↳{}", short_id);
                render_tool_activity_summary(
                    lines,
                    &name,
                    th().accent_dim,
                    true,
                    options.show_timing.then_some("     "),
                    &summary,
                );
                printed |= !summary.is_empty();
            }

            // Tool calls
            if options.tool_display.shows_details() {
                let align_ts = if options.show_timing {
                    Some("     ")
                } else {
                    None
                };
                for (block_idx, block) in blocks.iter().enumerate() {
                    if let ContentBlock::ToolUse { id, name, input } = block {
                        let output_id = make_tool_output_id(
                            entry_index,
                            Some(agent_id),
                            block_idx,
                            ToolOutputKind::ToolCall,
                            Some(id),
                        );
                        let expanded = options.expanded_tool_outputs.contains(&output_id);
                        let label = format!("↳{}", short_id);
                        render_tool_call(
                            lines,
                            name,
                            input,
                            &label,
                            th().accent_dim,
                            true,
                            options.content_width,
                            align_ts,
                            options.tool_display,
                            &output_id,
                            expanded,
                        );
                        printed = true;
                    }
                }
            }
        }
        _ => {}
    }

    if printed {
        lines.push(RenderedLine::new(vec![]));
    }
}

/// Render ledger block with styled markdown lines (dimmed for subagents)
fn render_ledger_block_styled_dimmed(
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
fn render_ledger_block_plain_dimmed(
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
fn render_continuation_dimmed(
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

/// Process user message text to handle command-related XML tags.
/// Returns None if the message should be skipped entirely (e.g., empty local-command-stdout).
fn process_command_message(text: &str) -> Option<String> {
    let trimmed = text.trim();

    // Check for local-command-caveat - skip these system messages entirely
    if trimmed.starts_with("<local-command-caveat>") && trimmed.ends_with("</local-command-caveat>")
    {
        return None;
    }

    // Check for empty or whitespace-only local-command-stdout - skip these entirely
    if trimmed.starts_with("<local-command-stdout>") && trimmed.ends_with("</local-command-stdout>")
    {
        let tag_start = "<local-command-stdout>".len();
        let tag_end = trimmed.len() - "</local-command-stdout>".len();
        let inner = &trimmed[tag_start..tag_end];
        if inner.trim().is_empty() {
            return None;
        }
        // Non-empty local-command-stdout: show the content without the tags
        return Some(inner.trim().to_string());
    }

    // Check if this is a command message with <command-name> tag
    if let Some(start) = trimmed.find("<command-name>")
        && let Some(end) = trimmed.find("</command-name>")
    {
        let content_start = start + "<command-name>".len();
        if content_start < end {
            let command_name = &trimmed[content_start..end];

            // Skip /clear commands - internal context-clearing, not meaningful to display
            if command_name == "/clear" {
                return None;
            }

            // Also extract command args if present
            if let Some(args_start) = trimmed.find("<command-args>")
                && let Some(args_end) = trimmed.find("</command-args>")
            {
                let args_content_start = args_start + "<command-args>".len();
                if args_content_start < args_end {
                    let args = trimmed[args_content_start..args_end].trim();
                    if !args.is_empty() {
                        return Some(format!("{} {}", command_name, args));
                    }
                }
            }

            return Some(command_name.to_string());
        }
    }

    // Skill invocation expanded prompts - show description instead of full prompt
    if trimmed.starts_with("Base directory for this skill:") {
        let description = trimmed
            .lines()
            .skip(1)
            .find(|l| !l.trim().is_empty())
            .unwrap_or("invoked");
        return Some(format!("*Skill: {}*", description));
    }

    Some(text.to_string())
}

#[cfg(test)]
mod tests {
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
        let entries = vec![
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
                    r#"{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"result"}]}}"#,
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
        ];
        let rendered =
            render_parsed_conversation(&entries, &test_render_options(ToolDisplayMode::Hidden));
        let text = rendered_text(&rendered);

        assert!(text.contains("Searched for 1 pattern, read 1 file, ran 1 shell command"));
        assert_eq!(text.matches("Claude").count(), 1);
        assert!(!text.contains("Result"));
    }

    #[test]
    fn hidden_tool_mode_status_label_is_summary() {
        assert_eq!(ToolDisplayMode::Hidden.status_label(), "sum");
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
}
