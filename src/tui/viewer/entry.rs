use std::borrow::Cow;

use crate::claude::{self, ContentBlock, LogEntry, UserContent};

use super::RenderedLine;

use super::commands::process_command_message;
use super::context::{render_dimmed_tool_result_body, render_subagent_tool_result_header};
use super::ledger::{render_ledger_block_styled, render_ledger_block_styled_dimmed};
use super::markdown::{apply_thinking_style, render_markdown_to_lines};
use super::summary::{render_tool_activity_summary, summarize_tool_calls};
use super::timing::{RowTiming, TimingSlot};
use super::tools::{
    ToolCallRenderSpec, ToolOutputKind, ToolResultRenderSpec, format_tool_result_content,
    make_tool_output_id, render_tool_call, render_tool_result, tool_result_display_text,
};
use super::*;

pub(super) fn render_entry(
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
            if parent_tool_use_id.is_some() && !options.show_thinking {
                return;
            }
            let ts = entry_timestamp(options, timestamp.as_deref());
            let parent_id = parent_tool_use_id.as_deref();
            let style = MessageStyle::for_user(parent_id, &message.content);
            let timing = RowTiming::new(options.show_timing, ts.as_deref());
            let mut renderer = MessageRenderer::new(style, parent_id, entry_index, options, timing);
            renderer.render_user_template(lines, &message.content);
        }
        LogEntry::Assistant {
            message,
            timestamp,
            parent_tool_use_id,
            ..
        } => {
            if parent_tool_use_id.is_some() && !options.show_thinking {
                return;
            }
            let ts = entry_timestamp(options, timestamp.as_deref());
            let parent_id = parent_tool_use_id.as_deref();
            let style = MessageStyle::for_assistant(parent_id);
            let timing = RowTiming::new(options.show_timing, ts.as_deref());
            let mut renderer = MessageRenderer::new(style, parent_id, entry_index, options, timing);
            renderer.render_assistant_template(lines, &message.content);
        }
    }
}

fn entry_timestamp(options: &RenderOptions, raw: Option<&str>) -> Option<String> {
    if options.show_timing {
        raw.and_then(format_timestamp)
    } else {
        None
    }
}

/// Captures the visual style and label for a single message render pass.
///
/// Each message kind (top-level user, top-level assistant, subagent
/// user/assistant, agent_progress user/assistant) maps to one of these
/// styles. Centralizing them removes the per-block "is this nested?"
/// branches that previously lived in `render_user_message` /
/// `render_assistant_message` / `render_agent_message`.
struct MessageStyle<'a> {
    label: Cow<'a, str>,
    label_color: (u8, u8, u8),
    /// Whether the label and content render dimmed (subagent / skill).
    dimmed: bool,
    /// Whether the first text-block label is bold.
    bold: bool,
    /// Whether the message renders as a nested/subagent message —
    /// controls the special tool-result header and skips thinking
    /// blocks. Distinct from `dimmed` because skill-mode user messages
    /// are dimmed but not nested.
    is_subagent: bool,
}

impl<'a> MessageStyle<'a> {
    fn for_user(parent_id: Option<&'a str>, content: &UserContent) -> Self {
        if let Some(p) = parent_id {
            return Self {
                label: Cow::Owned(subagent_label(p)),
                label_color: th().text_primary,
                dimmed: true,
                bold: false,
                is_subagent: true,
            };
        }
        let is_skill = match content {
            UserContent::String(s) => s.trim().starts_with("Base directory for this skill:"),
            UserContent::Blocks(blocks) => blocks.iter().any(|block| {
                matches!(block, ContentBlock::Text { text }
                    if text.trim().starts_with("Base directory for this skill:"))
            }),
        };
        Self {
            label: Cow::Borrowed("You"),
            label_color: th().text_primary,
            dimmed: is_skill,
            bold: !is_skill,
            is_subagent: false,
        }
    }

    fn for_assistant(parent_id: Option<&'a str>) -> Self {
        match parent_id {
            Some(p) => Self {
                label: Cow::Owned(subagent_label(p)),
                label_color: th().accent,
                dimmed: true,
                bold: false,
                is_subagent: true,
            },
            None => Self {
                label: Cow::Borrowed("Claude"),
                label_color: th().accent,
                dimmed: false,
                bold: true,
                is_subagent: false,
            },
        }
    }

    fn for_agent_user(short_id: &str) -> Self {
        Self {
            label: Cow::Owned(format!("↳{}", short_id)),
            label_color: th().text_primary,
            dimmed: true,
            bold: false,
            is_subagent: true,
        }
    }

    fn for_agent_assistant(short_id: &str) -> Self {
        Self {
            label: Cow::Owned(format!("↳{}", short_id)),
            label_color: th().accent,
            dimmed: true,
            bold: false,
            is_subagent: true,
        }
    }
}

/// Drives one entry's render pass through an explicit block-pipeline
/// template.
///
/// Each `render_*_template` method enumerates the blocks in the order
/// they should appear (text → tool summary → tool calls → thinking for
/// assistants; text → tool results for users) and calls the per-step
/// helpers below. The template ordering here — text first, then tool
/// activity, then thinking — is the documented behavior, not the raw
/// JSON content order.
struct MessageRenderer<'a> {
    style: MessageStyle<'a>,
    parent_id: Option<&'a str>,
    entry_index: usize,
    options: &'a RenderOptions,
    timing: RowTiming<'a>,
}

impl<'a> MessageRenderer<'a> {
    fn new(
        style: MessageStyle<'a>,
        parent_id: Option<&'a str>,
        entry_index: usize,
        options: &'a RenderOptions,
        timing: RowTiming<'a>,
    ) -> Self {
        Self {
            style,
            parent_id,
            entry_index,
            options,
            timing,
        }
    }

    /// User template: text (first), then tool results (second). Pushes
    /// a trailing blank only when something rendered.
    fn render_user_template(&mut self, lines: &mut Vec<RenderedLine>, content: &UserContent) {
        let mut printed = self.step_user_text(lines, content);
        if self.options.tool_display.shows_details()
            && let UserContent::Blocks(blocks) = content
        {
            printed |= self.step_user_tool_results(lines, blocks);
        }
        if printed {
            lines.push(RenderedLine::new(vec![]));
        }
    }

    /// Assistant template: text → tool summary → tool calls → thinking.
    /// Order is intentional and matches pre-refactor behavior, not the
    /// raw JSON block order.
    fn render_assistant_template(
        &mut self,
        lines: &mut Vec<RenderedLine>,
        blocks: &[ContentBlock],
    ) {
        let mut printed = self.step_assistant_text(lines, blocks);
        printed |= self.step_tool_summary(lines, blocks);
        if self.options.tool_display.shows_details() {
            printed |= self.step_tool_calls(lines, blocks);
        }
        printed |= self.step_thinking(lines, blocks);
        if printed {
            lines.push(RenderedLine::new(vec![]));
        }
    }

    /// Aggregated-text agent_progress template.
    ///
    /// agent_progress entries combine all text blocks into one block
    /// before rendering, then walk tool blocks individually.
    fn render_agent_progress_user(
        &mut self,
        lines: &mut Vec<RenderedLine>,
        blocks: &[ContentBlock],
    ) {
        let mut printed = self.step_aggregated_text(lines, blocks);
        if self.options.tool_display.shows_details() {
            printed |= self.step_agent_tool_results(lines, blocks);
        }
        if printed {
            lines.push(RenderedLine::new(vec![]));
        }
    }

    fn render_agent_progress_assistant(
        &mut self,
        lines: &mut Vec<RenderedLine>,
        blocks: &[ContentBlock],
    ) {
        let mut printed = self.step_aggregated_text(lines, blocks);
        printed |= self.step_tool_summary(lines, blocks);
        if self.options.tool_display.shows_details() {
            printed |= self.step_tool_calls(lines, blocks);
        }
        // agent_progress assistant entries do not render thinking blocks.
        if printed {
            lines.push(RenderedLine::new(vec![]));
        }
    }

    // ---- block-pipeline steps ------------------------------------------------

    fn step_user_text(&mut self, lines: &mut Vec<RenderedLine>, content: &UserContent) -> bool {
        let text = match content {
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
        let Some(text) = text else { return false };
        let md_lines = render_markdown_to_lines(&text, self.options.content_width);
        if self.style.dimmed {
            render_ledger_block_styled_dimmed(
                lines,
                &self.style.label,
                self.style.label_color,
                md_lines,
                self.timing.pad(),
            );
        } else {
            render_ledger_block_styled(
                lines,
                &self.style.label,
                self.style.label_color,
                self.style.bold,
                md_lines,
                self.timing.take_once(),
            );
        }
        // Top-level slot is now spent regardless of the branch taken.
        let _ = self.timing.take_once();
        true
    }

    fn step_assistant_text(
        &mut self,
        lines: &mut Vec<RenderedLine>,
        blocks: &[ContentBlock],
    ) -> bool {
        let mut printed = false;
        for block in blocks {
            let ContentBlock::Text { text } = block else {
                continue;
            };
            if text.trim().is_empty() {
                continue;
            }
            let md_lines = render_markdown_to_lines(text, self.options.content_width);
            if self.style.dimmed {
                render_ledger_block_styled_dimmed(
                    lines,
                    &self.style.label,
                    self.style.label_color,
                    md_lines,
                    self.timing.pad(),
                );
                let _ = self.timing.take_once();
            } else {
                render_ledger_block_styled(
                    lines,
                    &self.style.label,
                    self.style.label_color,
                    self.style.bold,
                    md_lines,
                    self.timing.take_once(),
                );
            }
            printed = true;
        }
        printed
    }

    /// Aggregated-text step used by agent_progress entries: joins all
    /// `ContentBlock::Text` blocks and renders them as one styled block.
    fn step_aggregated_text(
        &mut self,
        lines: &mut Vec<RenderedLine>,
        blocks: &[ContentBlock],
    ) -> bool {
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
        if texts.is_empty() {
            return false;
        }
        let combined = texts.join("\n\n");
        let md_lines = render_markdown_to_lines(&combined, self.options.content_width);
        render_ledger_block_styled_dimmed(
            lines,
            &self.style.label,
            self.style.label_color,
            md_lines,
            self.timing.pad(),
        );
        let _ = self.timing.take_once();
        true
    }

    fn step_tool_summary(
        &mut self,
        lines: &mut Vec<RenderedLine>,
        blocks: &[ContentBlock],
    ) -> bool {
        if !self.options.tool_display.is_summary() {
            return false;
        }
        let summary = summarize_tool_calls(blocks);
        if summary.is_empty() {
            return false;
        }
        render_tool_activity_summary(
            lines,
            &self.style.label,
            th().accent_dim,
            self.style.is_subagent,
            self.timing.consume(),
            &summary,
            None,
        );
        true
    }

    fn step_tool_calls(&mut self, lines: &mut Vec<RenderedLine>, blocks: &[ContentBlock]) -> bool {
        let mut printed = false;
        for (block_idx, block) in blocks.iter().enumerate() {
            let ContentBlock::ToolUse { id, name, input } = block else {
                continue;
            };
            let output_id = make_tool_output_id(
                self.entry_index,
                self.parent_id,
                block_idx,
                ToolOutputKind::ToolCall,
                Some(id),
            );
            let expanded = self.options.expanded_tool_outputs.contains(&output_id);
            let timing = if self.style.is_subagent {
                self.timing.pad()
            } else {
                self.timing.consume()
            };
            render_tool_call(
                lines,
                &ToolCallRenderSpec {
                    name,
                    input,
                    label: &self.style.label,
                    label_color: th().accent_dim,
                    dimmed: self.style.dimmed,
                    content_width: self.options.content_width,
                    timing,
                    tool_display: self.options.tool_display,
                    tool_output_id: &output_id,
                    expanded,
                },
            );
            printed = true;
        }
        printed
    }

    fn step_thinking(&mut self, lines: &mut Vec<RenderedLine>, blocks: &[ContentBlock]) -> bool {
        if !self.options.show_thinking || self.style.is_subagent {
            return false;
        }
        let mut printed = false;
        for block in blocks {
            let ContentBlock::Thinking { thinking, .. } = block else {
                continue;
            };
            if thinking.is_empty() {
                continue;
            }
            let md_lines = render_markdown_to_lines(thinking, self.options.content_width);
            let styled_lines = apply_thinking_style(md_lines);
            render_ledger_block_styled(
                lines,
                "Thinking",
                th().accent_dim,
                false,
                styled_lines,
                self.timing.consume(),
            );
            printed = true;
        }
        printed
    }

    fn step_user_tool_results(
        &mut self,
        lines: &mut Vec<RenderedLine>,
        blocks: &[ContentBlock],
    ) -> bool {
        let mut printed = false;
        for (block_idx, block) in blocks.iter().enumerate() {
            let ContentBlock::ToolResult {
                content,
                tool_use_id,
                ..
            } = block
            else {
                continue;
            };
            let output_id = make_tool_output_id(
                self.entry_index,
                self.parent_id,
                block_idx,
                ToolOutputKind::ToolResult,
                Some(tool_use_id),
            );
            let expanded = self.options.expanded_tool_outputs.contains(&output_id);
            if self.style.is_subagent {
                let content_str = format_tool_result_content(content.as_ref());
                let header_timing = self.timing.pad();
                render_subagent_tool_result_header(lines, header_timing);
                render_dimmed_tool_result_body(
                    lines,
                    self.options,
                    &output_id,
                    expanded,
                    &content_str,
                    header_timing,
                );
            } else {
                let content_str = tool_result_display_text(content.as_ref());
                render_tool_result(
                    lines,
                    &ToolResultRenderSpec {
                        text: &content_str,
                        content_width: self.options.content_width,
                        timing: self.timing.consume(),
                        tool_display: self.options.tool_display,
                        tool_output_id: &output_id,
                        expanded,
                    },
                );
            }
            printed = true;
        }
        printed
    }

    /// Tool-result step for agent_progress user blocks. Always renders
    /// as a dimmed subagent result regardless of `style.is_subagent`,
    /// because agent_progress user content always uses the dimmed
    /// subagent result presentation.
    fn step_agent_tool_results(
        &mut self,
        lines: &mut Vec<RenderedLine>,
        blocks: &[ContentBlock],
    ) -> bool {
        let mut printed = false;
        for (block_idx, block) in blocks.iter().enumerate() {
            let ContentBlock::ToolResult {
                content,
                tool_use_id,
                ..
            } = block
            else {
                continue;
            };
            let output_id = make_tool_output_id(
                self.entry_index,
                self.parent_id,
                block_idx,
                ToolOutputKind::ToolResult,
                Some(tool_use_id),
            );
            let expanded = self.options.expanded_tool_outputs.contains(&output_id);
            let header_timing = TimingSlot::from_show_timing(self.options.show_timing);
            render_subagent_tool_result_header(lines, header_timing);
            let content_str = format_tool_result_content(content.as_ref());
            render_dimmed_tool_result_body(
                lines,
                self.options,
                &output_id,
                expanded,
                &content_str,
                header_timing,
            );
            printed = true;
        }
        printed
    }
}

/// Get a truncated agent ID for display (max 7 characters)
fn short_agent_id(agent_id: &str) -> &str {
    &agent_id[..agent_id.len().min(7)]
}

/// Create a label for subagent entries from a parent_tool_use_id.
pub(super) fn subagent_label(parent_tool_use_id: &str) -> String {
    format!("↳{}", claude::short_parent_id(parent_tool_use_id))
}

/// Render agent (subagent) progress message.
///
/// agent_progress entries wrap a nested user/assistant message; this
/// function dispatches to the matching pipeline template above. The
/// `agent_id` is reused as the synthetic `parent_id` for tool output
/// IDs so that expanded-tool keys remain stable across re-renders.
fn render_agent_message(
    lines: &mut Vec<RenderedLine>,
    entry_index: usize,
    agent_progress: &crate::claude::AgentProgressData,
    options: &RenderOptions,
) {
    use crate::claude::AgentContent;

    let agent_id = &agent_progress.agent_id;
    let short_id = short_agent_id(agent_id);
    let msg = &agent_progress.message;

    match msg.message_type.as_str() {
        "user" => {
            let AgentContent::Blocks(blocks) = &msg.message.content;
            let style = MessageStyle::for_agent_user(short_id);
            let timing = RowTiming::column_only(options.show_timing);
            let mut renderer =
                MessageRenderer::new(style, Some(agent_id), entry_index, options, timing);
            renderer.render_agent_progress_user(lines, blocks);
        }
        "assistant" => {
            let AgentContent::Blocks(blocks) = &msg.message.content;
            let style = MessageStyle::for_agent_assistant(short_id);
            let timing = RowTiming::column_only(options.show_timing);
            let mut renderer =
                MessageRenderer::new(style, Some(agent_id), entry_index, options, timing);
            renderer.render_agent_progress_assistant(lines, blocks);
        }
        _ => {}
    }
}
