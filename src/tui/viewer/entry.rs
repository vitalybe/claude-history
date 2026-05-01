use crate::claude::{self, AssistantMessage, ContentBlock, LogEntry, UserContent};

use super::RenderedLine;

use super::commands::process_command_message;
use super::context::{
    RowTiming, render_dimmed_tool_result_body, render_subagent_tool_result_header,
};
use super::ledger::{render_ledger_block_styled, render_ledger_block_styled_dimmed};
use super::markdown::{apply_thinking_style, render_markdown_to_lines};
use super::summary::{render_tool_activity_summary, summarize_tool_calls};
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
    let mut timing = RowTiming::new(options.show_timing, timestamp);
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
                timing.show_timing(),
            );
        } else if is_skill {
            render_ledger_block_styled_dimmed(
                lines,
                "You",
                th().text_primary,
                md_lines,
                timing.show_timing(),
            );
        } else {
            render_ledger_block_styled(
                lines,
                "You",
                th().text_primary,
                true,
                md_lines,
                timing.take_once(),
            );
        }
        // Whichever branch ran, the top-level timestamp slot is now spent.
        let _ = timing.take_once();
        printed = true;
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
                    let content_str = format_tool_result_content(content.as_ref());
                    render_subagent_tool_result_header(lines, timing.show_timing());
                    render_dimmed_tool_result_body(
                        lines,
                        options,
                        &output_id,
                        expanded,
                        &content_str,
                    );
                } else {
                    let content_str = tool_result_display_text(content.as_ref());
                    render_tool_result(
                        lines,
                        &ToolResultRenderSpec {
                            text: &content_str,
                            content_width: options.content_width,
                            timestamp: timing.consume(),
                            tool_display: options.tool_display,
                            tool_output_id: &output_id,
                            expanded,
                        },
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

fn render_assistant_message(
    lines: &mut Vec<RenderedLine>,
    message: &AssistantMessage,
    options: &RenderOptions,
    timestamp: Option<&str>,
    parent_id: Option<&str>,
    entry_index: usize,
) {
    let mut printed = false;
    let mut timing = RowTiming::new(options.show_timing, timestamp);
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
                    timing.show_timing(),
                );
                // Nested rows do not display the timestamp, but a top-level
                // timestamp is still considered consumed by the first
                // rendered block — mirror the pre-refactor behavior.
                let _ = timing.take_once();
            } else {
                render_ledger_block_styled(
                    lines,
                    "Claude",
                    th().accent,
                    true,
                    md_lines,
                    timing.take_once(),
                );
            }
            printed = true;
        }
    }

    if options.tool_display.is_summary() {
        let summary = summarize_tool_calls(&message.content);
        if !summary.is_empty() {
            let label = nested_label.as_deref().unwrap_or("Claude");
            render_tool_activity_summary(
                lines,
                label,
                th().accent_dim,
                nested_label.is_some(),
                timing.consume(),
                &summary,
                None,
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
                    render_tool_call(
                        lines,
                        &ToolCallRenderSpec {
                            name,
                            input,
                            label,
                            label_color: th().accent_dim,
                            dimmed: true,
                            content_width: options.content_width,
                            timestamp: timing.pad(),
                            tool_display: options.tool_display,
                            tool_output_id: &output_id,
                            expanded,
                        },
                    );
                } else {
                    render_tool_call(
                        lines,
                        &ToolCallRenderSpec {
                            name,
                            input,
                            label: "Claude",
                            label_color: th().accent_dim,
                            dimmed: false,
                            content_width: options.content_width,
                            timestamp: timing.consume(),
                            tool_display: options.tool_display,
                            tool_output_id: &output_id,
                            expanded,
                        },
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
                render_ledger_block_styled(
                    lines,
                    "Thinking",
                    th().accent_dim,
                    false,
                    styled_lines,
                    timing.consume(),
                );
                printed = true;
            }
        }
    }

    if printed {
        lines.push(RenderedLine::new(vec![]));
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
                        render_subagent_tool_result_header(lines, options.show_timing);
                        let content_str = format_tool_result_content(content.as_ref());
                        render_dimmed_tool_result_body(
                            lines,
                            options,
                            &output_id,
                            expanded,
                            &content_str,
                        );
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

            let pad_ts = if options.show_timing {
                Some("     ")
            } else {
                None
            };
            if options.tool_display.is_summary() {
                let summary = summarize_tool_calls(blocks);
                let name = format!("↳{}", short_id);
                render_tool_activity_summary(
                    lines,
                    &name,
                    th().accent_dim,
                    true,
                    pad_ts,
                    &summary,
                    None,
                );
                printed |= !summary.is_empty();
            }

            // Tool calls
            if options.tool_display.shows_details() {
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
                            &ToolCallRenderSpec {
                                name,
                                input,
                                label: &label,
                                label_color: th().accent_dim,
                                dimmed: true,
                                content_width: options.content_width,
                                timestamp: pad_ts,
                                tool_display: options.tool_display,
                                tool_output_id: &output_id,
                                expanded,
                            },
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
