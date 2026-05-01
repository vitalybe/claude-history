use crate::claude::{self, AssistantMessage, ContentBlock, LogEntry, UserContent};
use crate::tui::app::RenderedLine;

use super::commands::process_command_message;
use super::ledger::{
    render_continuation_dimmed, render_ledger_block_plain_dimmed, render_ledger_block_styled,
    render_ledger_block_styled_dimmed, render_truncation_indicator,
};
use super::markdown::{apply_thinking_style, render_markdown_to_lines};
use super::summary::{render_tool_activity_summary, summarize_tool_calls};
use super::tools::{
    ToolOutputKind, extract_tool_result_text, format_tool_result_content, make_tool_output_id,
    render_tool_call, render_tool_result,
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
                        if total > TRUNCATED_RESULT_LINES {
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
                            if total > TRUNCATED_RESULT_LINES {
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
                    None,
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
