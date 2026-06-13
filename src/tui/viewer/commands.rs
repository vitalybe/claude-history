/// Process user message text to handle command-related XML tags.
/// Returns None if the message should be skipped entirely (e.g., empty local-command-stdout).
pub(crate) fn process_command_message(text: &str) -> Option<String> {
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
