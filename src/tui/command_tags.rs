pub(crate) fn parse_command_name(text: &str) -> Option<&str> {
    let trimmed = text.trim();

    if let Some(start) = trimmed.find("<command-name>")
        && let Some(end) = trimmed.find("</command-name>")
    {
        let content_start = start + "<command-name>".len();
        if content_start < end {
            return Some(&trimmed[content_start..end]);
        }
    }

    None
}

pub(crate) fn parse_command_name_and_args(text: &str) -> Option<String> {
    let trimmed = text.trim();
    let command_name = parse_command_name(trimmed)?;

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

    Some(command_name.to_string())
}
