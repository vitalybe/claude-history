use crate::history::Conversation;
use crate::semantic::types::SemanticHit;

pub fn format_hit(rank: usize, hit: &SemanticHit, conversations: &[&Conversation]) -> String {
    let conversation = conversations[hit.conversation_index];
    let project = conversation.project_name.as_deref().unwrap_or("(none)");
    let title = conversation
        .custom_title
        .as_deref()
        .or(conversation.summary.as_deref())
        .unwrap_or(&conversation.preview);

    format!(
        "#{rank:2} hybrid={:.3} semantic={:.3} lexical={:.3} | {project} | {}\n     {title}\n     {}\n",
        hit.hybrid_score,
        hit.semantic_score,
        hit.lexical_score,
        hit.session,
        truncate(&hit.snippet, 260)
    )
}

fn truncate(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_owned();
    }

    let mut out: String = text.chars().take(max_chars.saturating_sub(1)).collect();
    out.push('…');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Local;
    use std::path::PathBuf;

    fn conversation() -> Conversation {
        Conversation {
            path: PathBuf::from("/projects/project-a/session-1.jsonl"),
            index: 0,
            timestamp: Local::now(),
            preview: "preview title".to_string(),
            preview_first: "preview title".to_string(),
            preview_last: "preview title".to_string(),
            full_text: String::new(),
            semantic_turns: vec![],
            search_text_lower: String::new(),
            project_name: Some("project-a".to_string()),
            project_path: None,
            cwd: None,
            message_count: 1,
            parse_errors: vec![],
            summary: None,
            custom_title: Some("custom title".to_string()),
            model: None,
            total_tokens: 0,
            duration_minutes: None,
        }
    }

    #[test]
    fn formats_hit_with_selected_conversation_metadata() {
        let conversation = conversation();
        let hit = SemanticHit {
            conversation_index: 0,
            session: "session-1".to_string(),
            semantic_score: 0.5,
            lexical_score: 0.2,
            hybrid_score: 0.7,
            snippet: "snippet text".to_string(),
        };

        let formatted = format_hit(1, &hit, &[&conversation]);

        assert!(formatted.contains("# 1 hybrid=0.700 semantic=0.500 lexical=0.200"));
        assert!(formatted.contains("project-a | session-1"));
        assert!(formatted.contains("custom title"));
        assert!(formatted.contains("snippet text"));
    }

    #[test]
    fn truncates_long_snippets() {
        let text = "a".repeat(300);
        let truncated = truncate(&text, 10);

        assert_eq!(truncated.chars().count(), 10);
        assert!(truncated.ends_with('…'));
    }
}
