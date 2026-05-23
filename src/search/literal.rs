use crate::history::Conversation;
use rayon::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaseMode {
    Sensitive,
    Insensitive,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Literal {
    text: String,
    case_mode: CaseMode,
}

impl Literal {
    pub fn new(text: String) -> Self {
        let case_mode = if text.chars().any(char::is_uppercase) {
            CaseMode::Sensitive
        } else {
            CaseMode::Insensitive
        };
        Self { text, case_mode }
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn case_mode(&self) -> CaseMode {
        self.case_mode
    }

    pub fn matches(&self, text: &str) -> bool {
        match self.case_mode {
            CaseMode::Sensitive => text.contains(&self.text),
            CaseMode::Insensitive => text.to_lowercase().contains(&self.text.to_lowercase()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LiteralCorpusEntry {
    pub index: usize,
    pub text: String,
}

pub fn build_literal_corpus(conversations: &[Conversation]) -> Vec<LiteralCorpusEntry> {
    conversations
        .par_iter()
        .enumerate()
        .map(|(index, conversation)| LiteralCorpusEntry {
            index,
            text: literal_text(conversation),
        })
        .collect()
}

pub fn matches_all_literals(text: &str, literals: &[Literal]) -> bool {
    literals.iter().all(|literal| literal.matches(text))
}

pub fn exact_fallback(
    conversations: &[Conversation],
    corpus: &[LiteralCorpusEntry],
    literals: &[Literal],
    scope: impl Fn(usize) -> bool + Sync,
) -> Vec<usize> {
    if literals.is_empty() {
        return Vec::new();
    }

    let mut matches = corpus
        .par_iter()
        .filter(|entry| scope(entry.index) && matches_all_literals(&entry.text, literals))
        .map(|entry| (entry.index, conversations[entry.index].timestamp))
        .collect::<Vec<_>>();

    matches.sort_unstable_by(|a, b| b.1.cmp(&a.1));
    matches.into_iter().map(|(index, _)| index).collect()
}

fn literal_text(conversation: &Conversation) -> String {
    let mut text = String::new();
    push_part(&mut text, Some(&conversation.full_text));
    push_part(&mut text, conversation.project_name.as_deref());
    text
}

fn push_part(text: &mut String, part: Option<&str>) {
    let Some(part) = part else {
        return;
    };
    if part.is_empty() {
        return;
    }
    if !text.is_empty() {
        text.push(' ');
    }
    text.push_str(part);
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, Local};
    use std::path::PathBuf;

    fn make_conv_full(
        text: &str,
        project: Option<&str>,
        title: Option<&str>,
        summary: Option<&str>,
        timestamp: chrono::DateTime<Local>,
    ) -> Conversation {
        let mut full_text = text.to_string();
        if let Some(summary) = summary {
            full_text = format!("{} {}", summary, full_text);
        }
        if let Some(title) = title {
            full_text = format!("{} {}", title, full_text);
        }

        Conversation {
            path: PathBuf::new(),
            index: 0,
            timestamp,
            preview: text.to_string(),
            preview_first: text.to_string(),
            preview_last: text.to_string(),
            full_text: full_text.clone(),
            semantic_turns: vec![text.to_string()],
            search_text_lower: crate::search::normalize_for_search(&full_text),
            project_name: project.map(str::to_string),
            project_path: None,
            cwd: None,
            message_count: 1,
            parse_errors: vec![],
            summary: summary.map(str::to_string),
            custom_title: title.map(str::to_string),
            model: None,
            total_tokens: 0,
            duration_minutes: None,
        }
    }

    #[test]
    fn literal_uses_smart_case() {
        assert!(Literal::new("exact_phrase".to_string()).matches("EXACT_PHRASE"));
        assert!(!Literal::new("Exact_Phrase".to_string()).matches("exact_phrase"));
        assert!(Literal::new("Exact_Phrase".to_string()).matches("Exact_Phrase"));
    }

    #[test]
    fn corpus_preserves_punctuation_and_metadata() {
        let now = Local::now();
        let conversations = vec![make_conv_full(
            "body_with_under_score and punctuation: yes",
            Some("project_name/raw-value"),
            Some("Title: Raw_Value"),
            Some("Summary.with punctuation"),
            now,
        )];

        let corpus = build_literal_corpus(&conversations);

        assert!(corpus[0].text.contains("body_with_under_score"));
        assert!(corpus[0].text.contains("punctuation: yes"));
        assert!(corpus[0].text.contains("project_name/raw-value"));
        assert!(corpus[0].text.contains("Title: Raw_Value"));
        assert!(corpus[0].text.contains("Summary.with punctuation"));
    }

    #[test]
    fn corpus_does_not_duplicate_title_and_summary() {
        let now = Local::now();
        let conversations = vec![make_conv_full(
            "body sentinel",
            None,
            Some("TitleA"),
            Some("SummaryB"),
            now,
        )];

        let corpus = build_literal_corpus(&conversations);

        assert!(!corpus[0].text.contains("SummaryB TitleA"));
    }

    #[test]
    fn exact_fallback_returns_scoped_matches_newest_first() {
        let now = Local::now();
        let conversations = vec![
            make_conv_full("needle phrase", None, None, None, now - Duration::days(1)),
            make_conv_full("needle phrase", None, None, None, now),
            make_conv_full("needle phrase", None, None, None, now - Duration::hours(1)),
        ];
        let corpus = build_literal_corpus(&conversations);
        let literal = Literal::new("needle phrase".to_string());

        let results = exact_fallback(&conversations, &corpus, &[literal], |index| index != 1);

        assert_eq!(results, vec![2, 0]);
    }

    #[test]
    fn exact_fallback_requires_all_literals() {
        let now = Local::now();
        let conversations = vec![
            make_conv_full("alpha beta", None, None, None, now),
            make_conv_full("alpha gamma", None, None, None, now),
        ];
        let corpus = build_literal_corpus(&conversations);
        let literals = vec![
            Literal::new("alpha".to_string()),
            Literal::new("beta".to_string()),
        ];

        let results = exact_fallback(&conversations, &corpus, &literals, |_| true);

        assert_eq!(results, vec![0]);
    }
}
