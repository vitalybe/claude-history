#[cfg(not(feature = "semantic-poc"))]
use crate::error::AppError;
use crate::error::Result;
use crate::history::Conversation;

pub fn run(
    query: &str,
    conversations: &[Conversation],
    top: usize,
    limit: usize,
    local: bool,
) -> Result<()> {
    run_impl(query, conversations, top, limit, local)
}

#[cfg(not(feature = "semantic-poc"))]
fn run_impl(
    _query: &str,
    _conversations: &[Conversation],
    _top: usize,
    _limit: usize,
    _local: bool,
) -> Result<()> {
    Err(AppError::ConfigError(
        "semantic search POC requires building with `--features semantic-poc`".to_string(),
    ))
}

#[cfg(feature = "semantic-poc")]
fn run_impl(
    query: &str,
    conversations: &[Conversation],
    top: usize,
    limit: usize,
    local: bool,
) -> Result<()> {
    use crate::semantic::cache::{embed_chunks, read_embedding_cache, write_embedding_cache};
    use crate::semantic::chunk::build_chunks;
    use crate::semantic::embed::SemanticEmbedder;
    use crate::semantic::fastembed::FastembedEmbedder;
    use crate::semantic::output::format_hit;
    use crate::semantic::rank::rank_chunks;
    use crate::semantic::types::{ChunkConfig, MODEL_NAME};

    let selected = select_conversations(conversations, limit, local);
    if selected.is_empty() {
        eprintln!("No conversations available for semantic search.");
        return Ok(());
    }

    let chunk_config = ChunkConfig::default();
    let chunks = build_chunks(&selected, chunk_config);
    if chunks.is_empty() {
        eprintln!("No conversation text available for semantic search.");
        return Ok(());
    }

    let mut embedder = FastembedEmbedder::new(model_cache_dir())?;
    let mut cache = read_embedding_cache(chunk_config);
    let embedded_chunks = embed_chunks(&mut embedder, chunks, &mut cache)?;
    write_embedding_cache(&cache);

    eprintln!(
        "Semantic POC: searching {} cached chunk(s) from {} recent conversation(s) with fastembed {MODEL_NAME}",
        embedded_chunks.len(),
        selected.len()
    );

    let Some(query_embedding) = embedder.embed_query(query)? else {
        eprintln!("No query embedding returned.");
        return Ok(());
    };
    let hits = rank_chunks(query, &query_embedding, &embedded_chunks);

    for (rank, hit) in hits.iter().take(top).enumerate() {
        eprintln!("{}", format_hit(rank + 1, hit, &selected));
    }

    if hits.is_empty() {
        eprintln!("No semantic matches found.");
    }

    Ok(())
}

#[cfg(feature = "semantic-poc")]
fn select_conversations(
    conversations: &[Conversation],
    limit: usize,
    local: bool,
) -> Vec<&Conversation> {
    let current_project_dir_name = if local {
        std::env::current_dir()
            .ok()
            .map(|dir| crate::history::convert_path_to_project_dir_name(&dir))
    } else {
        None
    };

    let mut selected = Vec::new();
    for conversation in conversations {
        if let Some(ref project) = current_project_dir_name {
            let matches = conversation
                .path
                .parent()
                .and_then(|p| p.file_name())
                .is_some_and(|name| {
                    crate::history::is_same_project(&name.to_string_lossy(), project)
                });
            if !matches {
                continue;
            }
        }

        selected.push(conversation);
        if selected.len() >= limit {
            break;
        }
    }
    selected
}

#[cfg(feature = "semantic-poc")]
fn model_cache_dir() -> std::path::PathBuf {
    home::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".cache")
        .join("claude-history")
        .join("semantic-poc")
        .join("fastembed")
}

#[cfg(all(test, feature = "semantic-poc"))]
mod tests {
    use super::*;
    use chrono::Local;
    use std::path::PathBuf;

    fn test_conversation(path: &str, title: &str, semantic_turns: Vec<String>) -> Conversation {
        Conversation {
            path: PathBuf::from(path),
            index: 0,
            timestamp: Local::now(),
            preview: title.to_string(),
            preview_first: title.to_string(),
            preview_last: title.to_string(),
            full_text: title.to_string(),
            semantic_turns,
            search_text_lower: title.to_string(),
            project_name: Some("project-a".to_string()),
            project_path: Some(PathBuf::from("/projects/project-a")),
            cwd: None,
            message_count: 1,
            parse_errors: vec![],
            summary: None,
            custom_title: Some(title.to_string()),
            model: None,
            total_tokens: 0,
            duration_minutes: None,
        }
    }

    #[test]
    fn selection_applies_limit_before_chunking() {
        let conversations = vec![
            test_conversation(
                "/projects/project-a/session-1.jsonl",
                "one",
                vec!["one".to_string()],
            ),
            test_conversation(
                "/projects/project-a/session-2.jsonl",
                "two",
                vec!["two".to_string()],
            ),
        ];

        let selected = select_conversations(&conversations, 1, false);

        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].custom_title.as_deref(), Some("one"));
    }
}
