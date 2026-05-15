use crate::error::{AppError, Result};
use crate::history::Conversation;

pub fn run(
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

    let selected = select_conversations(conversations, limit, local)?;
    if selected.is_empty() {
        eprintln!("{}", no_conversations_message(local));
        return Ok(());
    }

    let chunk_config = ChunkConfig::default();
    let chunks = build_chunks(&selected, chunk_config);
    if chunks.is_empty() {
        eprintln!("No visible dialogue text available for semantic search.");
        return Ok(());
    }

    let mut embedder = FastembedEmbedder::new(model_cache_dir())?;
    let mut cache = read_embedding_cache(chunk_config);
    let embedded_chunks = embed_chunks(&mut embedder, chunks, &mut cache)?;
    write_embedding_cache(&cache);

    eprintln!(
        "Semantic search: searching {} cached chunk(s) from {} recent conversation(s) with fastembed {MODEL_NAME}",
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

pub fn generate_cache(conversations: &[Conversation], limit: usize, local: bool) -> Result<()> {
    use crate::semantic::cache::{
        embed_chunks_with_progress_and_save, read_embedding_cache, write_embedding_cache,
    };
    use crate::semantic::chunk::build_chunks;
    use crate::semantic::fastembed::FastembedEmbedder;
    use crate::semantic::types::ChunkConfig;

    let selected = select_conversations(conversations, limit, local)?;
    if selected.is_empty() {
        eprintln!("{}", no_conversations_message(local));
        return Ok(());
    }

    let chunk_config = ChunkConfig::default();
    let chunks = build_chunks(&selected, chunk_config);
    if chunks.is_empty() {
        eprintln!("No visible dialogue text available for semantic cache generation.");
        return Ok(());
    }

    let chunk_count = chunks.len();
    let mut embedder = FastembedEmbedder::new(model_cache_dir())?;
    let mut cache = read_embedding_cache(chunk_config);
    eprintln!(
        "Semantic cache: checking {chunk_count} chunk(s) from {} recent conversation(s).",
        selected.len()
    );
    let embedded_chunks = embed_chunks_with_progress_and_save(
        &mut embedder,
        chunks,
        &mut cache,
        |done, total| {
            eprintln!("Semantic cache: embedded {done}/{total} changed chunk(s)");
        },
        write_embedding_cache,
    )?;
    write_embedding_cache(&cache);

    eprintln!(
        "Semantic cache: cached {} chunk(s) from {} recent conversation(s).",
        embedded_chunks.len().min(chunk_count),
        selected.len()
    );

    Ok(())
}

fn select_conversations(
    conversations: &[Conversation],
    limit: usize,
    local: bool,
) -> Result<Vec<&Conversation>> {
    let current_project_dir_name = if local {
        let dir = std::env::current_dir().map_err(AppError::Io)?;
        Some(crate::history::convert_path_to_project_dir_name(&dir))
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
    Ok(selected)
}

fn no_conversations_message(local: bool) -> &'static str {
    if local {
        "No conversations available for semantic search in the current workspace."
    } else {
        "No conversations available for semantic search."
    }
}

fn model_cache_dir() -> std::path::PathBuf {
    home::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".cache")
        .join("claude-history")
        .join("semantic-poc")
        .join("fastembed")
}

#[cfg(test)]
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

        let selected =
            select_conversations(&conversations, 1, false).expect("select conversations");

        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].custom_title.as_deref(), Some("one"));
    }

    #[test]
    fn selection_filters_to_current_workspace_when_local() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cwd = std::env::current_dir().expect("cwd");
        std::env::set_current_dir(dir.path()).expect("set cwd");
        let project = crate::history::convert_path_to_project_dir_name(
            &std::env::current_dir().expect("current temp cwd"),
        );
        let conversations = vec![
            test_conversation(
                &format!("projects/{project}/session-1.jsonl"),
                "local",
                vec!["local".to_string()],
            ),
            test_conversation(
                "projects/other-project/session-2.jsonl",
                "other",
                vec!["other".to_string()],
            ),
        ];

        let selected =
            select_conversations(&conversations, 10, true).expect("select conversations");
        std::env::set_current_dir(cwd).expect("restore cwd");

        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].custom_title.as_deref(), Some("local"));
    }

    #[test]
    fn empty_conversation_message_mentions_local_scope() {
        assert_eq!(
            no_conversations_message(true),
            "No conversations available for semantic search in the current workspace."
        );
        assert_eq!(
            no_conversations_message(false),
            "No conversations available for semantic search."
        );
    }

    #[test]
    fn empty_corpus_returns_before_model_initialization() {
        run("cache", &[], 1, 1, false).expect("empty corpus returns");
    }
}
