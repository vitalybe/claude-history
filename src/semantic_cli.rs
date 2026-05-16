use crate::error::{AppError, Result};
use crate::history::Conversation;

pub fn run(query: &str, conversations: &[Conversation], top: usize, local: bool) -> Result<()> {
    use crate::semantic::cache::write_embedding_cache;
    use crate::semantic::fastembed::FastembedEmbedder;
    use crate::semantic::index::{SemanticIndexRequest, SemanticIndexState};
    use crate::semantic::output::format_hit;
    use crate::semantic::types::MODEL_NAME;

    let selected = select_conversations(conversations, local)?;
    if selected.is_empty() {
        eprintln!("{}", no_conversations_message(local));
        return Ok(());
    }

    let candidates = semantic_index_candidates(&selected);
    let request = SemanticIndexRequest {
        query,
        candidates: &candidates,
        prewarm: false,
    };
    let mut state = SemanticIndexState::new();
    if !state.has_chunks(&request) {
        state.clear_empty(&request);
        eprintln!("No visible dialogue text available for semantic search.");
        return Ok(());
    }

    let mut embedder = FastembedEmbedder::new()?;
    let (refresh, response) =
        refresh_and_rank_interactive(&request, &mut state, &mut embedder, write_embedding_cache)?;

    eprintln!(
        "Semantic search: searching {} cached chunk(s) from {} recent conversation(s) with fastembed {MODEL_NAME}",
        refresh.indexed_chunk_count,
        selected.len()
    );

    if !response.query_embedding_returned {
        eprintln!("No query embedding returned.");
        return Ok(());
    }

    for (rank, hit) in response.hits.iter().take(top).enumerate() {
        eprintln!("{}", format_hit(rank + 1, hit, &selected));
    }

    if response.hits.is_empty() {
        eprintln!("No semantic matches found.");
    }

    Ok(())
}

pub fn clear_cache() -> Result<()> {
    let cleared = crate::semantic::cache::clear_semantic_cache_files()?;
    if cleared {
        eprintln!("Semantic cache cleared.");
    } else {
        eprintln!("Semantic cache is already empty.");
    }
    Ok(())
}

pub fn generate_cache(conversations: &[Conversation], local: bool) -> Result<()> {
    use crate::semantic::cache::write_embedding_cache;
    use crate::semantic::chunk::build_chunks;
    use crate::semantic::fastembed::FastembedEmbedder;
    use crate::semantic::index::{SemanticIndexProgress, SemanticIndexRequest, SemanticIndexState};
    use crate::semantic::types::ChunkConfig;

    let selected = select_conversations(conversations, local)?;
    if selected.is_empty() {
        eprintln!("{}", no_conversations_message(local));
        return Ok(());
    }

    let chunk_count = build_chunks(&selected, ChunkConfig::default()).len();
    if chunk_count == 0 {
        eprintln!("No visible dialogue text available for semantic cache generation.");
        return Ok(());
    }

    let candidates = semantic_index_candidates(&selected);
    let request = SemanticIndexRequest {
        query: "",
        candidates: &candidates,
        prewarm: true,
    };
    let mut state = SemanticIndexState::new();
    eprintln!(
        "Semantic cache: checking {chunk_count} chunk(s) from {} recent conversation(s).",
        selected.len()
    );

    let mut embedder = FastembedEmbedder::new()?;
    let response = state.refresh_or_prewarm(
        &request,
        &mut embedder,
        |progress| {
            if let SemanticIndexProgress::Embedding { completed, total } = progress
                && total > 0
                && completed > 0
            {
                eprintln!("Semantic cache: embedded {completed}/{total} changed chunk(s)");
            }
        },
        write_embedding_cache,
    )?;
    write_embedding_cache(&state.cache);

    eprintln!(
        "Semantic cache: cached {} chunk(s) from {} recent conversation(s).",
        response.indexed_chunk_count.min(chunk_count),
        selected.len()
    );

    Ok(())
}

pub fn debug_search(query: &str, conversations: &[Conversation], local: bool) -> Result<()> {
    use crate::semantic::cache::{
        cache_entry_matches, cache_miss_count, cached_chunks, embedding_cache_file_path,
        read_embedding_cache,
    };
    use crate::semantic::chunk::build_chunks;
    use crate::semantic::embed::SemanticEmbedder;
    use crate::semantic::fastembed::FastembedEmbedder;
    use crate::semantic::output::format_hit;
    use crate::semantic::rank::rank_chunks;
    use crate::semantic::types::{ChunkConfig, MODEL_NAME};
    use std::collections::HashMap;

    let selected = select_conversations(conversations, local)?;
    eprintln!(
        "Semantic debug: selected {} conversation(s).",
        selected.len()
    );
    eprintln!(
        "Semantic debug: model cache: {}",
        FastembedEmbedder::cache_dir().display()
    );
    match embedding_cache_file_path() {
        Some(path) => eprintln!("Semantic debug: embedding cache: {}", path.display()),
        None => eprintln!("Semantic debug: embedding cache: unavailable"),
    }

    if selected.is_empty() {
        eprintln!("{}", no_conversations_message(local));
        return Ok(());
    }

    let turn_count = selected
        .iter()
        .map(|conversation| conversation.semantic_turns.len())
        .sum::<usize>();
    let chunk_config = ChunkConfig::default();
    let chunks = build_chunks(&selected, chunk_config);
    eprintln!(
        "Semantic debug: {} semantic turn(s), {} chunk(s), target={} overlap={} context_turns={}",
        turn_count,
        chunks.len(),
        chunk_config.target_chars,
        chunk_config.overlap_chars,
        chunk_config.context_turns
    );

    let mut chunk_counts = HashMap::<usize, usize>::new();
    for chunk in &chunks {
        *chunk_counts.entry(chunk.conversation_index).or_default() += 1;
    }
    let mut chunk_counts = chunk_counts.into_iter().collect::<Vec<_>>();
    chunk_counts.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    for (rank, (index, count)) in chunk_counts.iter().take(10).enumerate() {
        let conversation = selected[*index];
        let title = conversation
            .custom_title
            .as_deref()
            .or(conversation.summary.as_deref())
            .unwrap_or(&conversation.preview);
        eprintln!(
            "Semantic debug: chunk-heavy #{:2}: {} chunk(s) | {} | {}",
            rank + 1,
            count,
            conversation.path.display(),
            title
        );
    }

    let suspicious = chunks
        .iter()
        .filter(|chunk| {
            chunk.text.contains("```")
                || chunk.text.contains("<system-reminder>")
                || chunk.text.contains("<local-command-stdout>")
                || chunk.text.contains("<command-message>")
        })
        .take(5)
        .collect::<Vec<_>>();
    if suspicious.is_empty() {
        eprintln!(
            "Semantic debug: no sampled chunks contain code fences or command/system markers."
        );
    } else {
        for chunk in suspicious {
            eprintln!(
                "Semantic debug: suspicious chunk {}:{}: {}",
                chunk.session,
                chunk.chunk_index,
                truncate_debug(&chunk.text, 220)
            );
        }
    }

    if chunks.is_empty() {
        eprintln!("Semantic debug: no visible dialogue text available.");
        return Ok(());
    }

    let cache = read_embedding_cache(chunk_config);
    let missing = cache_miss_count(&chunks, &cache);
    let cached_count = chunks.len().saturating_sub(missing);
    eprintln!(
        "Semantic debug: cache entries={}, hits={}, misses={}",
        cache.entries.len(),
        cached_count,
        missing
    );
    for chunk in chunks
        .iter()
        .filter(|chunk| {
            cache
                .entries
                .get(&chunk.key)
                .is_none_or(|entry| !cache_entry_matches(entry, &chunk.text))
        })
        .take(5)
    {
        eprintln!(
            "Semantic debug: missing cache chunk {}:{}: {}",
            chunk.session,
            chunk.chunk_index,
            truncate_debug(&chunk.text, 180)
        );
    }

    let (embedded_chunks, _) = cached_chunks(chunks, &cache);
    if embedded_chunks.is_empty() {
        eprintln!("Semantic debug: no cached chunks available for ranking.");
        return Ok(());
    }

    let mut embedder = FastembedEmbedder::new()?;
    let Some(query_embedding) = embedder.embed_query(query)? else {
        eprintln!("Semantic debug: no query embedding returned.");
        return Ok(());
    };
    let hits = rank_chunks(query, &query_embedding, &embedded_chunks);
    eprintln!(
        "Semantic debug: ranked {} cached chunk(s) with fastembed {MODEL_NAME}.",
        embedded_chunks.len()
    );

    for (rank, hit) in hits.iter().take(20).enumerate() {
        eprintln!("{}", format_hit(rank + 1, hit, &selected));
    }
    if hits.is_empty() {
        eprintln!("Semantic debug: no semantic matches found.");
    }

    Ok(())
}

fn truncate_debug(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_owned();
    }

    let mut out: String = text.chars().take(max_chars.saturating_sub(1)).collect();
    out.push('…');
    out
}

fn refresh_and_rank_interactive(
    request: &crate::semantic::index::SemanticIndexRequest<'_>,
    state: &mut crate::semantic::index::SemanticIndexState,
    embedder: &mut dyn crate::semantic::embed::SemanticEmbedder,
    mut save_cache: impl FnMut(&crate::semantic::types::EmbeddingCache),
) -> Result<(
    crate::semantic::index::SemanticIndexResponse,
    crate::semantic::index::SemanticIndexResponse,
)> {
    let refresh = state.refresh_passages(request, embedder, |_| {}, |_| {})?;
    save_cache(&state.cache);
    let response = state.rank_refreshed(request, embedder, |_| {})?;
    Ok((refresh, response))
}

fn semantic_index_candidates(
    selected: &[&Conversation],
) -> Vec<crate::semantic::index::SemanticIndexCandidate> {
    selected
        .iter()
        .enumerate()
        .map(
            |(index, conversation)| crate::semantic::index::SemanticIndexCandidate {
                index,
                conversation: std::sync::Arc::new((*conversation).clone()),
            },
        )
        .collect()
}

fn select_conversations(conversations: &[Conversation], local: bool) -> Result<Vec<&Conversation>> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic::embed::SemanticEmbedder;
    use crate::semantic::types::{ChunkConfig, EmbeddingCache};
    use chrono::Local;
    use std::path::PathBuf;

    struct FakeEmbedder {
        passage_calls: usize,
        query_calls: usize,
        query_embedding: Option<Vec<f32>>,
        query_error: bool,
    }

    impl FakeEmbedder {
        fn new(query_embedding: Option<Vec<f32>>) -> Self {
            Self {
                passage_calls: 0,
                query_calls: 0,
                query_embedding,
                query_error: false,
            }
        }
    }

    impl SemanticEmbedder for FakeEmbedder {
        fn embed_passages(&mut self, passages: &[String]) -> Result<Vec<Vec<f32>>> {
            self.passage_calls += 1;
            Ok(passages
                .iter()
                .map(|passage| match passage.as_str() {
                    "visible beta" => vec![0.0, 1.0],
                    _ => vec![1.0, 0.0],
                })
                .collect())
        }

        fn embed_query(&mut self, query: &str) -> Result<Option<Vec<f32>>> {
            self.query_calls += 1;
            if self.query_error {
                return Err(AppError::ConfigError("query failed".to_string()));
            }
            Ok(if query.contains("beta") {
                Some(vec![0.0, 1.0])
            } else {
                self.query_embedding.clone()
            })
        }
    }

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
    fn interactive_refresh_writes_cache_once_before_query_ranking() {
        let conversations = vec![test_conversation(
            "/projects/project-a/session-1.jsonl",
            "one",
            vec!["visible alpha".to_string()],
        )];
        let selected = vec![&conversations[0]];
        let candidates = semantic_index_candidates(&selected);
        let query = "alpha".to_string();
        let request = crate::semantic::index::SemanticIndexRequest {
            query: &query,
            candidates: &candidates,
            prewarm: false,
        };
        let cache = crate::semantic::cache::empty_embedding_cache(ChunkConfig::default());
        let mut state =
            crate::semantic::index::SemanticIndexState::with_cache(ChunkConfig::default(), cache);
        let mut embedder = FakeEmbedder::new(None);
        let mut saved_entry_counts = Vec::new();

        let (refresh, response) = refresh_and_rank_interactive(
            &request,
            &mut state,
            &mut embedder,
            |cache: &EmbeddingCache| saved_entry_counts.push(cache.entries.len()),
        )
        .expect("interactive refresh and rank succeeds");

        assert_eq!(refresh.indexed_chunk_count, 1);
        assert_eq!(saved_entry_counts, vec![1]);
        assert!(!response.query_embedding_returned);
        assert_eq!(embedder.passage_calls, 1);
        assert_eq!(embedder.query_calls, 1);
    }

    #[test]
    fn interactive_refresh_writes_cache_before_query_error() {
        let conversations = vec![test_conversation(
            "/projects/project-a/session-1.jsonl",
            "one",
            vec!["visible alpha".to_string()],
        )];
        let selected = vec![&conversations[0]];
        let candidates = semantic_index_candidates(&selected);
        let query = "alpha".to_string();
        let request = crate::semantic::index::SemanticIndexRequest {
            query: &query,
            candidates: &candidates,
            prewarm: false,
        };
        let cache = crate::semantic::cache::empty_embedding_cache(ChunkConfig::default());
        let mut state =
            crate::semantic::index::SemanticIndexState::with_cache(ChunkConfig::default(), cache);
        let mut embedder = FakeEmbedder::new(Some(vec![1.0, 0.0]));
        embedder.query_error = true;
        let mut saved_entry_counts = Vec::new();

        let error = match refresh_and_rank_interactive(
            &request,
            &mut state,
            &mut embedder,
            |cache: &EmbeddingCache| saved_entry_counts.push(cache.entries.len()),
        ) {
            Ok(_) => panic!("query error should propagate"),
            Err(error) => error,
        };

        assert_eq!(saved_entry_counts, vec![1]);
        assert_eq!(embedder.passage_calls, 1);
        assert_eq!(embedder.query_calls, 1);
        assert!(error.to_string().contains("query failed"));
    }

    #[test]
    fn selection_includes_all_conversations() {
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

        let selected = select_conversations(&conversations, false).expect("select conversations");

        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].custom_title.as_deref(), Some("one"));
        assert_eq!(selected[1].custom_title.as_deref(), Some("two"));
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

        let selected = select_conversations(&conversations, true).expect("select conversations");
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
    fn semantic_index_candidates_use_selected_slice_indices() {
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
        let selected = vec![&conversations[1], &conversations[0]];

        let candidates = semantic_index_candidates(&selected);

        assert_eq!(candidates[0].index, 0);
        assert_eq!(
            candidates[0].conversation.custom_title.as_deref(),
            Some("two")
        );
        assert_eq!(candidates[1].index, 1);
        assert_eq!(
            candidates[1].conversation.custom_title.as_deref(),
            Some("one")
        );
    }

    #[test]
    fn empty_corpus_returns_before_model_initialization() {
        run("cache", &[], 1, false).expect("empty corpus returns");
    }
}
