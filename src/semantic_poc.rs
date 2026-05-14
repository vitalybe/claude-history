use crate::error::{AppError, Result};
#[cfg(feature = "semantic-poc")]
use crate::history;
use crate::history::Conversation;
#[cfg(feature = "semantic-poc")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "semantic-poc")]
use std::collections::HashMap;
#[cfg(feature = "semantic-poc")]
use std::io::Write;
#[cfg(feature = "semantic-poc")]
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(feature = "semantic-poc")]
const CHUNK_TARGET_CHARS: usize = 2_400;
#[cfg(feature = "semantic-poc")]
const CHUNK_OVERLAP_CHARS: usize = 300;
#[cfg(feature = "semantic-poc")]
const CHUNK_CONTEXT_TURNS: usize = 1;
#[cfg(feature = "semantic-poc")]
const EMBEDDING_BATCH_SIZE: usize = 32;
#[cfg(feature = "semantic-poc")]
const CACHE_SCHEMA_VERSION: u32 = 1;
#[cfg(feature = "semantic-poc")]
const MODEL_NAME: &str = "BGESmallENV15";

#[cfg(feature = "semantic-poc")]
struct SemanticChunk<'a> {
    conversation: &'a Conversation,
    key: String,
    text: String,
}

#[cfg(feature = "semantic-poc")]
struct EmbeddedChunk<'a> {
    conversation: &'a Conversation,
    text: String,
    embedding: Vec<f32>,
}

#[cfg(feature = "semantic-poc")]
#[derive(Serialize, Deserialize)]
struct EmbeddingCache {
    schema_version: u32,
    model: String,
    chunk_target_chars: usize,
    chunk_overlap_chars: usize,
    chunk_context_turns: usize,
    entries: HashMap<String, CachedChunk>,
}

#[cfg(feature = "semantic-poc")]
#[derive(Serialize, Deserialize)]
struct CachedChunk {
    file_size: u64,
    mtime_secs: u64,
    mtime_nsecs: u32,
    text: String,
    embedding: Vec<f32>,
}

#[cfg(feature = "semantic-poc")]
struct ConversationHit<'a> {
    conversation: &'a Conversation,
    semantic_score: f32,
    lexical_score: f32,
    hybrid_score: f32,
    snippet: String,
}

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
    use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};

    let selected = select_conversations(conversations, limit, local);
    if selected.is_empty() {
        eprintln!("No conversations available for semantic search.");
        return Ok(());
    }

    let chunks = build_chunks(&selected);
    if chunks.is_empty() {
        eprintln!("No conversation text available for semantic search.");
        return Ok(());
    }

    let mut model = TextEmbedding::try_new(
        TextInitOptions::new(EmbeddingModel::BGESmallENV15)
            .with_cache_dir(model_cache_dir())
            .with_show_download_progress(true),
    )
    .map_err(to_config_error)?;

    let mut cache = read_embedding_cache();
    let embedded_chunks = embed_chunks(&mut model, chunks, &mut cache)?;
    write_embedding_cache(&cache);

    eprintln!(
        "Semantic POC: searching {} cached chunk(s) from {} recent conversation(s) with fastembed {MODEL_NAME}",
        embedded_chunks.len(),
        selected.len()
    );

    let query_embeddings = model
        .embed(vec![format!("query: {query}")], Some(EMBEDDING_BATCH_SIZE))
        .map_err(to_config_error)?;
    let Some(query_embedding) = query_embeddings.first() else {
        eprintln!("No query embedding returned.");
        return Ok(());
    };

    let mut best_by_session: HashMap<String, ConversationHit<'_>> = HashMap::new();
    for chunk in &embedded_chunks {
        let embedding = &chunk.embedding;
        let semantic_score = cosine(query_embedding, embedding);
        let lexical_score = lexical_overlap(query, &chunk.text);
        let hybrid_score = semantic_score + lexical_score;
        let session = chunk
            .conversation
            .path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .to_owned();

        let replace = best_by_session
            .get(&session)
            .is_none_or(|existing| hybrid_score > existing.hybrid_score);
        if replace {
            best_by_session.insert(
                session,
                ConversationHit {
                    conversation: chunk.conversation,
                    semantic_score,
                    lexical_score,
                    hybrid_score,
                    snippet: chunk.text.clone(),
                },
            );
        }
    }

    let mut hits: Vec<_> = best_by_session.into_values().collect();
    hits.sort_by(|a, b| b.hybrid_score.total_cmp(&a.hybrid_score));

    for (rank, hit) in hits.iter().take(top).enumerate() {
        print_hit(rank + 1, hit);
    }

    if hits.is_empty() {
        eprintln!("No semantic matches found.");
    }

    Ok(())
}

#[cfg(feature = "semantic-poc")]
fn to_config_error(err: impl std::fmt::Display) -> AppError {
    AppError::ConfigError(err.to_string())
}

#[cfg(feature = "semantic-poc")]
fn embed_chunks<'a>(
    model: &mut fastembed::TextEmbedding,
    chunks: Vec<SemanticChunk<'a>>,
    cache: &mut EmbeddingCache,
) -> Result<Vec<EmbeddedChunk<'a>>> {
    let mut embedded = Vec::with_capacity(chunks.len());
    let mut misses = Vec::new();

    for chunk in chunks {
        let metadata = file_metadata(chunk.conversation);
        let cached = cache
            .entries
            .get(&chunk.key)
            .filter(|entry| cache_entry_matches(entry, &chunk.text));

        if let Some(entry) = cached {
            embedded.push(EmbeddedChunk {
                conversation: chunk.conversation,
                text: entry.text.clone(),
                embedding: entry.embedding.clone(),
            });
        } else {
            misses.push((chunk, metadata));
        }
    }

    if !misses.is_empty() {
        eprintln!("Semantic POC: embedding {} changed chunk(s)", misses.len());
        let texts = misses
            .iter()
            .map(|(chunk, _)| format!("passage: {}", chunk.text))
            .collect::<Vec<_>>();
        let embeddings = model
            .embed(texts, Some(EMBEDDING_BATCH_SIZE))
            .map_err(to_config_error)?;

        for ((chunk, metadata), embedding) in misses.into_iter().zip(embeddings) {
            if let Some(metadata) = metadata {
                cache.entries.insert(
                    chunk.key,
                    CachedChunk {
                        file_size: metadata.file_size,
                        mtime_secs: metadata.mtime_secs,
                        mtime_nsecs: metadata.mtime_nsecs,
                        text: chunk.text.clone(),
                        embedding: embedding.clone(),
                    },
                );
            }
            embedded.push(EmbeddedChunk {
                conversation: chunk.conversation,
                text: chunk.text,
                embedding,
            });
        }
    }

    Ok(embedded)
}

#[cfg(feature = "semantic-poc")]
struct FileMetadata {
    file_size: u64,
    mtime_secs: u64,
    mtime_nsecs: u32,
}

#[cfg(feature = "semantic-poc")]
fn file_metadata(conversation: &Conversation) -> Option<FileMetadata> {
    let metadata = std::fs::metadata(&conversation.path).ok()?;
    let modified = metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH);
    let duration_since_epoch = modified.duration_since(UNIX_EPOCH).unwrap_or_default();
    Some(FileMetadata {
        file_size: metadata.len(),
        mtime_secs: duration_since_epoch.as_secs(),
        mtime_nsecs: duration_since_epoch.subsec_nanos(),
    })
}

#[cfg(feature = "semantic-poc")]
fn cache_entry_matches(entry: &CachedChunk, text: &str) -> bool {
    entry.text == text
}

#[cfg(feature = "semantic-poc")]
fn read_embedding_cache() -> EmbeddingCache {
    let Some(path) = embedding_cache_path() else {
        return empty_embedding_cache();
    };
    let Ok(data) = std::fs::read(path) else {
        return empty_embedding_cache();
    };
    let Ok(cache) = bincode::deserialize::<EmbeddingCache>(&data) else {
        return empty_embedding_cache();
    };
    if cache.schema_version == CACHE_SCHEMA_VERSION
        && cache.model == MODEL_NAME
        && cache.chunk_target_chars == CHUNK_TARGET_CHARS
        && cache.chunk_overlap_chars == CHUNK_OVERLAP_CHARS
        && cache.chunk_context_turns == CHUNK_CONTEXT_TURNS
    {
        cache
    } else {
        empty_embedding_cache()
    }
}

#[cfg(feature = "semantic-poc")]
fn write_embedding_cache(cache: &EmbeddingCache) {
    let Some(path) = embedding_cache_path() else {
        return;
    };
    let Some(parent) = path.parent() else {
        return;
    };
    if std::fs::create_dir_all(parent).is_err() {
        return;
    }
    let Ok(data) = bincode::serialize(cache) else {
        return;
    };
    let Ok(mut tmp) = tempfile::NamedTempFile::new_in(parent) else {
        return;
    };
    if tmp.write_all(&data).is_err() {
        return;
    }
    let _ = tmp.persist(path);
}

#[cfg(feature = "semantic-poc")]
fn empty_embedding_cache() -> EmbeddingCache {
    EmbeddingCache {
        schema_version: CACHE_SCHEMA_VERSION,
        model: MODEL_NAME.to_string(),
        chunk_target_chars: CHUNK_TARGET_CHARS,
        chunk_overlap_chars: CHUNK_OVERLAP_CHARS,
        chunk_context_turns: CHUNK_CONTEXT_TURNS,
        entries: HashMap::new(),
    }
}

#[cfg(feature = "semantic-poc")]
fn embedding_cache_path() -> Option<std::path::PathBuf> {
    home::home_dir().map(|home| {
        home.join(".cache")
            .join("claude-history")
            .join("semantic-poc")
            .join("embeddings-v1.bin")
    })
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

#[cfg(feature = "semantic-poc")]
fn select_conversations(
    conversations: &[Conversation],
    limit: usize,
    local: bool,
) -> Vec<&Conversation> {
    let current_project_dir_name = if local {
        std::env::current_dir()
            .ok()
            .map(|dir| history::convert_path_to_project_dir_name(&dir))
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
                .is_some_and(|name| history::is_same_project(&name.to_string_lossy(), project));
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
fn build_chunks<'a>(conversations: &[&'a Conversation]) -> Vec<SemanticChunk<'a>> {
    let mut chunks = Vec::new();
    for conversation in conversations {
        let prefix = metadata_prefix(conversation);
        let semantic_turns = if conversation.semantic_turns.is_empty() {
            vec![conversation.full_text.as_str()]
        } else {
            conversation
                .semantic_turns
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>()
        };

        for (chunk_index, chunk) in group_turns(&semantic_turns).into_iter().enumerate() {
            push_chunk(&mut chunks, conversation, chunk_index, &prefix, &chunk);
        }
    }
    chunks
}

#[cfg(feature = "semantic-poc")]
fn group_turns(turns: &[&str]) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();

    for (index, turn) in turns.iter().enumerate() {
        let turn = turn.trim();
        if turn.is_empty() {
            continue;
        }

        if turn.len() > CHUNK_TARGET_CHARS {
            flush_chunk(&mut chunks, &mut current);
            split_long_text(turn, &mut chunks);
            continue;
        }

        let separator_len = if current.is_empty() { 0 } else { 2 };
        if !current.is_empty() && current.len() + separator_len + turn.len() > CHUNK_TARGET_CHARS {
            flush_chunk(&mut chunks, &mut current);
            append_context(turns, index, &mut current);
        }

        if !current.is_empty() {
            current.push_str("\n\n");
        }
        current.push_str(turn);
    }

    flush_chunk(&mut chunks, &mut current);
    chunks
}

#[cfg(feature = "semantic-poc")]
fn append_context(turns: &[&str], index: usize, current: &mut String) {
    let start = index.saturating_sub(CHUNK_CONTEXT_TURNS);
    for turn in &turns[start..index] {
        let turn = turn.trim();
        if turn.is_empty() || turn.len() + current.len() > CHUNK_OVERLAP_CHARS {
            continue;
        }
        if !current.is_empty() {
            current.push_str("\n\n");
        }
        current.push_str(turn);
    }
}

#[cfg(feature = "semantic-poc")]
fn flush_chunk(chunks: &mut Vec<String>, current: &mut String) {
    if !current.trim().is_empty() {
        chunks.push(std::mem::take(current));
    }
}

#[cfg(feature = "semantic-poc")]
fn split_long_text(mut text: &str, chunks: &mut Vec<String>) {
    while !text.is_empty() {
        let (chunk, rest) = split_chunk(text);
        chunks.push(chunk.to_owned());
        text = rest;
    }
}

#[cfg(feature = "semantic-poc")]
fn push_chunk<'a>(
    chunks: &mut Vec<SemanticChunk<'a>>,
    conversation: &'a Conversation,
    chunk_index: usize,
    prefix: &str,
    chunk: &str,
) {
    let text = if prefix.is_empty() {
        chunk.to_owned()
    } else {
        format!("{prefix}\n\n{chunk}")
    };
    let text = normalize_snippet(&text);
    if !text.is_empty() {
        let session = conversation
            .path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("?");
        let key = format!("{session}:{chunk_index}");
        chunks.push(SemanticChunk {
            conversation,
            key,
            text,
        });
    }
}

#[cfg(feature = "semantic-poc")]
fn metadata_prefix(conversation: &Conversation) -> String {
    let mut parts = Vec::new();
    if let Some(title) = &conversation.custom_title {
        parts.push(format!("Title: {title}"));
    }
    if let Some(summary) = &conversation.summary {
        parts.push(format!("Summary: {summary}"));
    }
    if let Some(project) = &conversation.project_name {
        parts.push(format!("Project: {project}"));
    }
    if let Some(cwd) = &conversation.cwd {
        parts.push(format!("Working directory: {}", cwd.display()));
    }
    parts.join("\n")
}

#[cfg(feature = "semantic-poc")]
fn split_chunk(text: &str) -> (&str, &str) {
    if text.len() <= CHUNK_TARGET_CHARS {
        return (text, "");
    }

    let end = floor_char_boundary(text, CHUNK_TARGET_CHARS);
    let chunk = &text[..end];
    let next_start = end.saturating_sub(CHUNK_OVERLAP_CHARS);
    let next_start = floor_char_boundary(text, next_start);
    (chunk, text[next_start..].trim_start())
}

#[cfg(feature = "semantic-poc")]
fn floor_char_boundary(text: &str, index: usize) -> usize {
    let mut index = index.min(text.len());
    while index > 0 && !text.is_char_boundary(index) {
        index -= 1;
    }
    index
}

#[cfg(feature = "semantic-poc")]
fn normalize_snippet(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(feature = "semantic-poc")]
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a.sqrt() * norm_b.sqrt())
    }
}

#[cfg(feature = "semantic-poc")]
fn lexical_overlap(query: &str, text: &str) -> f32 {
    let query_words = query
        .split_whitespace()
        .map(|word| word.to_lowercase())
        .collect::<Vec<_>>();
    if query_words.is_empty() {
        return 0.0;
    }

    let text = text.to_lowercase();
    let matches = query_words
        .iter()
        .filter(|word| text.contains(word.as_str()))
        .count();
    0.2 * matches as f32 / query_words.len() as f32
}

#[cfg(feature = "semantic-poc")]
fn print_hit(rank: usize, hit: &ConversationHit<'_>) {
    let project = hit.conversation.project_name.as_deref().unwrap_or("(none)");
    let session = hit
        .conversation
        .path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("?");
    let title = hit
        .conversation
        .custom_title
        .as_deref()
        .or(hit.conversation.summary.as_deref())
        .unwrap_or(&hit.conversation.preview);

    eprintln!(
        "#{rank:2} hybrid={:.3} semantic={:.3} lexical={:.3} | {project} | {session}\n     {title}\n     {}\n",
        hit.hybrid_score,
        hit.semantic_score,
        hit.lexical_score,
        truncate(&hit.snippet, 260)
    );
}

#[cfg(feature = "semantic-poc")]
fn truncate(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_owned();
    }

    let mut out: String = text.chars().take(max_chars.saturating_sub(1)).collect();
    out.push('…');
    out
}
