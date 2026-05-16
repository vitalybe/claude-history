use crate::error::Result;
use crate::semantic::embed::SemanticEmbedder;
use crate::semantic::types::{
    CACHE_SCHEMA_VERSION, CachedChunk, ChunkConfig, EmbeddedChunk, EmbeddingCache, MODEL_NAME,
    SemanticChunk,
};
use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};

pub fn embed_chunks(
    embedder: &mut dyn SemanticEmbedder,
    chunks: Vec<SemanticChunk>,
    cache: &mut EmbeddingCache,
) -> Result<Vec<EmbeddedChunk>> {
    embed_chunks_with_progress(embedder, chunks, cache, |_, _| {})
}

pub fn embed_chunks_with_progress(
    embedder: &mut dyn SemanticEmbedder,
    chunks: Vec<SemanticChunk>,
    cache: &mut EmbeddingCache,
    progress: impl FnMut(usize, usize),
) -> Result<Vec<EmbeddedChunk>> {
    embed_chunks_with_progress_and_save(embedder, chunks, cache, progress, |_| {})
}

pub fn embed_chunks_with_progress_and_save(
    embedder: &mut dyn SemanticEmbedder,
    chunks: Vec<SemanticChunk>,
    cache: &mut EmbeddingCache,
    mut progress: impl FnMut(usize, usize),
    mut save: impl FnMut(&EmbeddingCache),
) -> Result<Vec<EmbeddedChunk>> {
    prune_stale_entries(cache, &chunks);

    let mut embedded = Vec::with_capacity(chunks.len());
    let mut misses = Vec::new();

    for chunk in chunks {
        let cached = cache
            .entries
            .get(&chunk.key)
            .filter(|entry| cache_entry_matches(entry, &chunk.text));

        if let Some(entry) = cached {
            embedded.push(EmbeddedChunk {
                conversation_index: chunk.conversation_index,
                session: chunk.session,
                chunk_index: chunk.chunk_index,
                text: entry.text.clone(),
                embedding: entry.embedding.clone(),
            });
        } else {
            misses.push(chunk);
        }
    }

    let total_misses = misses.len();
    let mut completed = 0;
    for batch in misses.chunks(32) {
        let texts = batch
            .iter()
            .map(|chunk| chunk.text.clone())
            .collect::<Vec<_>>();
        let embeddings = embedder.embed_passages(&texts)?;

        for (chunk, embedding) in batch.iter().cloned().zip(embeddings) {
            let metadata = chunk.metadata.unwrap_or_default();
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
            embedded.push(EmbeddedChunk {
                conversation_index: chunk.conversation_index,
                session: chunk.session,
                chunk_index: chunk.chunk_index,
                text: chunk.text,
                embedding,
            });
        }
        completed += batch.len();
        save(cache);
        progress(completed, total_misses);
    }

    Ok(embedded)
}

pub fn cached_chunks(
    chunks: Vec<SemanticChunk>,
    cache: &EmbeddingCache,
) -> (Vec<EmbeddedChunk>, usize) {
    let mut embedded = Vec::with_capacity(chunks.len());
    let mut misses = 0;

    for chunk in chunks {
        let cached = cache
            .entries
            .get(&chunk.key)
            .filter(|entry| cache_entry_matches(entry, &chunk.text));

        if let Some(entry) = cached {
            embedded.push(EmbeddedChunk {
                conversation_index: chunk.conversation_index,
                session: chunk.session,
                chunk_index: chunk.chunk_index,
                text: entry.text.clone(),
                embedding: entry.embedding.clone(),
            });
        } else {
            misses += 1;
        }
    }

    (embedded, misses)
}

pub fn cache_entry_matches(entry: &CachedChunk, text: &str) -> bool {
    entry.text == text
}

fn prune_stale_entries(cache: &mut EmbeddingCache, chunks: &[SemanticChunk]) {
    let current_keys = chunks
        .iter()
        .map(|chunk| chunk.key.clone())
        .collect::<HashSet<_>>();
    let selected_prefixes = chunks
        .iter()
        .map(|chunk| cache_key_prefix(&chunk.key).to_owned())
        .collect::<HashSet<_>>();

    cache.entries.retain(|key, _| {
        !selected_prefixes.contains(cache_key_prefix(key)) || current_keys.contains(key)
    });
}

fn cache_key_prefix(key: &str) -> &str {
    key.rsplit_once(':').map_or(key, |(prefix, _)| prefix)
}

pub fn read_embedding_cache(config: ChunkConfig) -> EmbeddingCache {
    let Some(path) = embedding_cache_path() else {
        return empty_embedding_cache(config);
    };
    read_embedding_cache_from_path(&path, config)
}

fn read_embedding_cache_from_path(path: &Path, config: ChunkConfig) -> EmbeddingCache {
    let Ok(data) = std::fs::read(path) else {
        return empty_embedding_cache(config);
    };
    let Ok(cache) = bincode::deserialize::<EmbeddingCache>(&data) else {
        return empty_embedding_cache(config);
    };
    if cache_matches_config(&cache, config) {
        cache
    } else {
        empty_embedding_cache(config)
    }
}

pub fn write_embedding_cache(cache: &EmbeddingCache) {
    let Some(path) = embedding_cache_path() else {
        return;
    };
    write_embedding_cache_to_path(cache, &path);
}

pub fn embedding_cache_file_path() -> Option<PathBuf> {
    embedding_cache_path()
}

pub fn clear_semantic_cache_files() -> std::io::Result<bool> {
    let Some(path) = semantic_cache_dir() else {
        return Ok(false);
    };
    if !path.exists() {
        return Ok(false);
    }
    std::fs::remove_dir_all(path)?;
    Ok(true)
}

fn write_embedding_cache_to_path(cache: &EmbeddingCache, path: &Path) {
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

pub fn empty_embedding_cache(config: ChunkConfig) -> EmbeddingCache {
    EmbeddingCache {
        schema_version: CACHE_SCHEMA_VERSION,
        model: MODEL_NAME.to_string(),
        chunk_target_chars: config.target_chars,
        chunk_overlap_chars: config.overlap_chars,
        chunk_context_turns: config.context_turns,
        entries: HashMap::new(),
    }
}

fn cache_matches_config(cache: &EmbeddingCache, config: ChunkConfig) -> bool {
    cache.schema_version == CACHE_SCHEMA_VERSION
        && cache.model == MODEL_NAME
        && cache.chunk_target_chars == config.target_chars
        && cache.chunk_overlap_chars == config.overlap_chars
        && cache.chunk_context_turns == config.context_turns
}

fn embedding_cache_path() -> Option<PathBuf> {
    semantic_cache_dir().map(|path| path.join("embeddings-v1.bin"))
}

fn semantic_cache_dir() -> Option<PathBuf> {
    home::home_dir().map(|home| {
        home.join(".cache")
            .join("claude-history")
            .join("semantic-poc")
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic::types::FileMetadata;

    struct FakeEmbedder {
        calls: usize,
    }

    impl SemanticEmbedder for FakeEmbedder {
        fn embed_passages(&mut self, passages: &[String]) -> Result<Vec<Vec<f32>>> {
            self.calls += 1;
            Ok(passages
                .iter()
                .map(|text| vec![text.len() as f32, 1.0])
                .collect())
        }

        fn embed_query(&mut self, _query: &str) -> Result<Option<Vec<f32>>> {
            Ok(Some(vec![1.0, 0.0]))
        }
    }

    fn metadata() -> FileMetadata {
        FileMetadata {
            file_size: 10,
            mtime_secs: 20,
            mtime_nsecs: 30,
        }
    }

    fn chunk(key: &str, text: &str) -> SemanticChunk {
        let chunk_index = key
            .rsplit_once(':')
            .and_then(|(_, index)| index.parse::<usize>().ok())
            .unwrap_or(0);
        SemanticChunk {
            conversation_index: 0,
            session: "session".to_string(),
            chunk_index,
            key: key.to_string(),
            text: text.to_string(),
            metadata: Some(metadata()),
        }
    }

    fn cached(text: &str) -> CachedChunk {
        CachedChunk {
            file_size: 10,
            mtime_secs: 20,
            mtime_nsecs: 30,
            text: text.to_string(),
            embedding: vec![0.5, 0.5],
        }
    }

    #[test]
    fn embed_chunks_reuses_matching_cache_entry() {
        let config = ChunkConfig::default();
        let mut cache = empty_embedding_cache(config);
        cache
            .entries
            .insert("session:0".to_string(), cached("cached text"));
        let mut embedder = FakeEmbedder { calls: 0 };

        let embedded = embed_chunks(
            &mut embedder,
            vec![chunk("session:0", "cached text")],
            &mut cache,
        )
        .expect("embedding succeeds");

        assert_eq!(embedder.calls, 0);
        assert_eq!(embedded[0].embedding, vec![0.5, 0.5]);
        assert_eq!(embedded[0].chunk_index, 0);
    }

    #[test]
    fn embed_chunks_embeds_cache_misses() {
        let config = ChunkConfig::default();
        let mut cache = empty_embedding_cache(config);
        let mut embedder = FakeEmbedder { calls: 0 };

        let embedded = embed_chunks(
            &mut embedder,
            vec![chunk("session:0", "new text")],
            &mut cache,
        )
        .expect("embedding succeeds");

        assert_eq!(embedder.calls, 1);
        assert_eq!(embedded[0].embedding, vec![8.0, 1.0]);
        assert_eq!(embedded[0].chunk_index, 0);
        assert!(cache.entries.contains_key("session:0"));
    }

    #[test]
    fn embed_chunks_replaces_changed_cache_entry() {
        let config = ChunkConfig::default();
        let mut cache = empty_embedding_cache(config);
        cache
            .entries
            .insert("session:0".to_string(), cached("old text"));
        let mut embedder = FakeEmbedder { calls: 0 };

        let embedded = embed_chunks(
            &mut embedder,
            vec![chunk("session:0", "new text")],
            &mut cache,
        )
        .expect("embedding succeeds");

        let restored = cache.entries.get("session:0").expect("cache entry");
        assert_eq!(embedder.calls, 1);
        assert_eq!(embedded[0].embedding, vec![8.0, 1.0]);
        assert_eq!(restored.text, "new text");
        assert_eq!(restored.embedding, vec![8.0, 1.0]);
    }

    #[test]
    fn embed_chunks_replaces_changed_cache_entry_without_metadata() {
        let config = ChunkConfig::default();
        let mut cache = empty_embedding_cache(config);
        cache
            .entries
            .insert("session:0".to_string(), cached("old text"));
        let mut current = chunk("session:0", "new text");
        current.metadata = None;
        let mut embedder = FakeEmbedder { calls: 0 };

        embed_chunks(&mut embedder, vec![current], &mut cache).expect("embedding succeeds");

        let restored = cache.entries.get("session:0").expect("cache entry");
        assert_eq!(embedder.calls, 1);
        assert_eq!(restored.text, "new text");
        assert_eq!(restored.embedding, vec![8.0, 1.0]);
    }

    #[test]
    fn embed_chunks_prunes_selected_stale_entries() {
        let config = ChunkConfig::default();
        let mut cache = empty_embedding_cache(config);
        cache
            .entries
            .insert("session:0".to_string(), cached("current"));
        cache
            .entries
            .insert("session:1".to_string(), cached("stale"));
        cache
            .entries
            .insert("other-session:0".to_string(), cached("unselected"));
        let mut embedder = FakeEmbedder { calls: 0 };

        embed_chunks(
            &mut embedder,
            vec![chunk("session:0", "current")],
            &mut cache,
        )
        .expect("embedding succeeds");

        assert!(cache.entries.contains_key("session:0"));
        assert!(!cache.entries.contains_key("session:1"));
        assert!(cache.entries.contains_key("other-session:0"));
    }

    #[test]
    fn cache_config_mismatch_invalidates_cache() {
        let config = ChunkConfig::default();
        let mut cache = empty_embedding_cache(config);
        cache.chunk_target_chars += 1;

        assert!(!cache_matches_config(&cache, config));
    }

    #[test]
    fn wrong_schema_cache_is_ignored() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("cache.bin");
        let config = ChunkConfig::default();
        let mut cache = empty_embedding_cache(config);
        cache.schema_version = CACHE_SCHEMA_VERSION - 1;
        cache
            .entries
            .insert("session:0".to_string(), cached("stale"));
        write_embedding_cache_to_path(&cache, &path);

        let restored = read_embedding_cache_from_path(&path, config);

        assert_eq!(restored.schema_version, CACHE_SCHEMA_VERSION);
        assert!(restored.entries.is_empty());
    }

    #[test]
    fn corrupt_cache_is_ignored() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("cache.bin");
        let config = ChunkConfig::default();
        std::fs::write(&path, b"not bincode").expect("write corrupt cache");

        let restored = read_embedding_cache_from_path(&path, config);

        assert_eq!(restored.schema_version, CACHE_SCHEMA_VERSION);
        assert!(restored.entries.is_empty());
    }

    #[test]
    fn mismatched_config_cache_is_ignored_when_read() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("cache.bin");
        let config = ChunkConfig::default();
        let mut cache = empty_embedding_cache(config);
        cache.chunk_overlap_chars += 1;
        cache
            .entries
            .insert("session:0".to_string(), cached("stale"));
        write_embedding_cache_to_path(&cache, &path);

        let restored = read_embedding_cache_from_path(&path, config);

        assert_eq!(restored.chunk_overlap_chars, config.overlap_chars);
        assert!(restored.entries.is_empty());
    }

    #[test]
    fn cache_round_trips_when_config_matches() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("cache.bin");
        let config = ChunkConfig::default();
        let mut cache = empty_embedding_cache(config);
        cache
            .entries
            .insert("session:0".to_string(), cached("cached text"));

        write_embedding_cache_to_path(&cache, &path);
        let restored = read_embedding_cache_from_path(&path, config);

        assert!(restored.entries.contains_key("session:0"));
        assert!(cache_entry_matches(
            restored.entries.get("session:0").unwrap(),
            "cached text"
        ));
    }
}
