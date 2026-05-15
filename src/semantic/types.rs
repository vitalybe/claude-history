use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub const DEFAULT_CHUNK_TARGET_CHARS: usize = 2_400;
pub const DEFAULT_CHUNK_OVERLAP_CHARS: usize = 300;
pub const DEFAULT_CHUNK_CONTEXT_TURNS: usize = 1;
pub const DEFAULT_EMBEDDING_BATCH_SIZE: usize = 32;
pub const CACHE_SCHEMA_VERSION: u32 = 2;
pub const MODEL_NAME: &str = "BGESmallENV15";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChunkConfig {
    pub target_chars: usize,
    pub overlap_chars: usize,
    pub context_turns: usize,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            target_chars: DEFAULT_CHUNK_TARGET_CHARS,
            overlap_chars: DEFAULT_CHUNK_OVERLAP_CHARS,
            context_turns: DEFAULT_CHUNK_CONTEXT_TURNS,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SemanticChunk {
    pub conversation_index: usize,
    pub session: String,
    pub chunk_index: usize,
    pub key: String,
    pub text: String,
    pub metadata: Option<FileMetadata>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EmbeddedChunk {
    pub conversation_index: usize,
    pub session: String,
    pub text: String,
    pub embedding: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SemanticHit {
    pub conversation_index: usize,
    pub session: String,
    pub semantic_score: f32,
    pub lexical_score: f32,
    pub hybrid_score: f32,
    pub snippet: String,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct FileMetadata {
    pub file_size: u64,
    pub mtime_secs: u64,
    pub mtime_nsecs: u32,
}

#[derive(Serialize, Deserialize)]
pub struct EmbeddingCache {
    pub schema_version: u32,
    pub model: String,
    pub chunk_target_chars: usize,
    pub chunk_overlap_chars: usize,
    pub chunk_context_turns: usize,
    pub entries: HashMap<String, CachedChunk>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CachedChunk {
    pub file_size: u64,
    pub mtime_secs: u64,
    pub mtime_nsecs: u32,
    pub text: String,
    pub embedding: Vec<f32>,
}
