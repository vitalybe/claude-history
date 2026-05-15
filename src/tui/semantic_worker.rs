use crate::error::Result;
use crate::history::Conversation;
use crate::semantic::cache::{
    cache_entry_matches, embed_chunks_quiet, read_embedding_cache, write_embedding_cache,
};
use crate::semantic::chunk::build_chunks_with_indices;
use crate::semantic::embed::SemanticEmbedder;
use crate::semantic::fastembed::FastembedEmbedder;
use crate::semantic::rank::rank_chunks;
use crate::semantic::types::{ChunkConfig, EmbeddedChunk, EmbeddingCache};
use crate::tui::app::{SemanticProgress, SemanticResultMetadata};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc;

#[derive(Clone)]
pub struct SemanticSearchRequest {
    pub generation: u64,
    pub query: String,
    pub conversations: Arc<Vec<Conversation>>,
    pub candidate_indices: Vec<usize>,
}

pub enum SemanticSearchMessage {
    Progress {
        generation: u64,
        progress: SemanticProgress,
    },
    Complete(SemanticSearchResponse),
}

pub struct SemanticSearchResponse {
    pub generation: u64,
    pub filtered: Vec<usize>,
    pub metadata: HashMap<usize, SemanticResultMetadata>,
    pub error: Option<String>,
    pub progress: SemanticProgress,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SemanticIndexSignature {
    chunk_config: ChunkConfig,
    conversations: Vec<ConversationSignature>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ConversationSignature {
    index: usize,
    path: PathBuf,
    semantic_turns: Vec<String>,
}

struct SemanticWorkerState {
    signature: Option<SemanticIndexSignature>,
    embedded_chunks: Vec<EmbeddedChunk>,
    cache: EmbeddingCache,
    chunk_config: ChunkConfig,
}

impl SemanticWorkerState {
    fn new() -> Self {
        let chunk_config = ChunkConfig::default();
        Self {
            signature: None,
            embedded_chunks: Vec::new(),
            cache: read_embedding_cache(chunk_config),
            chunk_config,
        }
    }
}

pub fn spawn_semantic_worker() -> (
    mpsc::Sender<SemanticSearchRequest>,
    mpsc::Receiver<SemanticSearchMessage>,
) {
    let (cmd_tx, cmd_rx) = mpsc::channel::<SemanticSearchRequest>();
    let (res_tx, res_rx) = mpsc::channel::<SemanticSearchMessage>();

    std::thread::Builder::new()
        .name("semantic-search-worker".into())
        .spawn(move || {
            let mut state = SemanticWorkerState::new();
            let mut embedder: Option<FastembedEmbedder> = None;

            while let Ok(mut request) = cmd_rx.recv() {
                while let Ok(pending) = cmd_rx.try_recv() {
                    request = pending;
                }

                let next_signature = semantic_index_signature(&request, state.chunk_config);
                let has_chunks = semantic_index_has_chunks(
                    &request,
                    &state.signature,
                    &state.embedded_chunks,
                    state.chunk_config,
                );
                let response = if has_chunks {
                    if embedder.is_none() {
                        let _ = res_tx.send(SemanticSearchMessage::Progress {
                            generation: request.generation,
                            progress: SemanticProgress::InitializingModel,
                        });
                        embedder = match FastembedEmbedder::new_quiet(model_cache_dir()) {
                            Ok(embedder) => Some(embedder),
                            Err(error) => {
                                let _ = res_tx.send(SemanticSearchMessage::Complete(
                                    SemanticSearchResponse {
                                        generation: request.generation,
                                        filtered: Vec::new(),
                                        metadata: HashMap::new(),
                                        error: Some(error.to_string()),
                                        progress: SemanticProgress::Failed,
                                    },
                                ));
                                continue;
                            }
                        };
                    }
                    rank_semantic_request(
                        &request,
                        &mut state.signature,
                        &mut state.embedded_chunks,
                        &mut state.cache,
                        state.chunk_config,
                        embedder.as_mut().unwrap(),
                        &res_tx,
                    )
                    .unwrap_or_else(|error| SemanticSearchResponse {
                        generation: request.generation,
                        filtered: Vec::new(),
                        metadata: HashMap::new(),
                        error: Some(error.to_string()),
                        progress: SemanticProgress::Failed,
                    })
                } else {
                    state.signature = Some(next_signature);
                    state.embedded_chunks.clear();
                    SemanticSearchResponse {
                        generation: request.generation,
                        filtered: Vec::new(),
                        metadata: HashMap::new(),
                        error: None,
                        progress: SemanticProgress::EmptyCorpus,
                    }
                };

                let _ = res_tx.send(SemanticSearchMessage::Complete(response));
            }
        })
        .expect("failed to spawn semantic search worker thread");

    (cmd_tx, res_rx)
}

fn rank_semantic_request(
    request: &SemanticSearchRequest,
    signature: &mut Option<SemanticIndexSignature>,
    embedded_chunks: &mut Vec<EmbeddedChunk>,
    cache: &mut EmbeddingCache,
    chunk_config: ChunkConfig,
    embedder: &mut dyn SemanticEmbedder,
    res_tx: &mpsc::Sender<SemanticSearchMessage>,
) -> Result<SemanticSearchResponse> {
    let next_signature = semantic_index_signature(request, chunk_config);
    if signature.as_ref() != Some(&next_signature) {
        let chunks = semantic_chunks(request, chunk_config);

        if chunks.is_empty() {
            *signature = Some(next_signature);
            embedded_chunks.clear();
            return Ok(SemanticSearchResponse {
                generation: request.generation,
                filtered: Vec::new(),
                metadata: HashMap::new(),
                error: None,
                progress: SemanticProgress::EmptyCorpus,
            });
        }

        let miss_count = cache_miss_count(&chunks, cache);
        let progress = if miss_count == 0 {
            SemanticProgress::CacheReady
        } else {
            SemanticProgress::EmbeddingChangedChunks { count: miss_count }
        };
        let _ = res_tx.send(SemanticSearchMessage::Progress {
            generation: request.generation,
            progress,
        });
        *embedded_chunks = embed_chunks_quiet(embedder, chunks, cache)?;
        write_embedding_cache(cache);
        *signature = Some(next_signature);
    } else {
        let _ = res_tx.send(SemanticSearchMessage::Progress {
            generation: request.generation,
            progress: SemanticProgress::CacheReady,
        });
    }

    if embedded_chunks.is_empty() {
        return Ok(SemanticSearchResponse {
            generation: request.generation,
            filtered: Vec::new(),
            metadata: HashMap::new(),
            error: None,
            progress: SemanticProgress::EmptyCorpus,
        });
    }

    let _ = res_tx.send(SemanticSearchMessage::Progress {
        generation: request.generation,
        progress: SemanticProgress::Ranking,
    });
    let Some(query_embedding) = embedder.embed_query(&request.query)? else {
        return Ok(SemanticSearchResponse {
            generation: request.generation,
            filtered: Vec::new(),
            metadata: HashMap::new(),
            error: None,
            progress: SemanticProgress::EmptyCorpus,
        });
    };

    let hits = rank_chunks(&request.query, &query_embedding, embedded_chunks);
    let filtered = hits
        .iter()
        .map(|hit| hit.conversation_index)
        .collect::<Vec<_>>();
    let metadata = hits
        .into_iter()
        .map(|hit| {
            (
                hit.conversation_index,
                SemanticResultMetadata {
                    score: hit.hybrid_score,
                    snippet: hit.snippet,
                },
            )
        })
        .collect();

    let progress = if filtered.is_empty() {
        SemanticProgress::EmptyCorpus
    } else {
        SemanticProgress::Complete
    };

    Ok(SemanticSearchResponse {
        generation: request.generation,
        filtered,
        metadata,
        error: None,
        progress,
    })
}

fn cache_miss_count(
    chunks: &[crate::semantic::types::SemanticChunk],
    cache: &EmbeddingCache,
) -> usize {
    chunks
        .iter()
        .filter(|chunk| {
            cache
                .entries
                .get(&chunk.key)
                .is_none_or(|entry| !cache_entry_matches(entry, &chunk.text))
        })
        .count()
}

fn semantic_index_has_chunks(
    request: &SemanticSearchRequest,
    signature: &Option<SemanticIndexSignature>,
    embedded_chunks: &[EmbeddedChunk],
    chunk_config: ChunkConfig,
) -> bool {
    let next_signature = semantic_index_signature(request, chunk_config);
    if signature.as_ref() == Some(&next_signature) {
        return !embedded_chunks.is_empty();
    }
    !semantic_chunks(request, chunk_config).is_empty()
}

fn semantic_chunks(
    request: &SemanticSearchRequest,
    chunk_config: ChunkConfig,
) -> Vec<crate::semantic::types::SemanticChunk> {
    build_chunks_with_indices(
        request.candidate_indices.iter().filter_map(|&index| {
            request
                .conversations
                .get(index)
                .map(|conversation| (index, conversation))
        }),
        chunk_config,
    )
}

fn semantic_index_signature(
    request: &SemanticSearchRequest,
    chunk_config: ChunkConfig,
) -> SemanticIndexSignature {
    let conversations = request
        .candidate_indices
        .iter()
        .filter_map(|&index| {
            let conversation = request.conversations.get(index)?;
            Some(ConversationSignature {
                index,
                path: conversation.path.clone(),
                semantic_turns: conversation.semantic_turns.clone(),
            })
        })
        .collect();

    SemanticIndexSignature {
        chunk_config,
        conversations,
    }
}

fn model_cache_dir() -> PathBuf {
    home::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".cache")
        .join("claude-history")
        .join("semantic-poc")
        .join("fastembed")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic::cache::empty_embedding_cache;
    use crate::tui::search::normalize_for_search;
    use chrono::Local;
    use std::path::PathBuf;

    struct FakeEmbedder {
        passage_calls: usize,
        query_calls: usize,
        embedded_passages: Vec<Vec<String>>,
    }

    impl SemanticEmbedder for FakeEmbedder {
        fn embed_passages(&mut self, passages: &[String]) -> Result<Vec<Vec<f32>>> {
            self.passage_calls += 1;
            self.embedded_passages.push(passages.to_vec());
            Ok(passages
                .iter()
                .map(|passage| match passage.as_str() {
                    "visible alpha" => vec![1.0, 0.0],
                    "visible beta" => vec![0.0, 1.0],
                    _ => vec![0.5, 0.5],
                })
                .collect())
        }

        fn embed_query(&mut self, query: &str) -> Result<Option<Vec<f32>>> {
            self.query_calls += 1;
            Ok(Some(if query.contains("beta") {
                vec![0.0, 1.0]
            } else {
                vec![1.0, 0.0]
            }))
        }
    }

    fn conversation(path: &str, semantic_turns: Vec<&str>) -> Conversation {
        Conversation {
            path: PathBuf::from(path),
            index: 0,
            timestamp: Local::now(),
            preview: "preview sentinel".to_string(),
            preview_first: "preview sentinel".to_string(),
            preview_last: "preview sentinel".to_string(),
            full_text:
                "title sentinel summary sentinel cwd sentinel project sentinel tool output sentinel"
                    .to_string(),
            semantic_turns: semantic_turns.into_iter().map(str::to_string).collect(),
            search_text_lower: normalize_for_search(
                "title sentinel summary sentinel cwd sentinel project sentinel tool output sentinel",
            ),
            project_name: Some("project sentinel".to_string()),
            project_path: Some(PathBuf::from("/projects/project-sentinel")),
            cwd: Some(PathBuf::from("/cwd/sentinel")),
            message_count: 1,
            parse_errors: Vec::new(),
            summary: Some("summary sentinel".to_string()),
            custom_title: Some("title sentinel".to_string()),
            model: None,
            total_tokens: 0,
            duration_minutes: None,
        }
    }

    fn request(
        query: &str,
        conversations: Vec<Conversation>,
        candidate_indices: Vec<usize>,
    ) -> SemanticSearchRequest {
        SemanticSearchRequest {
            generation: 1,
            query: query.to_string(),
            conversations: Arc::new(conversations),
            candidate_indices,
        }
    }

    fn progress_tx() -> (
        mpsc::Sender<SemanticSearchMessage>,
        mpsc::Receiver<SemanticSearchMessage>,
    ) {
        mpsc::channel()
    }

    #[test]
    fn ranks_original_indices_and_records_metadata() {
        let conversations = vec![
            conversation("/projects/project-a/session-a.jsonl", vec!["visible alpha"]),
            conversation("/projects/project-a/session-b.jsonl", vec!["visible beta"]),
        ];
        let request = request("beta", conversations, vec![1, 0]);
        let mut signature = None;
        let mut embedded_chunks = Vec::new();
        let mut cache = empty_embedding_cache(ChunkConfig::default());
        let mut embedder = FakeEmbedder {
            passage_calls: 0,
            query_calls: 0,
            embedded_passages: Vec::new(),
        };

        let response = rank_semantic_request(
            &request,
            &mut signature,
            &mut embedded_chunks,
            &mut cache,
            ChunkConfig::default(),
            &mut embedder,
            &progress_tx().0,
        )
        .expect("rank succeeds");

        assert_eq!(response.filtered, vec![1, 0]);
        assert_eq!(response.metadata[&1].snippet, "visible beta");
        assert_eq!(response.metadata[&1].score, 1.2);
    }

    #[test]
    fn reuses_passage_embeddings_for_same_candidate_signature() {
        let conversations = vec![
            conversation("/projects/project-a/session-a.jsonl", vec!["visible alpha"]),
            conversation("/projects/project-a/session-b.jsonl", vec!["visible beta"]),
        ];
        let mut request = request("alpha", conversations, vec![0, 1]);
        let mut signature = None;
        let mut embedded_chunks = Vec::new();
        let mut cache = empty_embedding_cache(ChunkConfig::default());
        let mut embedder = FakeEmbedder {
            passage_calls: 0,
            query_calls: 0,
            embedded_passages: Vec::new(),
        };

        rank_semantic_request(
            &request,
            &mut signature,
            &mut embedded_chunks,
            &mut cache,
            ChunkConfig::default(),
            &mut embedder,
            &progress_tx().0,
        )
        .expect("first rank succeeds");
        request.query = "beta".to_string();
        rank_semantic_request(
            &request,
            &mut signature,
            &mut embedded_chunks,
            &mut cache,
            ChunkConfig::default(),
            &mut embedder,
            &progress_tx().0,
        )
        .expect("second rank succeeds");

        assert_eq!(embedder.passage_calls, 1);
        assert_eq!(embedder.query_calls, 2);
    }

    #[test]
    fn snippets_and_embeddings_use_only_semantic_turns() {
        let conversations = vec![conversation(
            "/projects/project-a/session-a.jsonl",
            vec!["visible alpha"],
        )];
        let request = request("alpha", conversations, vec![0]);
        let mut signature = None;
        let mut embedded_chunks = Vec::new();
        let mut cache = empty_embedding_cache(ChunkConfig::default());
        let mut embedder = FakeEmbedder {
            passage_calls: 0,
            query_calls: 0,
            embedded_passages: Vec::new(),
        };

        let response = rank_semantic_request(
            &request,
            &mut signature,
            &mut embedded_chunks,
            &mut cache,
            ChunkConfig::default(),
            &mut embedder,
            &progress_tx().0,
        )
        .expect("rank succeeds");

        assert_eq!(
            embedder.embedded_passages,
            vec![vec!["visible alpha".to_string()]]
        );
        assert_eq!(response.metadata[&0].snippet, "visible alpha");
        assert!(!response.metadata[&0].snippet.contains("sentinel"));
    }

    #[test]
    fn ranking_reports_embedding_then_ranking_progress() {
        let conversations = vec![conversation(
            "/projects/project-a/session-a.jsonl",
            vec!["visible alpha"],
        )];
        let request = request("alpha", conversations, vec![0]);
        let mut signature = None;
        let mut embedded_chunks = Vec::new();
        let mut cache = empty_embedding_cache(ChunkConfig::default());
        let mut embedder = FakeEmbedder {
            passage_calls: 0,
            query_calls: 0,
            embedded_passages: Vec::new(),
        };
        let (tx, rx) = progress_tx();

        let response = rank_semantic_request(
            &request,
            &mut signature,
            &mut embedded_chunks,
            &mut cache,
            ChunkConfig::default(),
            &mut embedder,
            &tx,
        )
        .expect("rank succeeds");
        let progress = rx.try_iter().collect::<Vec<_>>();

        assert!(matches!(
            progress.as_slice(),
            [
                SemanticSearchMessage::Progress {
                    progress: SemanticProgress::EmbeddingChangedChunks { count: 1 },
                    ..
                },
                SemanticSearchMessage::Progress {
                    progress: SemanticProgress::Ranking,
                    ..
                }
            ]
        ));
        assert_eq!(response.progress, SemanticProgress::Complete);
    }

    #[test]
    fn cached_signature_reports_cache_ready_before_ranking() {
        let conversations = vec![conversation(
            "/projects/project-a/session-a.jsonl",
            vec!["visible alpha"],
        )];
        let request = request("alpha", conversations, vec![0]);
        let mut signature = None;
        let mut embedded_chunks = Vec::new();
        let mut cache = empty_embedding_cache(ChunkConfig::default());
        let mut embedder = FakeEmbedder {
            passage_calls: 0,
            query_calls: 0,
            embedded_passages: Vec::new(),
        };
        let (tx, _rx) = progress_tx();
        rank_semantic_request(
            &request,
            &mut signature,
            &mut embedded_chunks,
            &mut cache,
            ChunkConfig::default(),
            &mut embedder,
            &tx,
        )
        .expect("first rank succeeds");
        let (tx, rx) = progress_tx();

        rank_semantic_request(
            &request,
            &mut signature,
            &mut embedded_chunks,
            &mut cache,
            ChunkConfig::default(),
            &mut embedder,
            &tx,
        )
        .expect("second rank succeeds");
        let progress = rx.try_iter().collect::<Vec<_>>();

        assert!(matches!(
            progress.as_slice(),
            [
                SemanticSearchMessage::Progress {
                    progress: SemanticProgress::CacheReady,
                    ..
                },
                SemanticSearchMessage::Progress {
                    progress: SemanticProgress::Ranking,
                    ..
                }
            ]
        ));
        assert_eq!(embedder.passage_calls, 1);
    }

    #[test]
    fn empty_visible_dialogue_returns_before_embedder_initialization() {
        let conversations = vec![conversation("/projects/project-a/session-a.jsonl", vec![])];
        let request = request("alpha", conversations, vec![0]);
        let mut signature = None;
        let mut embedded_chunks = Vec::new();
        let mut cache = empty_embedding_cache(ChunkConfig::default());
        let mut embedder = FakeEmbedder {
            passage_calls: 0,
            query_calls: 0,
            embedded_passages: Vec::new(),
        };

        let response = rank_semantic_request(
            &request,
            &mut signature,
            &mut embedded_chunks,
            &mut cache,
            ChunkConfig::default(),
            &mut embedder,
            &progress_tx().0,
        )
        .expect("empty corpus succeeds");

        assert!(response.filtered.is_empty());
        assert!(response.metadata.is_empty());
        assert_eq!(embedder.passage_calls, 0);
        assert_eq!(embedder.query_calls, 0);
        assert!(!semantic_index_has_chunks(
            &request,
            &signature,
            &embedded_chunks,
            ChunkConfig::default()
        ));
    }
}
