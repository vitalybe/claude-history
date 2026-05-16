use crate::error::Result;
use crate::history::Conversation;
use crate::semantic::cache::{
    cache_miss_count, embed_chunks_with_progress_and_save, read_embedding_cache,
};
use crate::semantic::chunk::build_chunks_with_indices;
use crate::semantic::embed::SemanticEmbedder;
use crate::semantic::rank::rank_chunks;
use crate::semantic::types::{
    ChunkConfig, EmbeddedChunk, EmbeddingCache, SemanticChunk, SemanticHit,
};
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Clone)]
pub struct SemanticIndexCandidate {
    pub index: usize,
    pub conversation: Arc<Conversation>,
}

pub struct SemanticIndexRequest<'a> {
    pub query: &'a str,
    pub candidates: &'a [SemanticIndexCandidate],
    pub prewarm: bool,
}

pub struct SemanticIndexResponse {
    pub hits: Vec<SemanticHit>,
    pub indexed_chunk_count: usize,
    pub query_embedding_returned: bool,
    pub progress: SemanticIndexProgress,
    pub prewarm: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SemanticIndexProgress {
    Embedding { completed: usize, total: usize },
    CacheReady,
    Ranking,
    Complete,
    EmptyCorpus,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SemanticIndexSignature {
    chunk_config: ChunkConfig,
    conversations: Vec<ConversationSignature>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ConversationSignature {
    index: usize,
    path: PathBuf,
    semantic_turns: Vec<String>,
}

pub struct SemanticIndexState {
    signature: Option<SemanticIndexSignature>,
    embedded_chunks: Vec<EmbeddedChunk>,
    pub cache: EmbeddingCache,
    pub chunk_config: ChunkConfig,
}

impl SemanticIndexState {
    pub fn new() -> Self {
        Self::with_chunk_config(ChunkConfig::default())
    }

    pub fn with_chunk_config(chunk_config: ChunkConfig) -> Self {
        Self {
            signature: None,
            embedded_chunks: Vec::new(),
            cache: read_embedding_cache(chunk_config),
            chunk_config,
        }
    }

    pub fn has_chunks(&self, request: &SemanticIndexRequest<'_>) -> bool {
        let next_signature = semantic_index_signature(request, self.chunk_config);
        if self.signature.as_ref() == Some(&next_signature) {
            return !self.embedded_chunks.is_empty();
        }
        !semantic_chunks(request, self.chunk_config).is_empty()
    }

    pub fn clear_empty(&mut self, request: &SemanticIndexRequest<'_>) {
        self.signature = Some(semantic_index_signature(request, self.chunk_config));
        self.embedded_chunks.clear();
    }

    pub fn refresh_passages(
        &mut self,
        request: &SemanticIndexRequest<'_>,
        embedder: &mut dyn SemanticEmbedder,
        mut progress: impl FnMut(SemanticIndexProgress),
        mut save_cache: impl FnMut(&EmbeddingCache),
    ) -> Result<SemanticIndexResponse> {
        let next_signature = semantic_index_signature(request, self.chunk_config);
        if self.signature.as_ref() != Some(&next_signature) {
            let chunks = semantic_chunks(request, self.chunk_config);

            if chunks.is_empty() {
                self.signature = Some(next_signature);
                self.embedded_chunks.clear();
                return Ok(SemanticIndexResponse {
                    hits: Vec::new(),
                    indexed_chunk_count: 0,
                    query_embedding_returned: true,
                    progress: SemanticIndexProgress::EmptyCorpus,
                    prewarm: request.prewarm,
                });
            }

            let miss_count = cache_miss_count(&chunks, &self.cache);
            progress(if miss_count > 0 {
                SemanticIndexProgress::Embedding {
                    completed: 0,
                    total: miss_count,
                }
            } else {
                SemanticIndexProgress::CacheReady
            });
            self.embedded_chunks = embed_chunks_with_progress_and_save(
                embedder,
                chunks,
                &mut self.cache,
                |completed, total| {
                    progress(SemanticIndexProgress::Embedding { completed, total });
                },
                &mut save_cache,
            )?;
            self.signature = Some(next_signature);
        } else {
            progress(SemanticIndexProgress::CacheReady);
        }

        Ok(SemanticIndexResponse {
            hits: Vec::new(),
            indexed_chunk_count: self.embedded_chunks.len(),
            query_embedding_returned: true,
            progress: if self.embedded_chunks.is_empty() {
                SemanticIndexProgress::EmptyCorpus
            } else {
                SemanticIndexProgress::CacheReady
            },
            prewarm: request.prewarm,
        })
    }

    pub fn rank_refreshed(
        &self,
        request: &SemanticIndexRequest<'_>,
        embedder: &mut dyn SemanticEmbedder,
        mut progress: impl FnMut(SemanticIndexProgress),
    ) -> Result<SemanticIndexResponse> {
        if self.embedded_chunks.is_empty() || request.prewarm {
            return Ok(SemanticIndexResponse {
                hits: Vec::new(),
                indexed_chunk_count: self.embedded_chunks.len(),
                query_embedding_returned: true,
                progress: if self.embedded_chunks.is_empty() {
                    SemanticIndexProgress::EmptyCorpus
                } else {
                    SemanticIndexProgress::CacheReady
                },
                prewarm: request.prewarm,
            });
        }

        progress(SemanticIndexProgress::Ranking);
        let Some(query_embedding) = embedder.embed_query(request.query)? else {
            return Ok(SemanticIndexResponse {
                hits: Vec::new(),
                indexed_chunk_count: self.embedded_chunks.len(),
                query_embedding_returned: false,
                progress: SemanticIndexProgress::EmptyCorpus,
                prewarm: request.prewarm,
            });
        };

        let hits = rank_chunks(request.query, &query_embedding, &self.embedded_chunks);
        let progress = if hits.is_empty() {
            SemanticIndexProgress::EmptyCorpus
        } else {
            SemanticIndexProgress::Complete
        };

        Ok(SemanticIndexResponse {
            hits,
            indexed_chunk_count: self.embedded_chunks.len(),
            query_embedding_returned: true,
            progress,
            prewarm: request.prewarm,
        })
    }

    pub fn refresh_or_prewarm(
        &mut self,
        request: &SemanticIndexRequest<'_>,
        embedder: &mut dyn SemanticEmbedder,
        mut progress: impl FnMut(SemanticIndexProgress),
        save_cache: impl FnMut(&EmbeddingCache),
    ) -> Result<SemanticIndexResponse> {
        let response = self.refresh_passages(request, embedder, &mut progress, save_cache)?;
        if response.progress == SemanticIndexProgress::EmptyCorpus || request.prewarm {
            return Ok(response);
        }
        self.rank_refreshed(request, embedder, progress)
    }

    #[cfg(test)]
    pub(crate) fn with_cache(chunk_config: ChunkConfig, cache: EmbeddingCache) -> Self {
        Self {
            signature: None,
            embedded_chunks: Vec::new(),
            cache,
            chunk_config,
        }
    }
}

impl Default for SemanticIndexState {
    fn default() -> Self {
        Self::new()
    }
}

fn semantic_chunks(
    request: &SemanticIndexRequest<'_>,
    chunk_config: ChunkConfig,
) -> Vec<SemanticChunk> {
    build_chunks_with_indices(
        request
            .candidates
            .iter()
            .map(|candidate| (candidate.index, candidate.conversation.as_ref())),
        chunk_config,
    )
}

fn semantic_index_signature(
    request: &SemanticIndexRequest<'_>,
    chunk_config: ChunkConfig,
) -> SemanticIndexSignature {
    let conversations = request
        .candidates
        .iter()
        .map(|candidate| ConversationSignature {
            index: candidate.index,
            path: candidate.conversation.path.clone(),
            semantic_turns: candidate.conversation.semantic_turns.clone(),
        })
        .collect();

    SemanticIndexSignature {
        chunk_config,
        conversations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic::cache::{cache_miss_count, empty_embedding_cache};
    use crate::semantic::types::{CachedChunk, SemanticQuality, SemanticRationaleKind};
    use chrono::Local;
    use std::path::PathBuf;

    struct FakeEmbedder {
        passage_calls: usize,
        query_calls: usize,
        embedded_passages: Vec<Vec<String>>,
        query_embedding: Option<Vec<f32>>,
    }

    impl FakeEmbedder {
        fn new() -> Self {
            Self {
                passage_calls: 0,
                query_calls: 0,
                embedded_passages: Vec::new(),
                query_embedding: Some(vec![1.0, 0.0]),
            }
        }
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
            Ok(if query.contains("beta") {
                Some(vec![0.0, 1.0])
            } else {
                self.query_embedding.clone()
            })
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
            search_text_lower:
                "title sentinel summary sentinel cwd sentinel project sentinel tool output sentinel"
                    .to_string(),
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
    ) -> (String, Vec<SemanticIndexCandidate>) {
        let candidates = candidate_indices
            .into_iter()
            .map(|index| SemanticIndexCandidate {
                index,
                conversation: Arc::new(conversations[index].clone()),
            })
            .collect();
        (query.to_string(), candidates)
    }

    fn index_request<'a>(
        query: &'a str,
        candidates: &'a [SemanticIndexCandidate],
    ) -> SemanticIndexRequest<'a> {
        SemanticIndexRequest {
            query,
            candidates,
            prewarm: false,
        }
    }

    fn cache_passage(cache: &mut EmbeddingCache, key: String, text: String, embedding: Vec<f32>) {
        cache.entries.insert(
            key,
            CachedChunk {
                file_size: 0,
                mtime_secs: 0,
                mtime_nsecs: 0,
                text,
                embedding,
            },
        );
    }

    fn cache_request_passages(cache: &mut EmbeddingCache, request: &SemanticIndexRequest<'_>) {
        for chunk in semantic_chunks(request, ChunkConfig::default()) {
            let embedding = match chunk.text.as_str() {
                "visible beta" => vec![0.0, 1.0],
                "visible alpha" => vec![1.0, 0.0],
                _ => vec![0.5, 0.5],
            };
            cache_passage(cache, chunk.key, chunk.text, embedding);
        }
    }

    #[test]
    fn ranks_original_indices_and_records_hits() {
        let conversations = vec![
            conversation("/projects/project-a/session-a.jsonl", vec!["visible alpha"]),
            conversation("/projects/project-a/session-b.jsonl", vec!["visible beta"]),
        ];
        let (query, candidates) = request("beta", conversations, vec![1, 0]);
        let request = index_request(&query, &candidates);
        let mut cache = empty_embedding_cache(ChunkConfig::default());
        cache_request_passages(&mut cache, &request);
        let mut state = SemanticIndexState::with_cache(ChunkConfig::default(), cache);
        let mut embedder = FakeEmbedder::new();

        let response = state
            .refresh_or_prewarm(&request, &mut embedder, |_| {}, |_| {})
            .expect("rank succeeds");

        let filtered = response
            .hits
            .iter()
            .map(|hit| hit.conversation_index)
            .collect::<Vec<_>>();
        assert_eq!(filtered, vec![1, 0]);
        let metadata = response
            .hits
            .iter()
            .find(|hit| hit.conversation_index == 1)
            .expect("beta hit");
        assert_eq!(metadata.score_breakdown.hybrid, 1.2);
        assert_eq!(metadata.score_breakdown.semantic, 1.0);
        assert_eq!(metadata.score_breakdown.lexical, 0.2);
        assert_eq!(metadata.explanation.quality, SemanticQuality::Strong);
        assert_eq!(metadata.explanation.quality_label, "strong");
        assert_eq!(
            metadata.explanation.rationale_kind,
            SemanticRationaleKind::LexicalBoosted
        );
        assert_eq!(metadata.explanation.evidence_preview, "visible beta");
        assert_eq!(metadata.explanation.matched_terms, vec!["beta"]);
        assert_eq!(metadata.explanation.chunk.conversation_index, 1);
        assert_eq!(metadata.explanation.chunk.session, "session-b");
        assert_eq!(metadata.explanation.chunk.chunk_index, 0);
    }

    #[test]
    fn reuses_passage_embeddings_for_same_candidate_signature() {
        let conversations = vec![
            conversation("/projects/project-a/session-a.jsonl", vec!["visible alpha"]),
            conversation("/projects/project-a/session-b.jsonl", vec!["visible beta"]),
        ];
        let (mut query, candidates) = request("alpha", conversations, vec![0, 1]);
        let mut cache = empty_embedding_cache(ChunkConfig::default());
        cache_request_passages(&mut cache, &index_request(&query, &candidates));
        let mut state = SemanticIndexState::with_cache(ChunkConfig::default(), cache);
        let mut embedder = FakeEmbedder::new();

        state
            .refresh_or_prewarm(
                &index_request(&query, &candidates),
                &mut embedder,
                |_| {},
                |_| {},
            )
            .expect("first rank succeeds");
        query = "beta".to_string();
        state
            .refresh_or_prewarm(
                &index_request(&query, &candidates),
                &mut embedder,
                |_| {},
                |_| {},
            )
            .expect("second rank succeeds");

        assert_eq!(embedder.passage_calls, 0);
        assert_eq!(embedder.query_calls, 2);
    }

    #[test]
    fn unchanged_signature_reuses_embeddings_until_semantic_turns_change() {
        let (mut query, mut candidates) = request(
            "alpha",
            vec![conversation(
                "/projects/project-a/session-a.jsonl",
                vec!["visible alpha"],
            )],
            vec![0],
        );
        let mut cache = empty_embedding_cache(ChunkConfig::default());
        cache_request_passages(&mut cache, &index_request(&query, &candidates));
        let mut state = SemanticIndexState::with_cache(ChunkConfig::default(), cache);
        let mut embedder = FakeEmbedder::new();

        state
            .refresh_or_prewarm(
                &index_request(&query, &candidates),
                &mut embedder,
                |_| {},
                |_| {},
            )
            .expect("first rank succeeds");
        query = "beta".to_string();
        state
            .refresh_or_prewarm(
                &index_request(&query, &candidates),
                &mut embedder,
                |_| {},
                |_| {},
            )
            .expect("same signature rank succeeds");
        candidates = vec![SemanticIndexCandidate {
            index: 0,
            conversation: Arc::new(conversation(
                "/projects/project-a/session-a.jsonl",
                vec!["visible beta"],
            )),
        }];
        cache_request_passages(&mut state.cache, &index_request(&query, &candidates));
        state
            .refresh_or_prewarm(
                &index_request(&query, &candidates),
                &mut embedder,
                |_| {},
                |_| {},
            )
            .expect("changed signature rank succeeds");

        assert_eq!(embedder.passage_calls, 0);
        assert_eq!(embedder.query_calls, 3);
        assert!(embedder.embedded_passages.is_empty());
    }

    #[test]
    fn snippets_and_embeddings_use_only_semantic_turns() {
        let conversations = vec![conversation(
            "/projects/project-a/session-a.jsonl",
            vec!["visible alpha"],
        )];
        let (query, candidates) = request("alpha", conversations, vec![0]);
        let request = index_request(&query, &candidates);
        let mut cache = empty_embedding_cache(ChunkConfig::default());
        cache_request_passages(&mut cache, &request);
        let mut state = SemanticIndexState::with_cache(ChunkConfig::default(), cache);
        let mut embedder = FakeEmbedder::new();

        let response = state
            .refresh_or_prewarm(&request, &mut embedder, |_| {}, |_| {})
            .expect("rank succeeds");

        assert!(embedder.embedded_passages.is_empty());
        assert_eq!(
            response.hits[0].explanation.evidence_preview,
            "visible alpha"
        );
        assert!(
            !response.hits[0]
                .explanation
                .evidence_preview
                .contains("sentinel")
        );
    }

    #[test]
    fn missing_cached_passages_are_embedded_and_ranked() {
        let conversations = vec![conversation(
            "/projects/project-a/session-a.jsonl",
            vec!["visible alpha"],
        )];
        let (query, candidates) = request("alpha", conversations, vec![0]);
        let request = index_request(&query, &candidates);
        let cache = empty_embedding_cache(ChunkConfig::default());
        let mut state = SemanticIndexState::with_cache(ChunkConfig::default(), cache);
        let mut embedder = FakeEmbedder::new();
        let mut save_calls = 0;
        let mut progress = Vec::new();

        let response = state
            .refresh_or_prewarm(
                &request,
                &mut embedder,
                |status| progress.push(status),
                |_| save_calls += 1,
            )
            .expect("missing cache embeds and ranks");

        assert_eq!(response.hits[0].conversation_index, 0);
        assert_eq!(response.progress, SemanticIndexProgress::Complete);
        assert_eq!(embedder.passage_calls, 1);
        assert_eq!(embedder.query_calls, 1);
        assert_eq!(save_calls, 1);
        assert_eq!(
            cache_miss_count(
                &semantic_chunks(&request, ChunkConfig::default()),
                &state.cache
            ),
            0
        );
        assert!(progress.contains(&SemanticIndexProgress::Embedding {
            completed: 0,
            total: 1,
        }));
    }

    #[test]
    fn partial_cached_passages_embed_missing_chunks_and_rank_all() {
        let conversations = vec![
            conversation("/projects/project-a/session-a.jsonl", vec!["visible alpha"]),
            conversation("/projects/project-a/session-b.jsonl", vec!["visible beta"]),
        ];
        let (query, candidates) = request("alpha", conversations, vec![0, 1]);
        let request = index_request(&query, &candidates);
        let mut cache = empty_embedding_cache(ChunkConfig::default());
        let first_chunk = semantic_chunks(&request, ChunkConfig::default())
            .into_iter()
            .find(|chunk| chunk.text == "visible alpha")
            .expect("alpha chunk");
        cache_passage(
            &mut cache,
            first_chunk.key,
            first_chunk.text,
            vec![1.0, 0.0],
        );
        let mut state = SemanticIndexState::with_cache(ChunkConfig::default(), cache);
        let mut embedder = FakeEmbedder::new();
        let mut progress = Vec::new();

        let response = state
            .refresh_or_prewarm(
                &request,
                &mut embedder,
                |status| progress.push(status),
                |_| {},
            )
            .expect("partial cache embeds misses and ranks all chunks");

        let filtered = response
            .hits
            .iter()
            .map(|hit| hit.conversation_index)
            .collect::<Vec<_>>();
        assert_eq!(filtered, vec![0, 1]);
        assert_eq!(response.progress, SemanticIndexProgress::Complete);
        assert_eq!(embedder.passage_calls, 1);
        assert_eq!(embedder.query_calls, 1);
        assert_eq!(
            embedder.embedded_passages,
            vec![vec!["visible beta".to_string()]]
        );
        assert!(progress.contains(&SemanticIndexProgress::Embedding {
            completed: 0,
            total: 1,
        }));
    }

    #[test]
    fn refresh_passages_can_skip_per_batch_cache_saves() {
        let conversations = vec![conversation(
            "/projects/project-a/session-a.jsonl",
            vec!["visible alpha"],
        )];
        let (query, candidates) = request("alpha", conversations, vec![0]);
        let request = index_request(&query, &candidates);
        let cache = empty_embedding_cache(ChunkConfig::default());
        let mut state = SemanticIndexState::with_cache(ChunkConfig::default(), cache);
        let mut embedder = FakeEmbedder::new();

        let response = state
            .refresh_passages(&request, &mut embedder, |_| {}, |_| {})
            .expect("refresh succeeds");

        assert_eq!(response.indexed_chunk_count, 1);
        assert_eq!(response.progress, SemanticIndexProgress::CacheReady);
        assert_eq!(embedder.passage_calls, 1);
        assert_eq!(embedder.query_calls, 0);
        assert_eq!(
            cache_miss_count(
                &semantic_chunks(&request, ChunkConfig::default()),
                &state.cache
            ),
            0
        );
    }

    #[test]
    fn rank_refreshed_index_reports_missing_query_embedding() {
        let conversations = vec![conversation(
            "/projects/project-a/session-a.jsonl",
            vec!["visible alpha"],
        )];
        let (query, candidates) = request("alpha", conversations, vec![0]);
        let request = index_request(&query, &candidates);
        let cache = empty_embedding_cache(ChunkConfig::default());
        let mut state = SemanticIndexState::with_cache(ChunkConfig::default(), cache);
        let mut embedder = FakeEmbedder::new();
        state
            .refresh_passages(&request, &mut embedder, |_| {}, |_| {})
            .expect("refresh succeeds");
        embedder.query_embedding = None;

        let response = state
            .rank_refreshed(&request, &mut embedder, |_| {})
            .expect("rank succeeds");

        assert!(response.hits.is_empty());
        assert!(!response.query_embedding_returned);
        assert_eq!(response.indexed_chunk_count, 1);
        assert_eq!(embedder.passage_calls, 1);
        assert_eq!(embedder.query_calls, 1);
    }

    #[test]
    fn prewarm_request_builds_cache_without_ranking_query() {
        let conversations = vec![conversation(
            "/projects/project-a/session-a.jsonl",
            vec!["visible alpha"],
        )];
        let (query, candidates) = request("", conversations, vec![0]);
        let request = SemanticIndexRequest {
            query: &query,
            candidates: &candidates,
            prewarm: true,
        };
        let cache = empty_embedding_cache(ChunkConfig::default());
        let mut state = SemanticIndexState::with_cache(ChunkConfig::default(), cache);
        let mut embedder = FakeEmbedder::new();

        let response = state
            .refresh_or_prewarm(&request, &mut embedder, |_| {}, |_| {})
            .expect("prewarm succeeds");

        assert!(response.hits.is_empty());
        assert_eq!(response.indexed_chunk_count, 1);
        assert_eq!(response.progress, SemanticIndexProgress::CacheReady);
        assert_eq!(embedder.passage_calls, 1);
        assert_eq!(embedder.query_calls, 0);
        assert_eq!(
            cache_miss_count(
                &semantic_chunks(&request, ChunkConfig::default()),
                &state.cache
            ),
            0
        );
    }

    #[test]
    fn reports_indexed_chunk_count() {
        let conversations = vec![
            conversation("/projects/project-a/session-a.jsonl", vec!["visible alpha"]),
            conversation("/projects/project-a/session-b.jsonl", vec!["visible beta"]),
        ];
        let (query, candidates) = request("beta", conversations, vec![1, 0]);
        let request = index_request(&query, &candidates);
        let mut cache = empty_embedding_cache(ChunkConfig::default());
        cache_request_passages(&mut cache, &request);
        let mut state = SemanticIndexState::with_cache(ChunkConfig::default(), cache);
        let mut embedder = FakeEmbedder::new();

        let response = state
            .refresh_or_prewarm(&request, &mut embedder, |_| {}, |_| {})
            .expect("rank succeeds");

        assert_eq!(response.indexed_chunk_count, 2);
        assert_eq!(
            response
                .hits
                .iter()
                .map(|hit| hit.conversation_index)
                .collect::<Vec<_>>(),
            vec![1, 0]
        );
    }

    #[test]
    fn semantic_index_ranks_more_than_legacy_limit_without_cap() {
        const LEGACY_LIMIT: usize = 100;
        let conversations = (0..LEGACY_LIMIT + 25)
            .map(|index| {
                conversation(
                    &format!("/projects/project-a/session-{index}.jsonl"),
                    vec!["visible alpha"],
                )
            })
            .collect::<Vec<_>>();
        let candidate_indices = (0..conversations.len()).collect::<Vec<_>>();
        let (query, candidates) = request("alpha", conversations, candidate_indices);
        let request = index_request(&query, &candidates);
        let mut cache = empty_embedding_cache(ChunkConfig::default());
        cache_request_passages(&mut cache, &request);
        let mut state = SemanticIndexState::with_cache(ChunkConfig::default(), cache);
        let mut embedder = FakeEmbedder::new();

        let response = state
            .refresh_or_prewarm(&request, &mut embedder, |_| {}, |_| {})
            .expect("rank succeeds");

        assert_eq!(response.indexed_chunk_count, LEGACY_LIMIT + 25);
        assert_eq!(response.hits.len(), LEGACY_LIMIT + 25);
        assert_eq!(
            response
                .hits
                .iter()
                .map(|hit| hit.conversation_index)
                .collect::<Vec<_>>(),
            (0..LEGACY_LIMIT + 25).collect::<Vec<_>>()
        );
    }

    #[test]
    fn cached_signature_reports_cache_ready_before_ranking() {
        let conversations = vec![conversation(
            "/projects/project-a/session-a.jsonl",
            vec!["visible alpha"],
        )];
        let (query, candidates) = request("alpha", conversations, vec![0]);
        let request = index_request(&query, &candidates);
        let mut cache = empty_embedding_cache(ChunkConfig::default());
        cache_request_passages(&mut cache, &request);
        let mut state = SemanticIndexState::with_cache(ChunkConfig::default(), cache);
        let mut embedder = FakeEmbedder::new();
        state
            .refresh_or_prewarm(&request, &mut embedder, |_| {}, |_| {})
            .expect("first rank succeeds");
        let mut progress = Vec::new();

        state
            .refresh_or_prewarm(
                &request,
                &mut embedder,
                |status| progress.push(status),
                |_| {},
            )
            .expect("second rank succeeds");

        assert_eq!(
            progress,
            vec![
                SemanticIndexProgress::CacheReady,
                SemanticIndexProgress::Ranking
            ]
        );
        assert_eq!(embedder.passage_calls, 0);
    }

    #[test]
    fn clear_empty_replaces_populated_index_state() {
        let populated = vec![conversation(
            "/projects/project-a/session-a.jsonl",
            vec!["visible alpha"],
        )];
        let (query, populated_candidates) = request("alpha", populated, vec![0]);
        let populated_request = index_request(&query, &populated_candidates);
        let mut cache = empty_embedding_cache(ChunkConfig::default());
        cache_request_passages(&mut cache, &populated_request);
        let mut state = SemanticIndexState::with_cache(ChunkConfig::default(), cache);
        let mut embedder = FakeEmbedder::new();
        state
            .refresh_or_prewarm(&populated_request, &mut embedder, |_| {}, |_| {})
            .expect("populated index succeeds");
        let empty = vec![conversation("/projects/project-a/session-a.jsonl", vec![])];
        let (empty_query, empty_candidates) = request("alpha", empty, vec![0]);
        let empty_request = index_request(&empty_query, &empty_candidates);

        let empty_signature = semantic_index_signature(&empty_request, ChunkConfig::default());
        state.clear_empty(&empty_request);

        assert_eq!(state.signature, Some(empty_signature));
        assert!(state.embedded_chunks.is_empty());
        assert!(!state.has_chunks(&empty_request));
    }

    #[test]
    fn empty_visible_dialogue_returns_without_embedding() {
        let conversations = vec![conversation("/projects/project-a/session-a.jsonl", vec![])];
        let (query, candidates) = request("alpha", conversations, vec![0]);
        let request = index_request(&query, &candidates);
        let cache = empty_embedding_cache(ChunkConfig::default());
        let mut state = SemanticIndexState::with_cache(ChunkConfig::default(), cache);
        let mut embedder = FakeEmbedder::new();

        let response = state
            .refresh_or_prewarm(&request, &mut embedder, |_| {}, |_| {})
            .expect("empty corpus succeeds");

        assert!(response.hits.is_empty());
        assert_eq!(response.progress, SemanticIndexProgress::EmptyCorpus);
        assert_eq!(embedder.passage_calls, 0);
        assert_eq!(embedder.query_calls, 0);
        assert!(!state.has_chunks(&request));
    }
}
