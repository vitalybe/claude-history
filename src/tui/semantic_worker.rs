use crate::error::Result;
use crate::semantic::cache::write_embedding_cache;
use crate::semantic::fastembed::FastembedEmbedder;
use crate::semantic::index::{
    SemanticIndexCandidate, SemanticIndexProgress, SemanticIndexRequest, SemanticIndexResponse,
    SemanticIndexState,
};
use crate::tui::app::{SemanticProgress, SemanticResultMetadata};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::mpsc;

pub type SemanticSearchCandidate = SemanticIndexCandidate;

#[derive(Clone)]
pub struct SemanticSearchRequest {
    pub generation: u64,
    pub query: String,
    pub candidates: Arc<Vec<SemanticSearchCandidate>>,
    pub prewarm: bool,
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
    pub prewarm: bool,
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
            let mut state = SemanticIndexState::new();
            let mut embedder: Option<FastembedEmbedder> = None;

            while let Ok(mut request) = cmd_rx.recv() {
                while let Ok(pending) = cmd_rx.try_recv() {
                    request = pending;
                }

                let index_request = semantic_index_request(&request);
                let response = if state.has_chunks(&index_request) {
                    if embedder.is_none() {
                        let _ = res_tx.send(SemanticSearchMessage::Progress {
                            generation: request.generation,
                            progress: SemanticProgress::InitializingModel,
                        });
                        embedder = match FastembedEmbedder::new_quiet() {
                            Ok(embedder) => Some(embedder),
                            Err(error) => {
                                let _ = res_tx.send(SemanticSearchMessage::Complete(
                                    SemanticSearchResponse {
                                        generation: request.generation,
                                        filtered: Vec::new(),
                                        metadata: HashMap::new(),
                                        error: Some(error.to_string()),
                                        progress: SemanticProgress::Failed,
                                        prewarm: request.prewarm,
                                    },
                                ));
                                continue;
                            }
                        };
                    }
                    rank_or_prewarm_semantic_request(
                        request.generation,
                        &index_request,
                        &mut state,
                        embedder.as_mut().unwrap(),
                        &res_tx,
                    )
                    .unwrap_or_else(|error| SemanticSearchResponse {
                        generation: request.generation,
                        filtered: Vec::new(),
                        metadata: HashMap::new(),
                        error: Some(error.to_string()),
                        progress: SemanticProgress::Failed,
                        prewarm: request.prewarm,
                    })
                } else {
                    state.clear_empty(&index_request);
                    empty_semantic_response(request.generation, request.prewarm)
                };

                let _ = res_tx.send(SemanticSearchMessage::Complete(response));
            }
        })
        .expect("failed to spawn semantic search worker thread");

    (cmd_tx, res_rx)
}

fn semantic_index_request(request: &SemanticSearchRequest) -> SemanticIndexRequest<'_> {
    SemanticIndexRequest {
        query: &request.query,
        candidates: &request.candidates,
        prewarm: request.prewarm,
    }
}

fn rank_or_prewarm_semantic_request(
    generation: u64,
    request: &SemanticIndexRequest<'_>,
    state: &mut SemanticIndexState,
    embedder: &mut FastembedEmbedder,
    res_tx: &mpsc::Sender<SemanticSearchMessage>,
) -> Result<SemanticSearchResponse> {
    let response = state.refresh_or_prewarm(
        request,
        embedder,
        |progress| {
            let _ = res_tx.send(SemanticSearchMessage::Progress {
                generation,
                progress: semantic_progress(progress),
            });
        },
        write_embedding_cache,
    )?;
    Ok(semantic_search_response(generation, response))
}

fn semantic_search_response(
    generation: u64,
    response: SemanticIndexResponse,
) -> SemanticSearchResponse {
    let filtered = response
        .hits
        .iter()
        .map(|hit| hit.conversation_index)
        .collect::<Vec<_>>();
    let metadata = response
        .hits
        .into_iter()
        .map(|hit| {
            (
                hit.conversation_index,
                SemanticResultMetadata {
                    score_breakdown: hit.score_breakdown,
                    explanation: hit.explanation,
                },
            )
        })
        .collect();

    SemanticSearchResponse {
        generation,
        filtered,
        metadata,
        error: None,
        progress: semantic_progress(response.progress),
        prewarm: response.prewarm,
    }
}

fn empty_semantic_response(generation: u64, prewarm: bool) -> SemanticSearchResponse {
    SemanticSearchResponse {
        generation,
        filtered: Vec::new(),
        metadata: HashMap::new(),
        error: None,
        progress: SemanticProgress::EmptyCorpus,
        prewarm,
    }
}

fn semantic_progress(progress: SemanticIndexProgress) -> SemanticProgress {
    match progress {
        SemanticIndexProgress::Embedding { completed, total } => {
            SemanticProgress::Embedding { completed, total }
        }
        SemanticIndexProgress::CacheReady => SemanticProgress::CacheReady,
        SemanticIndexProgress::Ranking => SemanticProgress::Ranking,
        SemanticIndexProgress::Complete => SemanticProgress::Complete,
        SemanticIndexProgress::EmptyCorpus => SemanticProgress::EmptyCorpus,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::history::Conversation;
    use crate::semantic::types::{
        SemanticChunkIdentity, SemanticExplanation, SemanticQuality, SemanticRationaleKind,
        SemanticScoreBreakdown,
    };
    use crate::tui::search::normalize_for_search;
    use chrono::Local;
    use std::path::PathBuf;
    use std::time::Duration;

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
        let candidates = candidate_indices
            .into_iter()
            .map(|index| SemanticSearchCandidate {
                index,
                conversation: Arc::new(conversations[index].clone()),
            })
            .collect();
        SemanticSearchRequest {
            generation: 1,
            query: query.to_string(),
            candidates: Arc::new(candidates),
            prewarm: false,
        }
    }

    #[test]
    fn maps_domain_hits_to_original_indices_and_metadata() {
        let response = semantic_search_response(
            7,
            SemanticIndexResponse {
                hits: vec![crate::semantic::types::SemanticHit::new(
                    SemanticScoreBreakdown {
                        hybrid: 1.2,
                        semantic: 1.0,
                        lexical: 0.2,
                    },
                    SemanticExplanation {
                        quality: SemanticQuality::Strong,
                        quality_label: "strong",
                        matched_terms: vec!["beta".to_string()],
                        evidence_preview: "visible beta".to_string(),
                        rationale_kind: SemanticRationaleKind::LexicalBoosted,
                        chunk: SemanticChunkIdentity {
                            conversation_index: 42,
                            session: "session-b".to_string(),
                            chunk_index: 0,
                        },
                    },
                )],
                indexed_chunk_count: 1,
                query_embedding_returned: true,
                progress: SemanticIndexProgress::Complete,
                prewarm: false,
            },
        );

        assert_eq!(response.generation, 7);
        assert_eq!(response.filtered, vec![42]);
        assert_eq!(response.progress, SemanticProgress::Complete);
        let metadata = &response.metadata[&42];
        assert_eq!(metadata.score_breakdown.hybrid, 1.2);
        assert_eq!(metadata.explanation.evidence_preview, "visible beta");
        assert_eq!(metadata.explanation.chunk.conversation_index, 42);
    }

    #[test]
    fn empty_visible_dialogue_returns_before_embedder_initialization() {
        let (tx, rx) = spawn_semantic_worker();
        let request = request(
            "alpha",
            vec![conversation("/projects/project-a/session-a.jsonl", vec![])],
            vec![0],
        );

        tx.send(request).expect("send semantic request");
        let message = rx
            .recv_timeout(Duration::from_secs(2))
            .expect("empty response");

        match message {
            SemanticSearchMessage::Complete(response) => {
                assert!(response.filtered.is_empty());
                assert!(response.metadata.is_empty());
                assert_eq!(response.error, None);
                assert_eq!(response.progress, SemanticProgress::EmptyCorpus);
            }
            SemanticSearchMessage::Progress { progress, .. } => {
                panic!("unexpected progress before empty response: {progress:?}");
            }
        }
    }
}
