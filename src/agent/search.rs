use crate::agent::refs::{AgentConversationKey, MessageRange, ResolvedConversation};
use crate::agent::retrieval::{
    AgentRetrievalOptions, AgentSearchHit as RetrievalHit, AgentTranscriptSearchTarget,
    retrieve_agent_hits_for_target,
};
use crate::agent::transcript::AgentTranscript;
use crate::error::{AppError, Result};
use crate::history::Conversation;
use crate::search::mode::{SearchMode, SearchModeResolution, resolve_search_mode};
use crate::search::query::ParsedQuery;
use crate::semantic::types::{SemanticHit, SemanticScoreBreakdown};
use chrono::{DateTime, Local};
use std::cmp::Ordering;
use std::collections::HashMap;

const SHORTLIST_MIN: usize = 50;
const SHORTLIST_FACTOR: usize = 5;
const SHORTLIST_MAX: usize = 500;
const RRF_K: f64 = 60.0;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AgentSearchScope {
    Global,
    Local,
}

#[derive(Clone, Debug)]
pub struct AgentSearchRequest {
    pub query: String,
    pub top: usize,
    pub _scope: AgentSearchScope,
    pub cli_mode: Option<SearchMode>,
    pub config_mode: Option<SearchMode>,
    pub tui_semantic_search: Option<bool>,
}

#[derive(Clone, Debug)]
pub struct AgentWithinRequest {
    pub query: String,
    pub top: usize,
    pub cli_mode: Option<SearchMode>,
    pub config_mode: Option<SearchMode>,
    pub tui_semantic_search: Option<bool>,
}

#[derive(Clone)]
pub struct AgentConversationInput<'a> {
    pub conversation: &'a Conversation,
    pub resolved: ResolvedConversation,
    pub original_index: usize,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AgentSearchStats {
    pub shortlisted: usize,
    pub transcripts_loaded: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AgentSearchOutput {
    pub protocol: AgentProtocolKind,
    pub query: String,
    pub mode: SearchMode,
    pub hits: Vec<AgentOutputHit>,
    pub stats: AgentSearchStats,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AgentProtocolKind {
    Search,
    Within,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AgentOutputHit {
    pub conversation_ref: String,
    pub title: String,
    pub score: f64,
    pub source: AgentHitKind,
    pub preview: String,
    pub focus_range: MessageRange,
    pub read_range: MessageRange,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AgentHitKind {
    Exact,
    Lexical,
    Semantic,
    Hybrid,
}

#[derive(Clone, Debug)]
struct RankedHit {
    hit: AgentOutputHit,
    lexical_rank: Option<usize>,
    semantic_rank: Option<usize>,
    exact: bool,
}

pub fn effective_agent_mode(
    query: &str,
    cli_mode: Option<SearchMode>,
    config_mode: Option<SearchMode>,
    tui_semantic_search: Option<bool>,
) -> SearchMode {
    let parsed = ParsedQuery::parse(query);
    if parsed.is_quoted_only() {
        SearchMode::Exact
    } else {
        resolve_search_mode(SearchModeResolution {
            cli_mode,
            config_mode,
            tui_semantic_search,
        })
    }
}

pub fn format_agent_output(output: &AgentSearchOutput) -> String {
    let protocol = match output.protocol {
        AgentProtocolKind::Search => "agent-search",
        AgentProtocolKind::Within => "agent-within",
    };
    let mut rendered = format!(
        "protocol {protocol} v=1 mode={} hits={}\n",
        mode_atom(output.mode),
        output.hits.len()
    );
    rendered.push_str(&format!(
        "query text={} hits={}\n",
        crate::agent::protocol::escape_atom(&output.query),
        output.hits.len()
    ));
    if output.hits.is_empty() {
        return rendered;
    }

    for hit in &output.hits {
        rendered.push_str(&format!(
            "title ref={} text={}\n",
            crate::agent::protocol::escape_atom(&hit.conversation_ref),
            crate::agent::protocol::escape_atom(&hit.title)
        ));
        rendered.push_str(&format!(
            "hit ref={} source={} score={:.6} focus=m{}..m{} preview={}\n",
            crate::agent::protocol::escape_atom(&hit.conversation_ref),
            hit_source_atom(hit.source),
            hit.score,
            hit.focus_range.start,
            hit.focus_range.end,
            crate::agent::protocol::escape_atom(&hit.preview)
        ));
        rendered.push_str(&format!(
            "read ref={}:m{}..m{} focus=m{}..m{}\n",
            crate::agent::protocol::escape_atom(&hit.conversation_ref),
            hit.read_range.start,
            hit.read_range.end,
            hit.focus_range.start,
            hit.focus_range.end
        ));
    }
    rendered
}

pub fn run_within_search(
    request: &AgentWithinRequest,
    conversation: &Conversation,
    resolved: &ResolvedConversation,
    transcript: &AgentTranscript,
    semantic_hits: &[SemanticHit],
) -> AgentSearchOutput {
    let mode = effective_agent_mode(
        &request.query,
        request.cli_mode,
        request.config_mode,
        request.tui_semantic_search,
    );
    let hits = match mode {
        SearchMode::Lexical | SearchMode::Exact => retrieval_hits(
            &request.query,
            request.top,
            conversation,
            resolved,
            transcript,
            mode,
        ),
        SearchMode::Semantic => semantic_output_hits(
            semantic_hits,
            request.top,
            &[AgentConversationInput {
                conversation,
                resolved: resolved.clone(),
                original_index: 0,
            }],
        ),
        SearchMode::Hybrid => hybrid_hits(
            retrieval_hits(
                &request.query,
                request.top,
                conversation,
                resolved,
                transcript,
                SearchMode::Lexical,
            ),
            semantic_output_hits(
                semantic_hits,
                request.top,
                &[AgentConversationInput {
                    conversation,
                    resolved: resolved.clone(),
                    original_index: 0,
                }],
            ),
            request.top,
        ),
    };

    AgentSearchOutput {
        protocol: AgentProtocolKind::Within,
        query: request.query.clone(),
        mode,
        hits,
        stats: AgentSearchStats {
            shortlisted: 1,
            transcripts_loaded: 1,
        },
    }
}

pub fn run_global_lexical_search(
    request: &AgentSearchRequest,
    conversations: &[Conversation],
    keys: &[AgentConversationKey],
    ranked_indices: &[usize],
    load_transcript: impl Fn(&AgentConversationKey) -> Result<AgentTranscript>,
) -> Result<AgentSearchOutput> {
    let mode = effective_agent_mode(
        &request.query,
        request.cli_mode,
        request.config_mode,
        request.tui_semantic_search,
    );
    let retrieval_mode = match mode {
        SearchMode::Exact => SearchMode::Exact,
        _ => SearchMode::Lexical,
    };
    let limit = shortlist_limit(request.top).min(ranked_indices.len());
    let key_by_path = keys
        .iter()
        .map(|key| (key.path.clone(), key.clone()))
        .collect::<HashMap<_, _>>();
    let mut hits = Vec::new();
    let mut transcripts_loaded = 0;

    for index in ranked_indices.iter().take(limit).copied() {
        let Some(conversation) = conversations.get(index) else {
            continue;
        };
        let Some(key) = key_by_path.get(&conversation.path) else {
            continue;
        };
        let transcript = load_transcript(key)?;
        transcripts_loaded += 1;
        let resolved = ResolvedConversation {
            key: key.clone(),
            reference: key.conversation_ref(),
        };
        hits.extend(retrieval_hits(
            &request.query,
            request.top,
            conversation,
            &resolved,
            &transcript,
            retrieval_mode,
        ));
        sort_output_hits(&mut hits);
        hits.truncate(request.top);
        if hits.len() >= request.top {
            break;
        }
    }

    Ok(AgentSearchOutput {
        protocol: AgentProtocolKind::Search,
        query: request.query.clone(),
        mode,
        hits,
        stats: AgentSearchStats {
            shortlisted: limit,
            transcripts_loaded,
        },
    })
}

pub fn run_global_semantic_search(
    request: &AgentSearchRequest,
    inputs: &[AgentConversationInput<'_>],
    semantic_hits: &[SemanticHit],
) -> AgentSearchOutput {
    let mode = effective_agent_mode(
        &request.query,
        request.cli_mode,
        request.config_mode,
        request.tui_semantic_search,
    );
    AgentSearchOutput {
        protocol: AgentProtocolKind::Search,
        query: request.query.clone(),
        mode,
        hits: semantic_output_hits(semantic_hits, request.top, inputs),
        stats: AgentSearchStats {
            shortlisted: inputs.len(),
            transcripts_loaded: 0,
        },
    }
}

pub fn run_global_hybrid_search(
    request: &AgentSearchRequest,
    lexical: AgentSearchOutput,
    semantic_hits: &[SemanticHit],
    inputs: &[AgentConversationInput<'_>],
) -> AgentSearchOutput {
    let semantic = semantic_output_hits(semantic_hits, request.top, inputs);
    AgentSearchOutput {
        protocol: AgentProtocolKind::Search,
        query: request.query.clone(),
        mode: SearchMode::Hybrid,
        hits: hybrid_hits(lexical.hits, semantic, request.top),
        stats: lexical.stats,
    }
}

pub fn scoped_conversation_inputs(
    conversations: &[Conversation],
    scope: AgentSearchScope,
    current_project_dir_name: Option<&str>,
) -> Result<Vec<usize>> {
    let mut indices = Vec::new();
    for (index, conversation) in conversations.iter().enumerate() {
        if scope == AgentSearchScope::Local {
            let Some(project) = current_project_dir_name else {
                return Err(AppError::ConfigError(
                    "local agent search requires a current project".to_string(),
                ));
            };
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
        indices.push(index);
    }
    Ok(indices)
}

pub fn shortlist_limit(top: usize) -> usize {
    top.saturating_mul(SHORTLIST_FACTOR)
        .clamp(SHORTLIST_MIN, SHORTLIST_MAX)
}

fn retrieval_hits(
    query: &str,
    limit: usize,
    conversation: &Conversation,
    resolved: &ResolvedConversation,
    transcript: &AgentTranscript,
    mode: SearchMode,
) -> Vec<AgentOutputHit> {
    let search_query = if mode == SearchMode::Exact && !ParsedQuery::parse(query).is_quoted_only() {
        quote_query(query)
    } else {
        query.to_string()
    };
    retrieve_agent_hits_for_target(
        AgentTranscriptSearchTarget {
            transcript,
            conversation_ref: Some(&resolved.reference.canonical()),
            timestamp: Some(conversation.timestamp),
        },
        &search_query,
        AgentRetrievalOptions {
            limit,
            ..AgentRetrievalOptions::default()
        },
    )
    .into_iter()
    .map(|hit| retrieval_output_hit(hit, conversation, resolved, mode))
    .collect()
}

fn retrieval_output_hit(
    hit: RetrievalHit,
    conversation: &Conversation,
    resolved: &ResolvedConversation,
    mode: SearchMode,
) -> AgentOutputHit {
    AgentOutputHit {
        conversation_ref: resolved.reference.canonical(),
        title: title_for_conversation(conversation),
        score: hit.score,
        source: if mode == SearchMode::Exact || ParsedQuery::parse(&hit.preview).is_quoted_only() {
            AgentHitKind::Exact
        } else {
            AgentHitKind::Lexical
        },
        preview: hit.preview,
        focus_range: hit.focus_range,
        read_range: hit.read_range,
    }
}

fn semantic_output_hits(
    hits: &[SemanticHit],
    limit: usize,
    inputs: &[AgentConversationInput<'_>],
) -> Vec<AgentOutputHit> {
    let mut output = hits
        .iter()
        .filter_map(|hit| {
            let input = inputs
                .iter()
                .find(|input| input.original_index == hit.conversation_index)?;
            Some(AgentOutputHit {
                conversation_ref: input.resolved.reference.canonical(),
                title: title_for_conversation(input.conversation),
                score: semantic_score(hit.score_breakdown),
                source: AgentHitKind::Semantic,
                preview: hit.snippet.clone(),
                focus_range: hit.message_range,
                read_range: hit.message_range,
            })
        })
        .collect::<Vec<_>>();
    sort_output_hits(&mut output);
    output.truncate(limit);
    output
}

fn hybrid_hits(
    lexical_hits: Vec<AgentOutputHit>,
    semantic_hits: Vec<AgentOutputHit>,
    limit: usize,
) -> Vec<AgentOutputHit> {
    let mut ranked = Vec::<RankedHit>::new();
    for (rank, hit) in lexical_hits.into_iter().enumerate() {
        ranked.push(RankedHit {
            exact: hit.source == AgentHitKind::Exact,
            hit,
            lexical_rank: Some(rank + 1),
            semantic_rank: None,
        });
    }
    for (rank, hit) in semantic_hits.into_iter().enumerate() {
        if let Some(existing) = ranked.iter_mut().find(|existing| {
            existing.hit.conversation_ref == hit.conversation_ref
                && existing.hit.focus_range == hit.focus_range
        }) {
            existing.semantic_rank = Some(rank + 1);
            existing.hit.score = rrf_score(existing.lexical_rank, existing.semantic_rank);
            existing.hit.source = AgentHitKind::Hybrid;
            existing.hit.read_range = existing.hit.read_range.union(&hit.read_range);
        } else {
            ranked.push(RankedHit {
                hit,
                lexical_rank: None,
                semantic_rank: Some(rank + 1),
                exact: false,
            });
        }
    }
    for ranked_hit in &mut ranked {
        ranked_hit.hit.score = rrf_score(ranked_hit.lexical_rank, ranked_hit.semantic_rank);
    }
    ranked.sort_by(|a, b| {
        b.hit
            .score
            .partial_cmp(&a.hit.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| source_priority(a).cmp(&source_priority(b)))
            .then_with(|| a.hit.conversation_ref.cmp(&b.hit.conversation_ref))
            .then_with(|| a.hit.focus_range.start.cmp(&b.hit.focus_range.start))
    });
    ranked.truncate(limit);
    ranked.into_iter().map(|ranked| ranked.hit).collect()
}

fn source_priority(hit: &RankedHit) -> u8 {
    if hit.exact {
        0
    } else if hit.lexical_rank.is_some() {
        1
    } else {
        2
    }
}

fn rrf_score(lexical_rank: Option<usize>, semantic_rank: Option<usize>) -> f64 {
    lexical_rank.map_or(0.0, |rank| 1.0 / (RRF_K + rank as f64))
        + semantic_rank.map_or(0.0, |rank| 1.0 / (RRF_K + rank as f64))
}

fn semantic_score(score: SemanticScoreBreakdown) -> f64 {
    score.hybrid as f64
}

fn sort_output_hits(hits: &mut [AgentOutputHit]) {
    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| source_rank(a.source).cmp(&source_rank(b.source)))
            .then_with(|| a.conversation_ref.cmp(&b.conversation_ref))
            .then_with(|| a.focus_range.start.cmp(&b.focus_range.start))
    });
}

fn source_rank(source: AgentHitKind) -> u8 {
    match source {
        AgentHitKind::Exact => 0,
        AgentHitKind::Lexical => 1,
        AgentHitKind::Hybrid => 2,
        AgentHitKind::Semantic => 3,
    }
}

fn quote_query(query: &str) -> String {
    format!("\"{}\"", query.replace('"', ""))
}

fn title_for_conversation(conversation: &Conversation) -> String {
    conversation
        .custom_title
        .as_deref()
        .or(conversation.summary.as_deref())
        .unwrap_or(&conversation.preview)
        .to_string()
}

fn mode_atom(mode: SearchMode) -> &'static str {
    match mode {
        SearchMode::Lexical => "lexical",
        SearchMode::Semantic => "semantic",
        SearchMode::Exact => "exact",
        SearchMode::Hybrid => "hybrid",
    }
}

fn hit_source_atom(source: AgentHitKind) -> &'static str {
    match source {
        AgentHitKind::Exact => "exact",
        AgentHitKind::Lexical => "lexical",
        AgentHitKind::Semantic => "semantic",
        AgentHitKind::Hybrid => "hybrid",
    }
}

#[allow(dead_code)]
fn _keep_timestamp_type(_: Option<DateTime<Local>>) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::refs::AgentConversationKey;
    use crate::agent::transcript::{
        AgentMessage, AgentMessagePart, AgentMessageRole, AgentPartSource,
    };
    use crate::semantic::types::{
        SemanticChunkIdentity, SemanticExplanation, SemanticQuality, SemanticRationaleKind,
    };
    use chrono::Local;
    use std::path::PathBuf;

    fn source(role: AgentMessageRole) -> AgentPartSource {
        AgentPartSource {
            role,
            timestamp: None,
            jsonl_line: 1,
            part_index: 0,
            assistant_message_id: None,
            parent_tool_use_id: None,
            tool_name: None,
        }
    }

    fn message(ordinal: usize, role: AgentMessageRole, text: &str) -> AgentMessage {
        AgentMessage {
            ordinal,
            role,
            timestamp: None,
            jsonl_line: ordinal,
            assistant_message_id: None,
            parent_tool_use_id: None,
            parts: vec![AgentMessagePart::Text {
                text: text.to_string(),
                source: source(role),
            }],
        }
    }

    fn transcript(messages: Vec<AgentMessage>) -> AgentTranscript {
        AgentTranscript {
            path: PathBuf::from("session.jsonl"),
            messages,
        }
    }

    fn conversation(path: &str, title: &str) -> Conversation {
        Conversation {
            path: PathBuf::from(path),
            index: 0,
            timestamp: Local::now(),
            preview: title.to_string(),
            preview_first: title.to_string(),
            preview_last: title.to_string(),
            full_text: title.to_string(),
            semantic_turns: vec![title.to_string()],
            semantic_turn_ranges: vec![MessageRange::single(1)],
            search_text_lower: title.to_string(),
            project_name: Some("project-a".to_string()),
            project_path: None,
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

    fn resolved(path: &str) -> ResolvedConversation {
        let key = AgentConversationKey::new("project-a", path, PathBuf::from(path));
        ResolvedConversation {
            reference: key.conversation_ref(),
            key,
        }
    }

    fn request(query: &str, mode: Option<SearchMode>) -> AgentWithinRequest {
        AgentWithinRequest {
            query: query.to_string(),
            top: 10,
            cli_mode: mode,
            config_mode: None,
            tui_semantic_search: None,
        }
    }

    fn semantic_hit(index: usize, range: MessageRange, text: &str, score: f32) -> SemanticHit {
        SemanticHit::new(
            SemanticScoreBreakdown {
                hybrid: score,
                semantic: score,
                lexical: 0.0,
            },
            SemanticExplanation {
                quality: SemanticQuality::Good,
                quality_label: "good",
                matched_terms: vec![],
                evidence_preview: text.to_string(),
                rationale_kind: SemanticRationaleKind::SemanticOnly,
                chunk: SemanticChunkIdentity {
                    conversation_index: index,
                    session: "session".to_string(),
                    chunk_index: range.start,
                    message_range: range,
                },
            },
        )
    }

    #[test]
    fn quoted_query_forces_exact_mode() {
        assert_eq!(
            effective_agent_mode(
                "\"literal needle\"",
                Some(SearchMode::Semantic),
                Some(SearchMode::Hybrid),
                Some(true),
            ),
            SearchMode::Exact
        );
    }

    #[test]
    fn zero_matches_emit_protocol_and_query_only() {
        let output = AgentSearchOutput {
            protocol: AgentProtocolKind::Search,
            query: "missing".to_string(),
            mode: SearchMode::Lexical,
            hits: vec![],
            stats: AgentSearchStats::default(),
        };

        assert_eq!(
            format_agent_output(&output),
            "protocol agent-search v=1 mode=lexical hits=0\nquery text=missing hits=0\n"
        );
    }

    #[test]
    fn within_lexical_formats_title_hit_and_read_lines() {
        let conv = conversation("session.jsonl", "cache title");
        let resolved = resolved("session.jsonl");
        let transcript = transcript(vec![
            message(1, AgentMessageRole::User, "question"),
            message(2, AgentMessageRole::Assistant, "cache warming answer"),
        ]);

        let output = run_within_search(
            &request("cache warming", None),
            &conv,
            &resolved,
            &transcript,
            &[],
        );
        let rendered = format_agent_output(&output);

        assert!(rendered.starts_with("protocol agent-within v=1 mode=lexical hits=1\n"));
        assert!(rendered.contains("title ref=ch_"));
        assert!(rendered.contains("hit ref=ch_"));
        assert!(rendered.contains("read ref=ch_"));
        assert!(rendered.contains("focus=m2..m2"));
    }

    #[test]
    fn within_semantic_returns_message_level_hits() {
        let conv = conversation("session.jsonl", "semantic title");
        let resolved = resolved("session.jsonl");
        let transcript = transcript(vec![message(1, AgentMessageRole::User, "placeholder")]);
        let output = run_within_search(
            &request("semantic", Some(SearchMode::Semantic)),
            &conv,
            &resolved,
            &transcript,
            &[
                semantic_hit(0, MessageRange::single(1), "first", 0.8),
                semantic_hit(0, MessageRange::single(3), "third", 0.7),
            ],
        );

        assert_eq!(output.hits.len(), 2);
        assert_eq!(output.hits[0].focus_range, MessageRange::single(1));
        assert_eq!(output.hits[1].focus_range, MessageRange::single(3));
    }

    #[test]
    fn hybrid_dedupes_same_focus_and_prefers_lexical_preview() {
        let lexical = vec![AgentOutputHit {
            conversation_ref: "ch_123456789abc".to_string(),
            title: "title".to_string(),
            score: 10.0,
            source: AgentHitKind::Lexical,
            preview: "lexical preview".to_string(),
            focus_range: MessageRange::single(2),
            read_range: MessageRange { start: 1, end: 3 },
        }];
        let semantic = vec![AgentOutputHit {
            conversation_ref: "ch_123456789abc".to_string(),
            title: "title".to_string(),
            score: 0.9,
            source: AgentHitKind::Semantic,
            preview: "semantic preview".to_string(),
            focus_range: MessageRange::single(2),
            read_range: MessageRange::single(2),
        }];

        let hits = hybrid_hits(lexical, semantic, 10);

        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].source, AgentHitKind::Hybrid);
        assert_eq!(hits[0].preview, "lexical preview");
        assert_eq!(hits[0].read_range, MessageRange { start: 1, end: 3 });
    }

    #[test]
    fn global_lexical_loads_only_bounded_shortlist_for_evidence() {
        let conversations = (0..60)
            .map(|index| conversation(&format!("session-{index}.jsonl"), "needle title"))
            .collect::<Vec<_>>();
        let keys = conversations
            .iter()
            .map(|conversation| {
                AgentConversationKey::new(
                    "project-a",
                    conversation.path.file_name().unwrap().to_string_lossy(),
                    conversation.path.clone(),
                )
            })
            .collect::<Vec<_>>();
        let ranked = (0..60).collect::<Vec<_>>();
        let request = AgentSearchRequest {
            query: "needle".to_string(),
            top: 3,
            _scope: AgentSearchScope::Global,
            cli_mode: Some(SearchMode::Lexical),
            config_mode: None,
            tui_semantic_search: None,
        };

        let output = run_global_lexical_search(&request, &conversations, &keys, &ranked, |_| {
            Ok(transcript(vec![message(
                1,
                AgentMessageRole::User,
                "needle evidence",
            )]))
        })
        .unwrap();

        assert_eq!(output.hits.len(), 3);
        assert_eq!(output.stats.shortlisted, 50);
        assert_eq!(output.stats.transcripts_loaded, 3);
    }
}
