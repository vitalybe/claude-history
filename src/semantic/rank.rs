use crate::semantic::types::{EmbeddedChunk, SemanticHit};
use std::cmp::Ordering;
use std::collections::HashMap;

pub fn rank_chunks(
    query: &str,
    query_embedding: &[f32],
    chunks: &[EmbeddedChunk],
) -> Vec<SemanticHit> {
    let mut best_by_conversation: HashMap<usize, SemanticHit> = HashMap::new();
    for chunk in chunks {
        let semantic_score = cosine(query_embedding, &chunk.embedding);
        let lexical_score = lexical_overlap(query, &chunk.text);
        let hybrid_score = semantic_score + lexical_score;
        let hit = SemanticHit {
            conversation_index: chunk.conversation_index,
            session: chunk.session.clone(),
            chunk_index: chunk.chunk_index,
            semantic_score,
            lexical_score,
            hybrid_score,
            snippet: chunk.text.clone(),
        };

        let replace = best_by_conversation
            .get(&chunk.conversation_index)
            .is_none_or(|existing| compare_hits(&hit, existing).is_lt());
        if replace {
            best_by_conversation.insert(chunk.conversation_index, hit);
        }
    }

    let mut hits: Vec<_> = best_by_conversation.into_values().collect();
    hits.sort_by(compare_hits);
    hits
}

fn compare_hits(a: &SemanticHit, b: &SemanticHit) -> Ordering {
    b.hybrid_score
        .total_cmp(&a.hybrid_score)
        .then_with(|| b.semantic_score.total_cmp(&a.semantic_score))
        .then_with(|| b.lexical_score.total_cmp(&a.lexical_score))
        .then_with(|| a.conversation_index.cmp(&b.conversation_index))
        .then_with(|| a.session.cmp(&b.session))
        .then_with(|| a.chunk_index.cmp(&b.chunk_index))
}

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

#[cfg(test)]
mod tests {
    use super::*;

    fn embedded(
        session: &str,
        conversation_index: usize,
        chunk_index: usize,
        text: &str,
        embedding: Vec<f32>,
    ) -> EmbeddedChunk {
        EmbeddedChunk {
            conversation_index,
            session: session.to_string(),
            chunk_index,
            text: text.to_string(),
            embedding,
        }
    }

    #[test]
    fn ranking_keeps_best_chunk_per_session() {
        let chunks = vec![
            embedded("session-a", 0, 0, "rust cache", vec![1.0, 0.0]),
            embedded("session-a", 0, 1, "unrelated", vec![0.0, 1.0]),
            embedded("session-b", 1, 0, "rust", vec![0.5, 0.5]),
        ];

        let hits = rank_chunks("rust cache", &[1.0, 0.0], &chunks);

        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].session, "session-a");
        assert_eq!(hits[0].snippet, "rust cache");
        assert!(hits[0].semantic_score > hits[1].semantic_score);
        assert_eq!(hits[0].lexical_score, 0.2);
    }

    #[test]
    fn ranking_uses_explicit_query_embedding() {
        let chunks = vec![
            embedded("session-a", 0, 0, "same words", vec![0.0, 1.0]),
            embedded("session-b", 1, 0, "same words", vec![1.0, 0.0]),
        ];

        let hits = rank_chunks("same words", &[0.0, 1.0], &chunks);

        assert_eq!(hits[0].session, "session-a");
        assert_eq!(hits[0].semantic_score, 1.0);
    }

    #[test]
    fn ranking_keeps_copied_sessions_separate() {
        let chunks = vec![
            embedded("session", 0, 0, "same words", vec![1.0, 0.0]),
            embedded("session", 1, 0, "same words", vec![1.0, 0.0]),
        ];

        let hits = rank_chunks("same words", &[1.0, 0.0], &chunks);

        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].conversation_index, 0);
        assert_eq!(hits[1].conversation_index, 1);
    }

    #[test]
    fn ranking_uses_stable_tiebreaks_for_sessions_and_chunks() {
        let chunks = vec![
            embedded("session-b", 1, 0, "same words", vec![1.0, 0.0]),
            embedded("session-a", 0, 1, "same words", vec![1.0, 0.0]),
            embedded("session-a", 0, 0, "same words", vec![1.0, 0.0]),
        ];

        let hits = rank_chunks("same words", &[1.0, 0.0], &chunks);

        assert_eq!(hits[0].conversation_index, 0);
        assert_eq!(hits[0].chunk_index, 0);
        assert_eq!(hits[1].conversation_index, 1);
    }
}
