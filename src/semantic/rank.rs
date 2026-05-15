use crate::semantic::types::{EmbeddedChunk, SemanticHit};
use std::collections::HashMap;

pub fn rank_chunks(
    query: &str,
    query_embedding: &[f32],
    chunks: &[EmbeddedChunk],
) -> Vec<SemanticHit> {
    let mut best_by_session: HashMap<String, SemanticHit> = HashMap::new();
    for chunk in chunks {
        let semantic_score = cosine(query_embedding, &chunk.embedding);
        let lexical_score = lexical_overlap(query, &chunk.text);
        let hybrid_score = semantic_score + lexical_score;

        let replace = best_by_session
            .get(&chunk.session)
            .is_none_or(|existing| hybrid_score > existing.hybrid_score);
        if replace {
            best_by_session.insert(
                chunk.session.clone(),
                SemanticHit {
                    conversation_index: chunk.conversation_index,
                    session: chunk.session.clone(),
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
    hits
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
        text: &str,
        embedding: Vec<f32>,
    ) -> EmbeddedChunk {
        EmbeddedChunk {
            conversation_index,
            session: session.to_string(),
            text: text.to_string(),
            embedding,
        }
    }

    #[test]
    fn ranking_keeps_best_chunk_per_session() {
        let chunks = vec![
            embedded("session-a", 0, "rust cache", vec![1.0, 0.0]),
            embedded("session-a", 0, "unrelated", vec![0.0, 1.0]),
            embedded("session-b", 1, "rust", vec![0.5, 0.5]),
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
            embedded("session-a", 0, "same words", vec![0.0, 1.0]),
            embedded("session-b", 1, "same words", vec![1.0, 0.0]),
        ];

        let hits = rank_chunks("same words", &[0.0, 1.0], &chunks);

        assert_eq!(hits[0].session, "session-a");
        assert_eq!(hits[0].semantic_score, 1.0);
    }
}
