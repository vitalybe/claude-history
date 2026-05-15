use crate::history::Conversation;
use crate::semantic::types::{ChunkConfig, FileMetadata, SemanticChunk};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub fn build_chunks(conversations: &[&Conversation], config: ChunkConfig) -> Vec<SemanticChunk> {
    let mut chunks = Vec::new();
    for (conversation_index, conversation) in conversations.iter().enumerate() {
        let semantic_turns = conversation
            .semantic_turns
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();

        for (chunk_index, chunk) in group_turns(&semantic_turns, config).into_iter().enumerate() {
            push_chunk(
                &mut chunks,
                conversation,
                conversation_index,
                chunk_index,
                &chunk,
            );
        }
    }
    chunks
}

fn group_turns(turns: &[&str], config: ChunkConfig) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();

    for (index, turn) in turns.iter().enumerate() {
        let turn = turn.trim();
        if turn.is_empty() {
            continue;
        }

        if turn.len() > config.target_chars {
            flush_chunk(&mut chunks, &mut current);
            split_long_text(turn, &mut chunks, config);
            continue;
        }

        let separator_len = if current.is_empty() { 0 } else { 2 };
        if !current.is_empty() && current.len() + separator_len + turn.len() > config.target_chars {
            flush_chunk(&mut chunks, &mut current);
            append_context(turns, index, &mut current, config);
        }

        if !current.is_empty() {
            current.push_str("\n\n");
        }
        current.push_str(turn);
    }

    flush_chunk(&mut chunks, &mut current);
    chunks
}

fn append_context(turns: &[&str], index: usize, current: &mut String, config: ChunkConfig) {
    let start = index.saturating_sub(config.context_turns);
    for turn in &turns[start..index] {
        let turn = turn.trim();
        if turn.is_empty() || turn.len() + current.len() > config.overlap_chars {
            continue;
        }
        if !current.is_empty() {
            current.push_str("\n\n");
        }
        current.push_str(turn);
    }
}

fn flush_chunk(chunks: &mut Vec<String>, current: &mut String) {
    if !current.trim().is_empty() {
        chunks.push(std::mem::take(current));
    }
}

fn split_long_text(mut text: &str, chunks: &mut Vec<String>, config: ChunkConfig) {
    while !text.is_empty() {
        let (chunk, rest) = split_chunk(text, config);
        chunks.push(chunk.to_owned());
        text = rest;
    }
}

fn push_chunk(
    chunks: &mut Vec<SemanticChunk>,
    conversation: &Conversation,
    conversation_index: usize,
    chunk_index: usize,
    chunk: &str,
) {
    let text = normalize_snippet(chunk);
    if !text.is_empty() {
        let session = conversation
            .path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .to_owned();
        let key = chunk_key(conversation, chunk_index);
        chunks.push(SemanticChunk {
            conversation_index,
            session,
            chunk_index,
            key,
            text,
            metadata: file_metadata(conversation),
        });
    }
}

fn chunk_key(conversation: &Conversation, chunk_index: usize) -> String {
    let path = normalized_cache_path(&conversation.path);
    format!("{}:{chunk_index}", path.display())
}

fn normalized_cache_path(path: &Path) -> PathBuf {
    if let Ok(path) = path.canonicalize() {
        return path;
    }
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(path))
            .unwrap_or_else(|_| path.to_path_buf())
    }
}

fn split_chunk(text: &str, config: ChunkConfig) -> (&str, &str) {
    if text.len() <= config.target_chars {
        return (text, "");
    }

    let end = floor_char_boundary(text, config.target_chars);
    let chunk = &text[..end];
    let next_start = end.saturating_sub(config.overlap_chars);
    let next_start = floor_char_boundary(text, next_start);
    (chunk, text[next_start..].trim_start())
}

fn floor_char_boundary(text: &str, index: usize) -> usize {
    let mut index = index.min(text.len());
    while index > 0 && !text.is_char_boundary(index) {
        index -= 1;
    }
    index
}

pub fn normalize_snippet(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Local;

    fn test_conversation(path: &str, semantic_turns: Vec<String>) -> Conversation {
        Conversation {
            path: PathBuf::from(path),
            index: 0,
            timestamp: Local::now(),
            preview: "visible user text".to_string(),
            preview_first: "visible user text".to_string(),
            preview_last: "visible assistant text".to_string(),
            full_text: "title sentinel summary sentinel cwd sentinel project sentinel tool output sentinel full text only sentinel".to_string(),
            semantic_turns,
            search_text_lower: "title sentinel summary sentinel cwd sentinel project sentinel tool output sentinel full text only sentinel".to_string(),
            project_name: Some("project sentinel".to_string()),
            project_path: Some(PathBuf::from("/projects/project-a")),
            cwd: Some(PathBuf::from("/cwd/sentinel")),
            message_count: 2,
            parse_errors: Vec::new(),
            summary: Some("summary sentinel".to_string()),
            custom_title: Some("title sentinel".to_string()),
            model: Some("claude-sonnet-4-6".to_string()),
            total_tokens: 0,
            duration_minutes: None,
        }
    }

    #[test]
    fn semantic_chunks_exclude_metadata_and_full_text() {
        let conversation = test_conversation(
            "/projects/project-a/session-1.jsonl",
            vec![
                "visible user text".to_string(),
                "visible assistant text".to_string(),
            ],
        );

        let chunks = build_chunks(&[&conversation], ChunkConfig::default());

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "visible user text visible assistant text");
        assert!(!chunks[0].text.contains("title sentinel"));
        assert!(!chunks[0].text.contains("summary sentinel"));
        assert!(!chunks[0].text.contains("cwd sentinel"));
        assert!(!chunks[0].text.contains("project sentinel"));
        assert!(!chunks[0].text.contains("tool output sentinel"));
        assert!(!chunks[0].text.contains("full text only sentinel"));
    }

    #[test]
    fn semantic_chunks_do_not_fall_back_to_full_text() {
        let conversation = test_conversation("/projects/project-a/session-1.jsonl", Vec::new());

        assert!(build_chunks(&[&conversation], ChunkConfig::default()).is_empty());
    }

    #[test]
    fn empty_semantic_turns_do_not_emit_chunks() {
        let conversation = test_conversation(
            "/projects/project-a/session-1.jsonl",
            vec!["".to_string(), "   ".to_string(), "\n\t".to_string()],
        );

        assert!(build_chunks(&[&conversation], ChunkConfig::default()).is_empty());
    }

    #[test]
    fn unicode_semantic_turns_are_preserved() {
        let conversation = test_conversation(
            "/projects/project-a/session-1.jsonl",
            vec!["你好，缓存 résumé".to_string()],
        );

        let chunks = build_chunks(&[&conversation], ChunkConfig::default());

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "你好，缓存 résumé");
    }

    #[test]
    fn long_turns_split_into_bounded_overlapping_chunks() {
        let text = "abcdef".repeat(4);
        let conversation = test_conversation("/projects/project-a/session-1.jsonl", vec![text]);
        let config = ChunkConfig {
            target_chars: 10,
            overlap_chars: 3,
            context_turns: 0,
        };

        let chunks = build_chunks(&[&conversation], config);

        assert!(chunks.len() > 1);
        assert!(
            chunks
                .iter()
                .all(|chunk| chunk.text.len() <= config.target_chars)
        );
        assert_eq!(chunks[0].text, "abcdefabcd");
        assert!(chunks[1].text.starts_with("bcd"));
    }

    #[test]
    fn long_unicode_turns_floor_to_char_boundaries() {
        let text = "éaébé".to_string();
        let conversation = test_conversation("/projects/project-a/session-1.jsonl", vec![text]);
        let config = ChunkConfig {
            target_chars: 5,
            overlap_chars: 2,
            context_turns: 0,
        };

        let chunks = build_chunks(&[&conversation], config);

        assert_eq!(chunks[0].text, "éaé");
        assert_eq!(chunks[1].text, "ébé");
        assert!(chunks[1].text.starts_with("é"));
    }

    #[test]
    fn chunk_identity_uses_selected_slice_index_and_session_key() {
        let first = test_conversation(
            "/projects/project-a/session-1.jsonl",
            vec!["first".to_string()],
        );
        let second = test_conversation(
            "/projects/project-a/session-2.jsonl",
            vec!["second".to_string()],
        );

        let chunks = build_chunks(&[&first, &second], ChunkConfig::default());

        assert_eq!(chunks[0].conversation_index, 0);
        assert_eq!(chunks[0].session, "session-1");
        assert_ne!(chunks[0].key, "session-1:0");
        assert_eq!(chunks[1].conversation_index, 1);
        assert_eq!(chunks[1].session, "session-2");
        assert_ne!(chunks[1].key, "session-2:0");
        assert_ne!(chunks[0].key, chunks[1].key);
    }

    #[test]
    fn chunk_identity_distinguishes_copied_sessions() {
        let first = test_conversation(
            "/projects/project-a/session.jsonl",
            vec!["first".to_string()],
        );
        let second = test_conversation(
            "/projects/project-b/session.jsonl",
            vec!["second".to_string()],
        );

        let chunks = build_chunks(&[&first, &second], ChunkConfig::default());

        assert_eq!(chunks[0].session, "session");
        assert_eq!(chunks[1].session, "session");
        assert_ne!(chunks[0].key, chunks[1].key);
    }

    #[test]
    fn chunk_identity_normalizes_existing_relative_paths() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("session.jsonl");
        std::fs::write(&path, "").expect("write session");
        let cwd = std::env::current_dir().expect("cwd");
        std::env::set_current_dir(dir.path()).expect("set cwd");
        let relative = test_conversation("session.jsonl", vec!["relative".to_string()]);
        let absolute = test_conversation(&path.to_string_lossy(), vec!["absolute".to_string()]);
        let chunks = build_chunks(&[&relative, &absolute], ChunkConfig::default());
        std::env::set_current_dir(cwd).expect("restore cwd");

        assert_eq!(chunks[0].key, chunks[1].key);
    }

    #[test]
    fn long_unicode_chunks_split_on_char_boundaries() {
        let text = "é".repeat(8);
        let conversation = test_conversation("/projects/project-a/session-1.jsonl", vec![text]);
        let config = ChunkConfig {
            target_chars: 5,
            overlap_chars: 2,
            context_turns: 0,
        };

        let chunks = build_chunks(&[&conversation], config);

        assert!(chunks.len() > 1);
        assert!(
            chunks
                .iter()
                .all(|chunk| chunk.text.is_char_boundary(chunk.text.len()))
        );
    }
}
