use crate::error::{AppError, Result};
use crate::semantic::embed::SemanticEmbedder;
use crate::semantic::types::DEFAULT_EMBEDDING_BATCH_SIZE;
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use std::path::PathBuf;

pub struct FastembedEmbedder {
    model: TextEmbedding,
}

impl FastembedEmbedder {
    pub fn new(cache_dir: PathBuf) -> Result<Self> {
        Self::new_with_download_progress(cache_dir, true)
    }

    pub fn new_quiet(cache_dir: PathBuf) -> Result<Self> {
        Self::new_with_download_progress(cache_dir, false)
    }

    fn new_with_download_progress(
        cache_dir: PathBuf,
        show_download_progress: bool,
    ) -> Result<Self> {
        let model = TextEmbedding::try_new(
            TextInitOptions::new(EmbeddingModel::BGESmallENV15)
                .with_cache_dir(cache_dir)
                .with_show_download_progress(show_download_progress),
        )
        .map_err(to_config_error)?;
        Ok(Self { model })
    }
}

impl SemanticEmbedder for FastembedEmbedder {
    fn embed_passages(&mut self, passages: &[String]) -> Result<Vec<Vec<f32>>> {
        self.model
            .embed(
                prefixed_passages(passages),
                Some(DEFAULT_EMBEDDING_BATCH_SIZE),
            )
            .map_err(to_config_error)
    }

    fn embed_query(&mut self, query: &str) -> Result<Option<Vec<f32>>> {
        let embeddings = self
            .model
            .embed(
                vec![prefixed_query(query)],
                Some(DEFAULT_EMBEDDING_BATCH_SIZE),
            )
            .map_err(to_config_error)?;
        Ok(embeddings.first().cloned())
    }
}

pub fn prefixed_query(query: &str) -> String {
    format!("query: {query}")
}

pub fn prefixed_passages(passages: &[String]) -> Vec<String> {
    passages
        .iter()
        .map(|passage| format!("passage: {passage}"))
        .collect()
}

fn to_config_error(err: impl std::fmt::Display) -> AppError {
    AppError::ConfigError(err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefixes_query_for_fastembed() {
        assert_eq!(prefixed_query("rust cache"), "query: rust cache");
    }

    #[test]
    fn prefixes_passages_for_fastembed() {
        assert_eq!(
            prefixed_passages(&["one".to_string(), "two".to_string()]),
            vec!["passage: one".to_string(), "passage: two".to_string()]
        );
    }
}
