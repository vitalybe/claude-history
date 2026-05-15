use crate::error::Result;

pub trait SemanticEmbedder {
    fn embed_passages(&mut self, passages: &[String]) -> Result<Vec<Vec<f32>>>;
    fn embed_query(&mut self, query: &str) -> Result<Option<Vec<f32>>>;
}
