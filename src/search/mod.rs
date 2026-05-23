pub mod lexical;
pub mod literal;
pub mod query;

pub use crate::text_match::{is_word_separator, normalize_for_search};
pub use lexical::{
    SearchableConversation, is_uuid, precompute_search_text, score_text_debug, search,
};
