pub mod evidence;
pub mod lexical;
pub mod literal;
pub mod mode;
pub mod query;

pub use crate::text_match::{is_word_separator, normalize_for_search};
pub use evidence::{LexicalEvidence, build_lexical_evidence};
pub use lexical::{
    LexicalDebugSearch, SearchableConversation, debug_search, is_uuid, precompute_search_text,
    score_text_debug, search,
};
