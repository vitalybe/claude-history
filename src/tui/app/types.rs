use crate::search::normalize_for_search;
use crate::semantic::types::{SemanticExplanation, SemanticScoreBreakdown};
use crate::tui::viewer::{
    MessageRange, RenderableEntry, RenderedLine, ToolDisplayMode, ToolOutputId,
};
use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Arc;

/// Result of running the TUI
pub enum Action {
    Select(PathBuf),
    Delete(PathBuf),
    Resume(PathBuf),
    ForkResume(PathBuf),
    Quit,
}

/// Dialog overlay mode (for confirmations, menus)
#[derive(Clone, Debug, PartialEq)]
pub enum DialogMode {
    /// No dialog shown
    None,
    /// Confirming deletion of the selected conversation
    ConfirmDelete,
    /// Export menu (save to file)
    ExportMenu { selected: usize },
    /// Yank menu (copy to clipboard)
    YankMenu { selected: usize },
    /// Help overlay showing keyboard shortcuts
    Help { scroll: usize },
    /// Semantic result debug details
    SemanticDebug,
    /// Rename the selected conversation
    Rename { input: String, cursor: usize },
}

/// Main application mode
#[derive(Clone, Debug)]
#[allow(clippy::large_enum_variant)]
pub enum AppMode {
    /// List mode - browsing conversations
    List,
    /// View mode - reading a conversation
    View(ViewState),
}

/// State for the conversation viewer
#[derive(Clone, Debug)]
pub struct ViewState {
    /// Path to the conversation file (stable identity)
    pub conversation_path: PathBuf,
    /// Parsed renderable entries for the currently open view.
    pub parsed_entries: Option<Arc<Vec<RenderableEntry>>>,
    /// Current scroll position (line offset)
    pub scroll_offset: usize,
    /// Pre-rendered conversation lines
    pub rendered_lines: Vec<RenderedLine>,
    /// Total content height in lines
    pub total_lines: usize,
    /// Tool display mode (hidden/truncated/full)
    pub tool_display: ToolDisplayMode,
    /// Whether to show thinking blocks
    pub show_thinking: bool,
    /// Whether to show timing information (timestamps + durations)
    pub show_timing: bool,
    /// Content width used for rendering (for resize detection)
    pub content_width: usize,
    /// Search mode state
    pub search_mode: ViewSearchMode,
    /// Current search query
    pub search_query: String,
    /// Line indices with matches
    pub search_matches: Vec<usize>,
    /// Current match index
    pub current_match: usize,
    /// Message boundary ranges from rendering
    pub message_ranges: Vec<MessageRange>,
    /// Currently focused message index
    pub focused_message: Option<usize>,
    /// Whether message navigation mode is active (shows gutter indicator)
    pub message_nav_active: bool,
    /// Tool outputs expanded independently from global tool display mode
    pub expanded_tool_outputs: BTreeSet<ToolOutputId>,
    /// Tool output currently under the mouse cursor
    pub hovered_tool_output: Option<ToolOutputId>,
}

/// Search mode within view
#[derive(Clone, Debug, PartialEq, Default)]
pub enum ViewSearchMode {
    #[default]
    Off,
    /// Typing search query
    Typing,
    /// Search active, navigating results
    Active,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TuiSearchOptions {
    pub semantic_search_default: bool,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ListSearchMode {
    #[default]
    Lexical,
    Semantic,
}

impl ListSearchMode {
    pub fn label(self) -> &'static str {
        match self {
            ListSearchMode::Lexical => "lex",
            ListSearchMode::Semantic => "sem",
        }
    }
}

pub const LIST_LINES_PER_ITEM: usize = 3;

pub fn list_lines_per_item(mode: ListSearchMode, query: &str) -> usize {
    let query_normalized: String = normalize_for_search(query.trim())
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    if query_normalized.is_empty() || mode == ListSearchMode::Semantic {
        LIST_LINES_PER_ITEM
    } else {
        4
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub enum SemanticProgress {
    #[default]
    Idle,
    InitializingModel,
    CacheReady,
    Embedding {
        completed: usize,
        total: usize,
    },
    Ranking,
    Complete,
    EmptyCorpus,
    Failed,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SemanticResultMetadata {
    pub score_breakdown: SemanticScoreBreakdown,
    pub explanation: SemanticExplanation,
}

/// Loading state for the TUI
#[derive(Clone, Debug)]
pub enum LoadingState {
    /// Still loading conversations
    Loading { loaded: usize },
    /// All conversations loaded and ready
    Ready,
}
