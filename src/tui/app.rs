use crate::config::KeyBindings;
use crate::debug_log;
use crate::error::{AppError, Result};
use crate::history::{
    Conversation, LoaderMessage, format_short_name_from_path, process_conversation_file,
};
use crate::search::{self, SearchableConversation};
#[cfg(test)]
use crate::semantic::types::{SemanticExplanation, SemanticScoreBreakdown};
#[cfg(test)]
use crate::tui::semantic_worker::{SemanticSearchMessage, SemanticWorkerCommand};
use crate::tui::ui;
use crate::tui::viewer::{MessageRange, RenderedLine, ToolDisplayMode, ToolOutputId};
#[cfg(test)]
use chrono::Local;
use crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind, KeyModifiers,
    MouseButton, MouseEventKind,
};
use crossterm::terminal::{self, EnterAlternateScreen, LeaveAlternateScreen};
use ratatui::prelude::*;
#[cfg(test)]
use std::collections::HashMap;
use std::collections::{BTreeSet, HashSet};
use std::io::{self, Stdout};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver};
use std::time::Duration;

mod list_state;
mod search_state;
mod types;
#[allow(unused_imports)]
use search_state::{SearchCommand, SearchResponse, SemanticSearchState, spawn_search_worker};
#[allow(unused_imports)]
pub use types::{
    Action, AppMode, DialogMode, LIST_LINES_PER_ITEM, ListSearchMode, LoadingState,
    SemanticProgress, SemanticResultMetadata, TuiSearchOptions, ViewSearchMode, ViewState,
    list_lines_per_item,
};

/// Export format options for menus
const EXPORT_OPTIONS: [&str; 4] = [
    "Ledger (formatted)",
    "Plain text",
    "Markdown",
    "JSONL (raw)",
];

/// App state
pub struct App {
    /// All loaded conversations
    conversations: Vec<Conversation>,
    /// Shared search data snapshot for background workers
    conversations_snapshot: Arc<Vec<Conversation>>,
    /// Shared semantic conversation snapshot for background workers
    semantic_conversations_snapshot: Arc<Vec<Arc<Conversation>>>,
    /// Version of the semantic corpus snapshot
    semantic_corpus_version: u64,
    /// Version of the semantic scope snapshot
    semantic_scope_version: u64,
    /// Last corpus version sent to the semantic worker
    semantic_sent_corpus_version: u64,
    /// Last scope signature sent to the semantic worker
    semantic_sent_scope_signature: Option<(u64, Arc<Vec<usize>>)>,
    /// Precomputed search data
    searchable: Vec<SearchableConversation>,
    /// Indices into conversations, sorted by current score
    filtered: Vec<usize>,
    /// Currently selected index into filtered (None if no results)
    selected: Option<usize>,
    /// Current search query
    query: String,
    /// Cursor position in query (character index, not byte)
    cursor_pos: usize,
    /// Loading state
    loading_state: LoadingState,
    /// Current dialog overlay (confirm, menu)
    dialog_mode: DialogMode,
    /// Main app mode (list or view)
    app_mode: AppMode,
    /// Status message with timestamp for auto-clear
    status_message: Option<(String, std::time::Instant)>,
    /// Persistent view setting: tool display mode
    tool_display: ToolDisplayMode,
    /// Persistent view setting: whether to show thinking blocks
    show_thinking: bool,
    /// Persistent view setting: whether to show timing information
    show_timing: bool,
    /// Whether the app is running in single file mode (direct input, no list)
    single_file_mode: bool,
    /// Configurable keybindings
    keys: KeyBindings,
    /// Whether workspace filter is active (only show current project's conversations)
    workspace_filter: bool,
    /// The encoded project directory name for the current workspace (for filtering)
    current_project_dir_name: Option<String>,
    /// Exact project names hidden from list-mode display
    excluded_projects: HashSet<String>,
    /// Channel to send commands to the background search worker
    search_tx: mpsc::Sender<SearchCommand>,
    /// Channel to receive results from the background search worker
    search_rx: mpsc::Receiver<SearchResponse>,
    /// Monotonic generation counter for search requests
    search_generation: u64,
    /// Whether a search is currently in-flight on the worker thread
    search_in_flight: bool,
    /// Current list search mode
    list_search_mode: ListSearchMode,
    /// Semantic TUI state
    semantic_search: SemanticSearchState,
}

impl App {
    /// Create a new app with all conversations pre-loaded
    #[allow(dead_code)]
    pub fn new(
        conversations: Vec<Conversation>,
        tool_display: ToolDisplayMode,
        show_thinking: bool,
        keys: KeyBindings,
        exclude_projects: Vec<String>,
    ) -> Self {
        Self::new_with_options(
            conversations,
            tool_display,
            show_thinking,
            keys,
            exclude_projects,
            TuiSearchOptions::default(),
        )
    }

    #[allow(dead_code)]
    pub fn new_with_options(
        conversations: Vec<Conversation>,
        tool_display: ToolDisplayMode,
        show_thinking: bool,
        keys: KeyBindings,
        exclude_projects: Vec<String>,
        search_options: TuiSearchOptions,
    ) -> Self {
        let searchable = search::precompute_search_text(&conversations);
        let excluded_projects = exclude_projects.into_iter().collect();
        let filtered = list_state::filter_conversation_indices(
            0..conversations.len(),
            &conversations,
            &excluded_projects,
            false,
            None,
        );
        let selected = if filtered.is_empty() { None } else { Some(0) };
        let (search_tx, search_rx) = spawn_search_worker();

        let conversations_snapshot = Arc::new(conversations.clone());
        let semantic_conversations_snapshot = Arc::new(
            conversations
                .iter()
                .cloned()
                .map(Arc::new)
                .collect::<Vec<_>>(),
        );

        // Send initial data to the worker
        let _ = search_tx.send(SearchCommand::UpdateData {
            conversations: conversations_snapshot.clone(),
            searchable: Arc::new(searchable.clone()),
        });

        Self {
            conversations_snapshot,
            semantic_conversations_snapshot,
            semantic_corpus_version: 1,
            semantic_scope_version: 0,
            semantic_sent_corpus_version: 0,
            semantic_sent_scope_signature: None,
            conversations,
            searchable,
            filtered,
            selected,
            query: String::new(),
            cursor_pos: 0,
            loading_state: LoadingState::Ready,
            dialog_mode: DialogMode::None,
            app_mode: AppMode::List,
            status_message: None,
            tool_display,
            show_thinking,
            show_timing: false,
            single_file_mode: false,
            keys,
            workspace_filter: false,
            current_project_dir_name: None,
            excluded_projects,
            search_tx,
            search_rx,
            search_generation: 0,
            search_in_flight: false,
            list_search_mode: if search_options.semantic_search_default {
                ListSearchMode::Semantic
            } else {
                ListSearchMode::Lexical
            },
            semantic_search: SemanticSearchState {
                available: true,
                ..Default::default()
            },
        }
    }

    /// Create a new app in loading state
    pub fn new_loading_with_options(
        tool_display: ToolDisplayMode,
        show_thinking: bool,
        keys: KeyBindings,
        workspace_filter: bool,
        current_project_dir_name: Option<String>,
        exclude_projects: Vec<String>,
        search_options: TuiSearchOptions,
    ) -> Self {
        let (search_tx, search_rx) = spawn_search_worker();
        let excluded_projects = exclude_projects.into_iter().collect();

        Self {
            conversations: Vec::new(),
            conversations_snapshot: Arc::new(Vec::new()),
            semantic_conversations_snapshot: Arc::new(Vec::new()),
            semantic_corpus_version: 1,
            semantic_scope_version: 0,
            semantic_sent_corpus_version: 0,
            semantic_sent_scope_signature: None,
            searchable: Vec::new(),
            filtered: Vec::new(),
            selected: None,
            query: String::new(),
            cursor_pos: 0,
            loading_state: LoadingState::Loading { loaded: 0 },
            dialog_mode: DialogMode::None,
            app_mode: AppMode::List,
            status_message: None,
            tool_display,
            show_thinking,
            show_timing: false,
            single_file_mode: false,
            keys,
            workspace_filter,
            current_project_dir_name,
            excluded_projects,
            search_tx,
            search_rx,
            search_generation: 0,
            search_in_flight: false,
            list_search_mode: if search_options.semantic_search_default {
                ListSearchMode::Semantic
            } else {
                ListSearchMode::Lexical
            },
            semantic_search: SemanticSearchState {
                available: true,
                ..Default::default()
            },
        }
    }

    /// Create a new app for viewing a single file directly
    pub fn new_single_file(
        path: PathBuf,
        tool_display: ToolDisplayMode,
        show_thinking: bool,
        keys: KeyBindings,
    ) -> Self {
        let (search_tx, search_rx) = spawn_search_worker();

        // Parse using the same parser as the main list
        let modified = std::fs::metadata(&path).and_then(|m| m.modified()).ok();

        let mut conversations = Vec::new();
        let mut filtered = Vec::new();
        let mut selected = None;

        if let Ok(Some(mut conv)) = process_conversation_file(path.clone(), modified, None) {
            // Set project_name the same way as the loader does
            let project_path = conv.cwd.clone().unwrap_or_else(|| path.clone());
            conv.project_name = Some(format_short_name_from_path(&project_path));

            conversations.push(conv);
            filtered.push(0);
            selected = Some(0);
        }

        Self {
            conversations_snapshot: Arc::new(conversations.clone()),
            semantic_conversations_snapshot: Arc::new(
                conversations
                    .iter()
                    .cloned()
                    .map(Arc::new)
                    .collect::<Vec<_>>(),
            ),
            semantic_corpus_version: 1,
            semantic_scope_version: 0,
            semantic_sent_corpus_version: 0,
            semantic_sent_scope_signature: None,
            conversations,
            searchable: Vec::new(),
            filtered,
            selected,
            query: String::new(),
            cursor_pos: 0,
            loading_state: LoadingState::Ready,
            dialog_mode: DialogMode::None,
            app_mode: AppMode::View(ViewState {
                conversation_path: path,
                parsed_entries: None,
                scroll_offset: 0,
                rendered_lines: Vec::new(),
                total_lines: 0,
                tool_display,
                show_thinking,
                show_timing: false,
                content_width: 0,
                search_mode: ViewSearchMode::Off,
                search_query: String::new(),
                search_matches: Vec::new(),
                current_match: 0,
                message_ranges: Vec::new(),
                focused_message: None,
                message_nav_active: false,
                expanded_tool_outputs: BTreeSet::new(),
                hovered_tool_output: None,
            }),
            status_message: None,
            tool_display,
            show_thinking,
            show_timing: false,
            single_file_mode: true,
            keys,
            workspace_filter: false,
            current_project_dir_name: None,
            excluded_projects: HashSet::new(),
            search_tx,
            search_rx,
            search_generation: 0,
            search_in_flight: false,
            list_search_mode: ListSearchMode::Lexical,
            semantic_search: SemanticSearchState::default(),
        }
    }

    pub fn keys(&self) -> &KeyBindings {
        &self.keys
    }

    /// Append a batch of conversations during loading
    /// Note: Does NOT precompute search text - that's deferred to finish_loading
    pub fn append_conversations(&mut self, new_convs: Vec<Conversation>) {
        let start_idx = self.conversations.len();
        self.conversations.extend(new_convs);
        let end_idx = self.conversations.len();

        let new_filtered = self.filter_indices(start_idx..end_idx);
        self.filtered.extend(new_filtered);

        // Select first item if nothing selected yet
        if self.selected.is_none() && !self.filtered.is_empty() {
            self.selected = Some(0);
        }

        // Update loading count
        self.loading_state = LoadingState::Loading {
            loaded: self.conversations.len(),
        };
    }

    /// Mark loading as complete: sort, precompute search, and transition to Ready
    pub fn finish_loading(&mut self) {
        // Sort all conversations by timestamp (newest first)
        self.conversations
            .sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

        // Reindex after sorting
        for (idx, conv) in self.conversations.iter_mut().enumerate() {
            conv.index = idx;
        }

        self.conversations_snapshot = Arc::new(self.conversations.clone());
        self.rebuild_semantic_conversations_snapshot();

        // Now precompute search text (only once, at the end)
        self.searchable = search::precompute_search_text(&self.conversations);

        // Send data snapshot to the background search worker
        let _ = self.search_tx.send(SearchCommand::UpdateData {
            conversations: self.conversations_snapshot.clone(),
            searchable: Arc::new(self.searchable.clone()),
        });

        self.loading_state = LoadingState::Ready;

        self.invalidate_search_generation();

        // Apply filter (handles query, exclusions, and workspace filter)
        self.update_filter();
        if self.list_search_mode == ListSearchMode::Semantic && !self.query.trim().is_empty() {
            self.dispatch_search();
        } else {
            self.prewarm_semantic_cache();
        }
    }

    /// Consume the app and return its conversations
    pub fn into_conversations(self) -> Vec<Conversation> {
        self.conversations
    }

    pub fn loading_state(&self) -> &LoadingState {
        &self.loading_state
    }

    pub fn is_loading(&self) -> bool {
        matches!(self.loading_state, LoadingState::Loading { .. })
    }

    // Getters for UI access
    pub fn filtered(&self) -> &[usize] {
        &self.filtered
    }

    pub fn conversations(&self) -> &[Conversation] {
        &self.conversations
    }

    pub fn selected(&self) -> Option<usize> {
        self.selected
    }

    pub fn query(&self) -> &str {
        &self.query
    }

    pub fn dialog_mode(&self) -> &DialogMode {
        &self.dialog_mode
    }

    #[cfg(test)]
    pub fn set_dialog_mode_for_test(&mut self, dialog_mode: DialogMode) {
        self.dialog_mode = dialog_mode;
    }

    pub fn app_mode(&self) -> &AppMode {
        &self.app_mode
    }

    pub fn status_message(&self) -> Option<&(String, std::time::Instant)> {
        self.status_message.as_ref()
    }

    /// Returns how long until the active status message expires, if any
    pub fn status_message_remaining(&self) -> Option<Duration> {
        const STATUS_TTL: Duration = Duration::from_secs(3);
        self.status_message
            .as_ref()
            .and_then(|(_, instant)| STATUS_TTL.checked_sub(instant.elapsed()))
    }

    pub fn cursor_pos(&self) -> usize {
        self.cursor_pos
    }

    pub fn is_single_file_mode(&self) -> bool {
        self.single_file_mode
    }

    pub fn workspace_filter(&self) -> bool {
        self.workspace_filter
    }

    pub fn list_search_mode(&self) -> ListSearchMode {
        self.list_search_mode
    }

    pub fn semantic_search_available(&self) -> bool {
        self.semantic_search.available
    }

    pub fn has_project_context(&self) -> bool {
        self.current_project_dir_name.is_some()
    }

    pub fn semantic_toggle_available(&self) -> bool {
        self.semantic_search.available
            && !self
                .keys
                .rename
                .matches(KeyCode::Char('t'), KeyModifiers::CONTROL)
            && !self
                .keys
                .delete
                .matches(KeyCode::Char('t'), KeyModifiers::CONTROL)
            && !self
                .keys
                .resume
                .matches(KeyCode::Char('t'), KeyModifiers::CONTROL)
            && !self
                .keys
                .fork
                .matches(KeyCode::Char('t'), KeyModifiers::CONTROL)
    }

    /// Move cursor left by one character
    fn cursor_left(&mut self) {
        if self.cursor_pos > 0 {
            self.cursor_pos -= 1;
        }
    }

    /// Move cursor right by one character
    fn cursor_right(&mut self) {
        let len = self.query.chars().count();
        if self.cursor_pos < len {
            self.cursor_pos += 1;
        }
    }

    /// Move cursor to the beginning of the line (Ctrl+A)
    fn cursor_home(&mut self) {
        self.cursor_pos = 0;
    }

    /// Move cursor to the end of the line (Ctrl+E)
    fn cursor_end(&mut self) {
        self.cursor_pos = self.query.chars().count();
    }

    /// Move cursor one word to the left (Ctrl+Left / Alt+B)
    fn cursor_word_left(&mut self) {
        let chars: Vec<char> = self.query.chars().collect();
        let mut pos = self.cursor_pos.min(chars.len());
        // Skip separators to the left
        while pos > 0 && search::is_word_separator(chars[pos - 1]) {
            pos -= 1;
        }
        // Skip non-separators (the word)
        while pos > 0 && !search::is_word_separator(chars[pos - 1]) {
            pos -= 1;
        }
        self.cursor_pos = pos;
    }

    /// Move cursor one word to the right (Ctrl+Right / Alt+F)
    fn cursor_word_right(&mut self) {
        let chars: Vec<char> = self.query.chars().collect();
        let len = chars.len();
        let mut pos = self.cursor_pos.min(len);
        // Skip non-separators (the word)
        while pos < len && !search::is_word_separator(chars[pos]) {
            pos += 1;
        }
        // Skip separators
        while pos < len && search::is_word_separator(chars[pos]) {
            pos += 1;
        }
        self.cursor_pos = pos;
    }

    /// Delete from cursor to end of line (Ctrl+K). Returns true if modified.
    fn kill_to_end(&mut self) -> bool {
        let len = self.query.chars().count();
        if self.cursor_pos >= len {
            return false;
        }
        let byte_pos = self
            .query
            .char_indices()
            .nth(self.cursor_pos)
            .map(|(i, _)| i)
            .unwrap_or(self.query.len());
        self.query.truncate(byte_pos);
        true
    }

    /// Delete from beginning of line to cursor (Ctrl+U). Returns true if modified.
    fn kill_to_start(&mut self) -> bool {
        if self.cursor_pos == 0 {
            return false;
        }
        let byte_pos = self
            .query
            .char_indices()
            .nth(self.cursor_pos)
            .map(|(i, _)| i)
            .unwrap_or(self.query.len());
        self.query.replace_range(..byte_pos, "");
        self.cursor_pos = 0;
        true
    }

    /// Delete the word before the cursor (Ctrl+W behavior).
    /// Returns true if the query was modified.
    fn delete_word_backwards(&mut self) -> bool {
        let chars: Vec<char> = self.query.chars().collect();
        let cursor = self.cursor_pos.min(chars.len());
        if cursor == 0 {
            return false;
        }

        let mut new_pos = cursor;

        // First, consume any separators to the left of cursor
        while new_pos > 0 && search::is_word_separator(chars[new_pos - 1]) {
            new_pos -= 1;
        }

        // Then, consume non-separators (the actual word)
        while new_pos > 0 && !search::is_word_separator(chars[new_pos - 1]) {
            new_pos -= 1;
        }

        if new_pos == cursor {
            return false;
        }

        // Convert char indices to byte indices for safe string manipulation
        let start_byte = self
            .query
            .char_indices()
            .nth(new_pos)
            .map(|(i, _)| i)
            .unwrap_or(0);

        let end_byte = self
            .query
            .char_indices()
            .nth(cursor)
            .map(|(i, _)| i)
            .unwrap_or(self.query.len());

        self.query.replace_range(start_byte..end_byte, "");
        self.cursor_pos = new_pos;
        true
    }

    /// Handle a key event during confirmation mode
    fn handle_confirm_key(&mut self, code: KeyCode) -> Option<Action> {
        match code {
            KeyCode::Char('y') | KeyCode::Char('Y') => {
                self.dialog_mode = DialogMode::None;
                self.get_selected_path().map(Action::Delete)
            }
            KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
                self.dialog_mode = DialogMode::None;
                None
            }
            _ => None,
        }
    }

    /// Handle a key event during export/yank menu mode
    fn handle_menu_key(&mut self, code: KeyCode) -> Option<Action> {
        let (selected, is_yank) = match &mut self.dialog_mode {
            DialogMode::ExportMenu { selected } => (selected, false),
            DialogMode::YankMenu { selected } => (selected, true),
            _ => return None,
        };

        match code {
            // Navigate up
            KeyCode::Up | KeyCode::Char('k') => {
                *selected = selected.saturating_sub(1);
                None
            }
            // Navigate down
            KeyCode::Down | KeyCode::Char('j') => {
                *selected = (*selected + 1).min(EXPORT_OPTIONS.len() - 1);
                None
            }
            // Number keys for direct selection
            KeyCode::Char('1') => {
                self.perform_export(0, is_yank);
                self.dialog_mode = DialogMode::None;
                None
            }
            KeyCode::Char('2') => {
                self.perform_export(1, is_yank);
                self.dialog_mode = DialogMode::None;
                None
            }
            KeyCode::Char('3') => {
                self.perform_export(2, is_yank);
                self.dialog_mode = DialogMode::None;
                None
            }
            KeyCode::Char('4') => {
                self.perform_export(3, is_yank);
                self.dialog_mode = DialogMode::None;
                None
            }
            // Enter to select current option
            KeyCode::Enter => {
                let sel = *selected;
                self.perform_export(sel, is_yank);
                self.dialog_mode = DialogMode::None;
                None
            }
            // Escape to cancel
            KeyCode::Esc => {
                self.dialog_mode = DialogMode::None;
                None
            }
            _ => None,
        }
    }

    /// Handle a key event during help overlay mode
    fn handle_help_key(&mut self, code: KeyCode, viewport_height: usize) -> Option<Action> {
        let DialogMode::Help { scroll } = &mut self.dialog_mode else {
            return None;
        };

        match code {
            KeyCode::Char('?') | KeyCode::Char('q') | KeyCode::Esc => {
                self.dialog_mode = DialogMode::None;
                None
            }
            KeyCode::Down | KeyCode::Char('j') => {
                *scroll = scroll.saturating_add(1);
                None
            }
            KeyCode::Up | KeyCode::Char('k') => {
                *scroll = scroll.saturating_sub(1);
                None
            }
            KeyCode::PageDown | KeyCode::Char('d') => {
                *scroll = scroll.saturating_add(viewport_height.max(1));
                None
            }
            KeyCode::PageUp | KeyCode::Char('u') => {
                *scroll = scroll.saturating_sub(viewport_height.max(1));
                None
            }
            KeyCode::Home | KeyCode::Char('g') => {
                *scroll = 0;
                None
            }
            _ => None,
        }
    }

    fn start_rename(&mut self) {
        let Some(idx) = self.get_selected_conversation_index() else {
            return;
        };
        let input = self.conversations[idx]
            .custom_title
            .clone()
            .unwrap_or_default();
        let cursor = input.chars().count();
        self.dialog_mode = DialogMode::Rename { input, cursor };
    }

    fn handle_rename_key(&mut self, code: KeyCode, modifiers: KeyModifiers) -> Option<Action> {
        match code {
            KeyCode::Esc => {
                self.dialog_mode = DialogMode::None;
            }
            KeyCode::Char('c') if modifiers.contains(KeyModifiers::CONTROL) => {
                self.dialog_mode = DialogMode::None;
            }
            KeyCode::Enter => self.submit_rename(),
            KeyCode::Left => {
                if let DialogMode::Rename { cursor, .. } = &mut self.dialog_mode {
                    *cursor = cursor.saturating_sub(1);
                }
            }
            KeyCode::Right => {
                if let DialogMode::Rename { input, cursor } = &mut self.dialog_mode {
                    *cursor = (*cursor + 1).min(input.chars().count());
                }
            }
            KeyCode::Char('u') if modifiers.contains(KeyModifiers::CONTROL) => {
                if let DialogMode::Rename { input, cursor } = &mut self.dialog_mode {
                    input.clear();
                    *cursor = 0;
                }
            }
            KeyCode::Home | KeyCode::Char('a') if modifiers.contains(KeyModifiers::CONTROL) => {
                if let DialogMode::Rename { cursor, .. } = &mut self.dialog_mode {
                    *cursor = 0;
                }
            }
            KeyCode::End | KeyCode::Char('e') if modifiers.contains(KeyModifiers::CONTROL) => {
                if let DialogMode::Rename { input, cursor } = &mut self.dialog_mode {
                    *cursor = input.chars().count();
                }
            }
            KeyCode::Backspace => {
                if let DialogMode::Rename { input, cursor } = &mut self.dialog_mode
                    && *cursor > 0
                    && let Some((byte_pos, _)) = input.char_indices().nth(*cursor - 1)
                {
                    input.remove(byte_pos);
                    *cursor -= 1;
                }
            }
            KeyCode::Delete => {
                if let DialogMode::Rename { input, cursor } = &mut self.dialog_mode
                    && *cursor < input.chars().count()
                    && let Some((byte_pos, _)) = input.char_indices().nth(*cursor)
                {
                    input.remove(byte_pos);
                }
            }
            KeyCode::Char(ch) if modifiers.is_empty() || modifiers == KeyModifiers::SHIFT => {
                if let DialogMode::Rename { input, cursor } = &mut self.dialog_mode {
                    let byte_pos = input
                        .char_indices()
                        .nth(*cursor)
                        .map(|(i, _)| i)
                        .unwrap_or(input.len());
                    input.insert(byte_pos, ch);
                    *cursor += 1;
                }
            }
            _ => {}
        }
        None
    }

    fn submit_rename(&mut self) {
        let title = match &self.dialog_mode {
            DialogMode::Rename { input, .. } => input.trim().to_string(),
            _ => return,
        };
        let Some(idx) = self.get_selected_conversation_index() else {
            self.dialog_mode = DialogMode::None;
            return;
        };
        let path = self.conversations[idx].path.clone();

        match crate::history::append_session_rename(&path, &title)
            .and_then(|_| crate::history::process_conversation_file(path.clone(), None, None))
        {
            Ok(Some(mut conv)) => {
                conv.index = idx;
                conv.project_name = self.conversations[idx].project_name.clone();
                conv.project_path = self.conversations[idx].project_path.clone();
                self.conversations[idx] = conv;
                self.dialog_mode = DialogMode::None;
                self.status_message =
                    Some(("Session renamed".to_string(), std::time::Instant::now()));
                self.refresh_search_data();
                self.update_filter();
                if let Some(new_selected) = self
                    .filtered
                    .iter()
                    .position(|&i| self.conversations[i].path == path)
                {
                    self.selected = Some(new_selected);
                }
            }
            Ok(None) => {
                self.status_message = Some((
                    "Failed to rename: conversation became empty".to_string(),
                    std::time::Instant::now(),
                ));
            }
            Err(e) => {
                self.status_message = Some((
                    format!("Failed to rename: {}", e),
                    std::time::Instant::now(),
                ));
            }
        }
    }

    /// Perform export or yank operation
    fn perform_export(&mut self, option: usize, to_clipboard: bool) {
        let (path, options) = match &self.app_mode {
            AppMode::View(state) => (
                state.conversation_path.clone(),
                crate::tui::export::ExportOptions {
                    show_tools: state.tool_display.is_visible(),
                    show_thinking: state.show_thinking,
                },
            ),
            _ => return,
        };

        let format = match crate::tui::export::ExportFormat::from_index(option) {
            Some(f) => f,
            None => return,
        };

        let result = if to_clipboard {
            crate::tui::export::export_to_clipboard(&path, format, options)
        } else {
            crate::tui::export::export_to_file(&path, format, options)
        };

        self.status_message = Some((result.message, std::time::Instant::now()));
    }

    /// Handle a key event, returns Some(Action) if the app should exit
    /// viewport_height is the visible content area height for view mode scrolling
    pub fn handle_key(
        &mut self,
        code: KeyCode,
        modifiers: KeyModifiers,
        viewport_height: usize,
    ) -> Option<Action> {
        // Handle dialogs first
        match self.dialog_mode {
            DialogMode::ConfirmDelete => return self.handle_confirm_key(code),
            DialogMode::ExportMenu { .. } | DialogMode::YankMenu { .. } => {
                return self.handle_menu_key(code);
            }
            DialogMode::Help { .. } => return self.handle_help_key(code, viewport_height),
            DialogMode::SemanticDebug => {
                self.dialog_mode = DialogMode::None;
                return None;
            }
            DialogMode::Rename { .. } => return self.handle_rename_key(code, modifiers),
            DialogMode::None => {}
        }

        if self.list_search_mode == ListSearchMode::Semantic
            && self.semantic_result_metadata_for_selection().is_some()
            && matches!(code, KeyCode::Char('s'))
            && modifiers.contains(KeyModifiers::CONTROL)
        {
            self.dialog_mode = DialogMode::SemanticDebug;
            return None;
        }

        // Delegate based on app mode
        match &self.app_mode {
            AppMode::View(_) => self.handle_view_key(code, modifiers, viewport_height),
            AppMode::List => self.handle_list_key(code, modifiers, viewport_height),
        }
    }

    /// Handle key events in view mode
    fn handle_view_key(
        &mut self,
        code: KeyCode,
        modifiers: KeyModifiers,
        viewport_height: usize,
    ) -> Option<Action> {
        // First check if we're in search typing mode
        if let AppMode::View(ref state) = self.app_mode
            && state.search_mode == ViewSearchMode::Typing
        {
            return self.handle_search_typing_key(code, modifiers);
        }

        // Check configurable keybindings before the match block
        if self.keys.delete.matches(code, modifiers) {
            if !self.single_file_mode {
                self.dialog_mode = DialogMode::ConfirmDelete;
            }
            return None;
        }
        if self.keys.resume.matches(code, modifiers) {
            return if self.single_file_mode {
                None
            } else {
                self.get_selected_path().map(Action::Resume)
            };
        }
        if self.keys.fork.matches(code, modifiers) {
            return if self.single_file_mode {
                None
            } else {
                self.get_selected_path().map(Action::ForkResume)
            };
        }

        let state = match &mut self.app_mode {
            AppMode::View(s) => s,
            _ => return None,
        };

        let max_scroll = state.total_lines.saturating_sub(viewport_height);

        match code {
            // Exit view mode (or clear search if active)
            KeyCode::Esc => {
                // Exit message nav mode first
                if let AppMode::View(ref mut state) = self.app_mode
                    && state.message_nav_active
                {
                    state.message_nav_active = false;
                    return None;
                }
                // If search is active, clear it first before exiting view
                if let AppMode::View(ref mut state) = self.app_mode
                    && state.search_mode == ViewSearchMode::Active
                {
                    state.search_mode = ViewSearchMode::Off;
                    state.search_matches.clear();
                    state.search_query.clear();
                    return None;
                }
                // In single file mode, Esc quits the app
                if self.single_file_mode {
                    return Some(Action::Quit);
                }
                self.app_mode = AppMode::List;
                None
            }

            KeyCode::Char('q') => {
                // In single file mode, q quits the app
                if self.single_file_mode {
                    return Some(Action::Quit);
                }
                self.app_mode = AppMode::List;
                None
            }

            // Scroll down one line
            KeyCode::Down | KeyCode::Char('j') => {
                state.scroll_offset = (state.scroll_offset + 1).min(max_scroll);
                self.sync_focus_after_scroll(viewport_height);
                None
            }

            // Scroll up one line
            KeyCode::Up | KeyCode::Char('k') => {
                state.scroll_offset = state.scroll_offset.saturating_sub(1);
                self.sync_focus_after_scroll(viewport_height);
                None
            }

            // Jump to next message
            KeyCode::Char('J') | KeyCode::Char(']') => {
                self.focus_next_message(viewport_height);
                None
            }

            // Jump to previous message
            KeyCode::Char('K') | KeyCode::Char('[') => {
                self.focus_prev_message(viewport_height);
                None
            }

            // Scroll down half page
            KeyCode::Char('d') if !modifiers.contains(KeyModifiers::CONTROL) => {
                state.scroll_offset = (state.scroll_offset + viewport_height / 2).min(max_scroll);
                self.sync_focus_after_scroll(viewport_height);
                None
            }

            // Scroll up half page
            KeyCode::Char('u') if !modifiers.contains(KeyModifiers::CONTROL) => {
                let half_page = viewport_height / 2;
                state.scroll_offset = state.scroll_offset.saturating_sub(half_page);
                self.sync_focus_after_scroll(viewport_height);
                None
            }

            // Page down
            KeyCode::PageDown => {
                state.scroll_offset = (state.scroll_offset + viewport_height).min(max_scroll);
                self.sync_focus_after_scroll(viewport_height);
                None
            }

            // Page up
            KeyCode::PageUp => {
                state.scroll_offset = state.scroll_offset.saturating_sub(viewport_height);
                self.sync_focus_after_scroll(viewport_height);
                None
            }

            // Jump to top
            KeyCode::Char('g') | KeyCode::Home => {
                state.scroll_offset = 0;
                self.sync_focus_after_scroll(viewport_height);
                None
            }

            // Jump to bottom
            KeyCode::Char('G') | KeyCode::End => {
                state.scroll_offset = max_scroll;
                self.sync_focus_after_scroll(viewport_height);
                None
            }

            // Start search
            KeyCode::Char('/') => {
                self.start_view_search();
                None
            }

            // Next match
            KeyCode::Char('n') if !modifiers.contains(KeyModifiers::CONTROL) => {
                if let AppMode::View(ref state) = self.app_mode
                    && state.search_mode == ViewSearchMode::Active
                {
                    self.next_search_match(viewport_height);
                }
                None
            }

            // Previous match
            KeyCode::Char('N') => {
                if let AppMode::View(ref state) = self.app_mode
                    && state.search_mode == ViewSearchMode::Active
                {
                    self.prev_search_match(viewport_height);
                }
                None
            }

            // Toggle tools
            KeyCode::Char('t') => {
                self.toggle_view_tools(viewport_height);
                None
            }

            // Toggle thinking
            KeyCode::Char('T') => {
                self.toggle_view_thinking(viewport_height);
                None
            }

            // Toggle timing (timestamps + durations)
            KeyCode::Char('i') => {
                self.toggle_view_timing(viewport_height);
                None
            }

            // Show path
            KeyCode::Char('p') => {
                if let AppMode::View(ref state) = self.app_mode {
                    self.status_message = Some((
                        state.conversation_path.display().to_string(),
                        std::time::Instant::now(),
                    ));
                }
                None
            }

            // Copy path to clipboard
            KeyCode::Char('Y') => {
                if let AppMode::View(ref state) = self.app_mode {
                    let path_str = state.conversation_path.display().to_string();
                    match crate::tui::export::copy_to_system_clipboard(&path_str) {
                        Ok(()) => {
                            self.status_message = Some((
                                "Path copied to clipboard".to_string(),
                                std::time::Instant::now(),
                            ));
                        }
                        Err(e) => {
                            self.status_message = Some((e, std::time::Instant::now()));
                        }
                    }
                }
                None
            }

            // Copy session ID to clipboard
            KeyCode::Char('I') => {
                if let AppMode::View(ref state) = self.app_mode
                    && let Some(id) = state.conversation_path.file_stem().and_then(|s| s.to_str())
                {
                    match crate::tui::export::copy_to_system_clipboard(id) {
                        Ok(()) => {
                            self.status_message = Some((
                                "Session ID copied to clipboard".to_string(),
                                std::time::Instant::now(),
                            ));
                        }
                        Err(e) => {
                            self.status_message = Some((e, std::time::Instant::now()));
                        }
                    }
                }
                None
            }

            // Open export menu (save to file)
            KeyCode::Char('e') => {
                self.dialog_mode = DialogMode::ExportMenu { selected: 0 };
                None
            }

            // Yank: copy message if in nav mode, otherwise open yank menu
            KeyCode::Char('y') => {
                let nav_active = matches!(
                    self.app_mode,
                    AppMode::View(ViewState {
                        message_nav_active: true,
                        ..
                    })
                );
                if nav_active {
                    self.copy_focused_message(viewport_height);
                } else {
                    self.dialog_mode = DialogMode::YankMenu { selected: 0 };
                }
                None
            }

            // Open help overlay
            KeyCode::Char('?') => {
                self.dialog_mode = DialogMode::Help { scroll: 0 };
                None
            }

            // Ctrl+D - half page down (vim-style, same as 'd')
            KeyCode::Char('d') if modifiers.contains(KeyModifiers::CONTROL) => {
                state.scroll_offset = (state.scroll_offset + viewport_height / 2).min(max_scroll);
                self.sync_focus_after_scroll(viewport_height);
                None
            }

            // Ctrl+U - half page up (vim-style, same as 'u')
            KeyCode::Char('u') if modifiers.contains(KeyModifiers::CONTROL) => {
                let half_page = viewport_height / 2;
                state.scroll_offset = state.scroll_offset.saturating_sub(half_page);
                self.sync_focus_after_scroll(viewport_height);
                None
            }

            // Ctrl+C - quit the app
            KeyCode::Char('c') if modifiers.contains(KeyModifiers::CONTROL) => Some(Action::Quit),

            _ => None,
        }
    }

    /// Handle key events while typing a search query
    fn handle_search_typing_key(
        &mut self,
        code: KeyCode,
        modifiers: KeyModifiers,
    ) -> Option<Action> {
        match code {
            // Ctrl+C: cancel search
            KeyCode::Char('c') if modifiers.contains(KeyModifiers::CONTROL) => {
                if let AppMode::View(ref mut state) = self.app_mode {
                    state.search_mode = ViewSearchMode::Off;
                    state.search_query.clear();
                    state.search_matches.clear();
                }
                None
            }
            // Ctrl+U: clear entire query
            KeyCode::Char('u') if modifiers.contains(KeyModifiers::CONTROL) => {
                if let AppMode::View(ref mut state) = self.app_mode
                    && !state.search_query.is_empty()
                {
                    state.search_query.clear();
                    self.update_search_results();
                }
                None
            }
            // Ctrl+W: delete last word
            KeyCode::Char('w') if modifiers.contains(KeyModifiers::CONTROL) => {
                if let AppMode::View(ref mut state) = self.app_mode {
                    let trimmed = state.search_query.trim_end();
                    if let Some(last_space) = trimmed.rfind(|c: char| c.is_whitespace()) {
                        state.search_query.truncate(last_space + 1);
                    } else {
                        state.search_query.clear();
                    }
                }
                self.update_search_results();
                None
            }
            KeyCode::Char(c) => {
                if let AppMode::View(ref mut state) = self.app_mode {
                    state.search_query.push(c);
                }
                self.update_search_results();
                None
            }
            KeyCode::Backspace => {
                if let AppMode::View(ref mut state) = self.app_mode {
                    state.search_query.pop();
                }
                self.update_search_results();
                None
            }
            KeyCode::Enter => {
                if let AppMode::View(ref mut state) = self.app_mode {
                    if !state.search_matches.is_empty() {
                        state.search_mode = ViewSearchMode::Active;
                    } else {
                        state.search_mode = ViewSearchMode::Off;
                    }
                }
                None
            }
            KeyCode::Esc => {
                if let AppMode::View(ref mut state) = self.app_mode {
                    state.search_mode = ViewSearchMode::Off;
                    state.search_query.clear();
                    state.search_matches.clear();
                }
                None
            }
            _ => None,
        }
    }

    /// Handle key events in list mode
    fn handle_list_key(
        &mut self,
        code: KeyCode,
        modifiers: KeyModifiers,
        viewport_height: usize,
    ) -> Option<Action> {
        // During loading, allow navigation and typing but not Enter selection
        if self.is_loading() {
            return match code {
                KeyCode::Esc => {
                    if self.query.is_empty() {
                        Some(Action::Quit)
                    } else {
                        self.query.clear();
                        self.cursor_pos = 0;
                        self.dispatch_search();
                        None
                    }
                }
                KeyCode::Char('c') if modifiers.contains(KeyModifiers::CONTROL) => {
                    Some(Action::Quit)
                }
                KeyCode::Char('t') if modifiers.contains(KeyModifiers::CONTROL) => {
                    if self.semantic_toggle_available() {
                        self.toggle_list_search_mode();
                    }
                    None
                }
                // Ctrl+Left: move cursor one word left
                KeyCode::Left if modifiers.contains(KeyModifiers::CONTROL) => {
                    self.cursor_word_left();
                    None
                }
                // Ctrl+Right: move cursor one word right
                KeyCode::Right if modifiers.contains(KeyModifiers::CONTROL) => {
                    self.cursor_word_right();
                    None
                }
                KeyCode::Left => {
                    self.cursor_left();
                    None
                }
                KeyCode::Right => {
                    self.cursor_right();
                    None
                }
                KeyCode::Up => {
                    self.select_prev();
                    None
                }
                KeyCode::Down => {
                    self.select_next();
                    None
                }
                KeyCode::Char('n') if modifiers.contains(KeyModifiers::CONTROL) => {
                    self.select_next();
                    None
                }
                KeyCode::Char('p') if modifiers.contains(KeyModifiers::CONTROL) => {
                    self.select_prev();
                    None
                }
                // Ctrl+A: cursor to beginning of line
                KeyCode::Char('a') if modifiers.contains(KeyModifiers::CONTROL) => {
                    self.cursor_home();
                    None
                }
                // Ctrl+E: cursor to end of line
                KeyCode::Char('e') if modifiers.contains(KeyModifiers::CONTROL) => {
                    self.cursor_end();
                    None
                }
                // Ctrl+B: cursor left (emacs-style)
                KeyCode::Char('b') if modifiers.contains(KeyModifiers::CONTROL) => {
                    self.cursor_left();
                    None
                }
                // Ctrl+F: cursor right (emacs-style)
                KeyCode::Char('f') if modifiers.contains(KeyModifiers::CONTROL) => {
                    self.cursor_right();
                    None
                }
                // Alt+B: move cursor one word left
                KeyCode::Char('b') if modifiers.contains(KeyModifiers::ALT) => {
                    self.cursor_word_left();
                    None
                }
                // Alt+F: move cursor one word right
                KeyCode::Char('f') if modifiers.contains(KeyModifiers::ALT) => {
                    self.cursor_word_right();
                    None
                }
                // Ctrl+K: kill from cursor to end of line
                KeyCode::Char('k') if modifiers.contains(KeyModifiers::CONTROL) => {
                    self.kill_to_end();
                    None
                }
                // Ctrl+U: kill from beginning of line to cursor
                KeyCode::Char('u') if modifiers.contains(KeyModifiers::CONTROL) => {
                    self.kill_to_start();
                    None
                }
                KeyCode::PageUp => {
                    self.select_page_up();
                    None
                }
                KeyCode::PageDown => {
                    self.select_page_down();
                    None
                }
                KeyCode::Char('w') if modifiers.contains(KeyModifiers::CONTROL) => {
                    self.delete_word_backwards();
                    None
                }
                // Tab: toggle workspace/global filter
                KeyCode::Tab => {
                    self.toggle_workspace_filter();
                    None
                }
                // Open help overlay
                KeyCode::Char('?') => {
                    self.dialog_mode = DialogMode::Help { scroll: 0 };
                    None
                }
                // Allow typing during loading - query is buffered for when loading finishes
                KeyCode::Char(c) => {
                    // Insert at cursor position
                    let byte_pos = self
                        .query
                        .char_indices()
                        .nth(self.cursor_pos)
                        .map(|(i, _)| i)
                        .unwrap_or(self.query.len());
                    self.query.insert(byte_pos, c);
                    self.cursor_pos += 1;
                    None
                }
                KeyCode::Backspace => {
                    if self.cursor_pos > 0
                        && let Some((byte_pos, _)) =
                            self.query.char_indices().nth(self.cursor_pos - 1)
                    {
                        self.query.remove(byte_pos);
                        self.cursor_pos -= 1;
                    }
                    None
                }
                KeyCode::Delete => {
                    let len = self.query.chars().count();
                    if self.cursor_pos < len
                        && let Some((byte_pos, _)) = self.query.char_indices().nth(self.cursor_pos)
                    {
                        self.query.remove(byte_pos);
                    }
                    None
                }
                _ => None,
            };
        }

        // Check configurable keybindings before the match block
        if self.keys.delete.matches(code, modifiers) {
            if self.get_selected_path().is_some() {
                self.dialog_mode = DialogMode::ConfirmDelete;
            }
            return None;
        }
        if self.keys.resume.matches(code, modifiers) {
            return self.get_selected_path().map(Action::Resume);
        }
        if self.keys.fork.matches(code, modifiers) {
            return self.get_selected_path().map(Action::ForkResume);
        }

        // Normal handling when ready
        match code {
            _ if self.keys.rename.matches(code, modifiers) => {
                if self.get_selected_path().is_some() {
                    self.start_rename();
                }
                None
            }
            KeyCode::Esc => {
                if self.query.is_empty() {
                    Some(Action::Quit)
                } else {
                    self.query.clear();
                    self.cursor_pos = 0;
                    self.dispatch_search();
                    None
                }
            }
            // Enter now triggers view mode entry (handled in run loop)
            KeyCode::Enter => None,
            // Ctrl+Left: move cursor one word left
            KeyCode::Left if modifiers.contains(KeyModifiers::CONTROL) => {
                self.cursor_word_left();
                None
            }
            // Ctrl+Right: move cursor one word right
            KeyCode::Right if modifiers.contains(KeyModifiers::CONTROL) => {
                self.cursor_word_right();
                None
            }
            KeyCode::Left => {
                self.cursor_left();
                None
            }
            KeyCode::Right => {
                self.cursor_right();
                None
            }
            KeyCode::Up => {
                self.select_prev();
                None
            }
            KeyCode::Down => {
                self.select_next();
                None
            }
            KeyCode::Home => {
                self.select_first();
                None
            }
            KeyCode::End => {
                self.select_last();
                None
            }
            KeyCode::PageUp => {
                self.select_page_up();
                None
            }
            KeyCode::PageDown => {
                self.select_page_down();
                None
            }
            KeyCode::Char('c') if modifiers.contains(KeyModifiers::CONTROL) => Some(Action::Quit),
            KeyCode::Char('t') if modifiers.contains(KeyModifiers::CONTROL) => {
                if self.semantic_toggle_available() {
                    self.toggle_list_search_mode();
                }
                None
            }
            // Ctrl+A: cursor to beginning of line
            KeyCode::Char('a') if modifiers.contains(KeyModifiers::CONTROL) => {
                self.cursor_home();
                None
            }
            // Ctrl+E: cursor to end of line
            KeyCode::Char('e') if modifiers.contains(KeyModifiers::CONTROL) => {
                self.cursor_end();
                None
            }
            // Ctrl+B: cursor left (emacs-style)
            KeyCode::Char('b') if modifiers.contains(KeyModifiers::CONTROL) => {
                self.cursor_left();
                None
            }
            // Ctrl+F: cursor right (emacs-style)
            KeyCode::Char('f') if modifiers.contains(KeyModifiers::CONTROL) => {
                self.cursor_right();
                None
            }
            // Alt+B: move cursor one word left
            KeyCode::Char('b') if modifiers.contains(KeyModifiers::ALT) => {
                self.cursor_word_left();
                None
            }
            // Alt+F: move cursor one word right
            KeyCode::Char('f') if modifiers.contains(KeyModifiers::ALT) => {
                self.cursor_word_right();
                None
            }
            // Ctrl+K: kill from cursor to end of line
            KeyCode::Char('k') if modifiers.contains(KeyModifiers::CONTROL) => {
                if self.kill_to_end() {
                    self.dispatch_search();
                }
                None
            }
            KeyCode::Char('n') if modifiers.contains(KeyModifiers::CONTROL) => {
                self.select_next();
                None
            }
            KeyCode::Char('p') if modifiers.contains(KeyModifiers::CONTROL) => {
                self.select_prev();
                None
            }
            // Ctrl+D - half page down (vim-style)
            KeyCode::Char('d') if modifiers.contains(KeyModifiers::CONTROL) => {
                self.select_half_page_down(viewport_height);
                None
            }
            // Ctrl+U - kill from beginning of line to cursor (emacs-style)
            KeyCode::Char('u') if modifiers.contains(KeyModifiers::CONTROL) => {
                if self.kill_to_start() {
                    self.dispatch_search();
                }
                None
            }
            // Ctrl+O - select and exit (for scripting, --show-path)
            KeyCode::Char('o') if modifiers.contains(KeyModifiers::CONTROL) => {
                self.get_selected_path().map(Action::Select)
            }
            KeyCode::Char('w') if modifiers.contains(KeyModifiers::CONTROL) => {
                if self.delete_word_backwards() {
                    self.dispatch_search();
                }
                None
            }
            // Tab: toggle workspace/global filter
            KeyCode::Tab => {
                self.toggle_workspace_filter();
                None
            }
            // Open help overlay
            KeyCode::Char('?') => {
                self.dialog_mode = DialogMode::Help { scroll: 0 };
                None
            }
            KeyCode::Char(c) => {
                // Insert at cursor position
                let byte_pos = self
                    .query
                    .char_indices()
                    .nth(self.cursor_pos)
                    .map(|(i, _)| i)
                    .unwrap_or(self.query.len());
                self.query.insert(byte_pos, c);
                self.cursor_pos += 1;
                self.dispatch_search();
                None
            }
            KeyCode::Backspace => {
                let mut changed = false;
                if self.cursor_pos > 0
                    && let Some((byte_pos, _)) = self.query.char_indices().nth(self.cursor_pos - 1)
                {
                    self.query.remove(byte_pos);
                    self.cursor_pos -= 1;
                    changed = true;
                }
                if changed {
                    self.dispatch_search();
                }
                None
            }
            KeyCode::Delete => {
                let mut changed = false;
                let len = self.query.chars().count();
                if self.cursor_pos < len
                    && let Some((byte_pos, _)) = self.query.char_indices().nth(self.cursor_pos)
                {
                    self.query.remove(byte_pos);
                    changed = true;
                }
                if changed {
                    self.dispatch_search();
                }
                None
            }
            _ => None,
        }
    }

    /// Enter view mode for the currently selected conversation
    pub fn enter_view_mode(&mut self, content_width: usize) {
        use crate::tui::viewer::{
            RenderOptions, parse_conversation_file, render_parsed_conversation,
        };

        let Some(selected) = self.selected else {
            return;
        };
        let Some(&conv_idx) = self.filtered.get(selected) else {
            return;
        };
        let path = self.conversations[conv_idx].path.clone();

        let options = RenderOptions {
            tool_display: self.tool_display,
            show_thinking: self.show_thinking,
            show_timing: self.show_timing,
            content_width,
            expanded_tool_outputs: BTreeSet::new(),
        };

        match parse_conversation_file(&path) {
            Ok(entries) => {
                let entries = Arc::new(entries);
                let rendered = render_parsed_conversation(&entries, &options);
                let total_lines = rendered.lines.len();
                let first_msg = if rendered.messages.is_empty() {
                    None
                } else {
                    Some(0)
                };
                self.app_mode = AppMode::View(ViewState {
                    conversation_path: path,
                    parsed_entries: Some(entries),
                    scroll_offset: 0,
                    rendered_lines: rendered.lines,
                    total_lines,
                    tool_display: self.tool_display,
                    show_thinking: self.show_thinking,
                    show_timing: self.show_timing,
                    content_width,
                    search_mode: ViewSearchMode::Off,
                    search_query: String::new(),
                    search_matches: Vec::new(),
                    current_match: 0,
                    message_ranges: rendered.messages,
                    focused_message: first_msg,
                    message_nav_active: false,
                    expanded_tool_outputs: BTreeSet::new(),
                    hovered_tool_output: None,
                });
            }
            Err(e) => {
                self.status_message =
                    Some((format!("Failed to open: {}", e), std::time::Instant::now()));
            }
        }
    }

    /// Exit view mode and return to list
    pub fn exit_view_mode(&mut self) {
        self.app_mode = AppMode::List;
    }

    /// Start search mode in view
    fn start_view_search(&mut self) {
        if let AppMode::View(ref mut state) = self.app_mode {
            state.search_mode = ViewSearchMode::Typing;
            state.search_query.clear();
            state.search_matches.clear();
            state.current_match = 0;
        }
    }

    /// Update search results based on current query
    fn update_search_results(&mut self) {
        if let AppMode::View(ref mut state) = self.app_mode {
            let query_lower = state.search_query.to_lowercase();
            if query_lower.is_empty() {
                state.search_matches.clear();
                return;
            }

            state.search_matches = state
                .rendered_lines
                .iter()
                .enumerate()
                .filter(|(_, line)| line_matches_query(line, &query_lower))
                .map(|(i, _)| i)
                .collect();

            // Jump to first match if any
            if !state.search_matches.is_empty() {
                state.current_match = 0;
                let match_line = state.search_matches[0];
                state.scroll_offset = match_line;
                Self::focus_message_at_line(state, match_line);
            }
        }
    }

    /// Go to next search match
    fn next_search_match(&mut self, viewport_height: usize) {
        if let AppMode::View(ref mut state) = self.app_mode {
            if state.search_matches.is_empty() {
                return;
            }
            state.current_match = (state.current_match + 1) % state.search_matches.len();
            let match_line = state.search_matches[state.current_match];
            // Scroll to show match in viewport
            if match_line < state.scroll_offset
                || match_line >= state.scroll_offset + viewport_height
            {
                state.scroll_offset = match_line;
            }
            Self::focus_message_at_line(state, match_line);
        }
    }

    /// Go to previous search match
    fn prev_search_match(&mut self, viewport_height: usize) {
        if let AppMode::View(ref mut state) = self.app_mode {
            if state.search_matches.is_empty() {
                return;
            }
            state.current_match = if state.current_match == 0 {
                state.search_matches.len() - 1
            } else {
                state.current_match - 1
            };
            let match_line = state.search_matches[state.current_match];
            if match_line < state.scroll_offset
                || match_line >= state.scroll_offset + viewport_height
            {
                state.scroll_offset = match_line;
            }
            Self::focus_message_at_line(state, match_line);
        }
    }

    /// Cycle tool display mode in view mode
    fn toggle_view_tools(&mut self, viewport_height: usize) {
        if let AppMode::View(ref mut state) = self.app_mode {
            state.tool_display = state.tool_display.next();
            self.tool_display = state.tool_display; // Persist at app level
            self.re_render_view(viewport_height);
        }
    }

    /// Toggle thinking visibility in view mode
    fn toggle_view_thinking(&mut self, viewport_height: usize) {
        if let AppMode::View(ref mut state) = self.app_mode {
            state.show_thinking = !state.show_thinking;
            self.show_thinking = state.show_thinking; // Persist at app level
            self.re_render_view(viewport_height);
        }
    }

    /// Toggle timing visibility in view mode (timestamps + durations)
    fn toggle_view_timing(&mut self, viewport_height: usize) {
        if let AppMode::View(ref mut state) = self.app_mode {
            state.show_timing = !state.show_timing;
            self.show_timing = state.show_timing; // Persist at app level
            self.re_render_view(viewport_height);
        }
    }

    /// Re-render the view with current toggle settings
    fn re_render_view(&mut self, viewport_height: usize) {
        use crate::tui::viewer::{
            RenderOptions, parse_conversation_file, render_parsed_conversation,
        };

        if let AppMode::View(ref mut state) = self.app_mode {
            let options = RenderOptions {
                tool_display: state.tool_display,
                show_thinking: state.show_thinking,
                show_timing: state.show_timing,
                content_width: state.content_width,
                expanded_tool_outputs: state.expanded_tool_outputs.clone(),
            };

            // Capture an anchor against the current layout so we can restore the
            // viewport against the same message after the total line count changes.
            let anchor = capture_anchor(
                &state.message_ranges,
                state.scroll_offset,
                state.focused_message,
                state.message_nav_active,
            );
            let old_scroll = state.scroll_offset;

            let entries = match state.parsed_entries.clone() {
                Some(entries) => entries,
                None => match parse_conversation_file(&state.conversation_path) {
                    Ok(entries) => {
                        let entries = Arc::new(entries);
                        state.parsed_entries = Some(entries.clone());
                        entries
                    }
                    Err(_) => return,
                },
            };
            let rendered = render_parsed_conversation(&entries, &options);
            state.total_lines = rendered.lines.len();
            state.rendered_lines = rendered.lines;
            state.message_ranges = rendered.messages;

            let max_scroll = state.total_lines.saturating_sub(viewport_height);

            // Resolve focused message by entry_index, falling back to the
            // previous surviving entry if the exact one disappeared. If no
            // anchor existed (ranges was previously empty) but ranges is now
            // non-empty, default to the first message so nav mode has a
            // valid focus target.
            let resolved_idx = anchor
                .and_then(|a| find_message_idx_or_prev(&state.message_ranges, a.entry_index))
                .or_else(|| (!state.message_ranges.is_empty()).then_some(0));
            state.focused_message = resolved_idx;

            state.scroll_offset = match (anchor, resolved_idx) {
                (Some(a), Some(idx)) => {
                    let new_msg = &state.message_ranges[idx];
                    // If the anchor vanished, cap relative_row at 0 so the
                    // fallback message sits at the top of the viewport rather
                    // than being pushed down (revealing already-read content).
                    let rel = if new_msg.entry_index == a.entry_index {
                        a.relative_row
                    } else {
                        a.relative_row.min(0)
                    };
                    let raw = new_msg.start_line as isize - rel;
                    raw.clamp(0, max_scroll as isize) as usize
                }
                _ => old_scroll.min(max_scroll),
            };

            // Recompute search matches for new content
            if state.search_mode == ViewSearchMode::Active && !state.search_query.is_empty() {
                let query_lower = state.search_query.to_lowercase();
                state.search_matches = state
                    .rendered_lines
                    .iter()
                    .enumerate()
                    .filter(|(_, line)| line_matches_query(line, &query_lower))
                    .map(|(i, _)| i)
                    .collect();

                // Clamp current_match to valid range
                if state.search_matches.is_empty() {
                    state.current_match = 0;
                } else {
                    state.current_match = state.current_match.min(state.search_matches.len() - 1);
                }
            }
        }
    }

    /// Jump to the next message (activates message nav mode)
    fn focus_next_message(&mut self, viewport_height: usize) {
        if let AppMode::View(ref mut state) = self.app_mode {
            if state.message_ranges.is_empty() {
                return;
            }
            // On first activation, sync focus to current scroll position
            if !state.message_nav_active {
                state.message_nav_active = true;
                Self::sync_focus_to_scroll(state, viewport_height);
            }
            let next = match state.focused_message {
                Some(i) if i + 1 < state.message_ranges.len() => i + 1,
                Some(i) => i, // already at last
                None => 0,
            };
            state.focused_message = Some(next);
            Self::ensure_message_visible(state, viewport_height);
        }
    }

    /// Jump to the previous message (activates message nav mode)
    fn focus_prev_message(&mut self, viewport_height: usize) {
        if let AppMode::View(ref mut state) = self.app_mode {
            if state.message_ranges.is_empty() {
                return;
            }
            // On first activation, sync focus to current scroll position
            if !state.message_nav_active {
                state.message_nav_active = true;
                Self::sync_focus_to_scroll(state, viewport_height);
            }
            let prev = match state.focused_message {
                Some(i) if i > 0 => i - 1,
                Some(i) => i, // already at first
                None => 0,
            };
            state.focused_message = Some(prev);
            Self::ensure_message_visible(state, viewport_height);
        }
    }

    /// Focus the message containing the given line index, activating nav mode
    fn focus_message_at_line(state: &mut ViewState, line_idx: usize) {
        let found = state
            .message_ranges
            .iter()
            .position(|m| line_idx >= m.start_line && line_idx < m.end_line);
        if let Some(idx) = found {
            state.message_nav_active = true;
            state.focused_message = Some(idx);
        }
    }

    /// Sync focus after a scroll operation (only when message nav is active)
    fn sync_focus_after_scroll(&mut self, viewport_height: usize) {
        if let AppMode::View(ref mut state) = self.app_mode
            && state.message_nav_active
        {
            Self::sync_focus_to_scroll(state, viewport_height);
        }
    }

    /// Scroll the view by a signed number of lines (positive = down, negative = up).
    /// Only affects the conversation viewer; no-op in other modes or while typing a search.
    pub fn scroll_view(&mut self, delta: isize, viewport_height: usize) {
        if let AppMode::View(ref mut state) = self.app_mode {
            if state.search_mode == ViewSearchMode::Typing {
                return;
            }
            let max_scroll = state.total_lines.saturating_sub(viewport_height);
            let new_offset = if delta >= 0 {
                state
                    .scroll_offset
                    .saturating_add(delta as usize)
                    .min(max_scroll)
            } else {
                state.scroll_offset.saturating_sub((-delta) as usize)
            };
            state.scroll_offset = new_offset;
            self.sync_focus_after_scroll(viewport_height);
        }
    }

    /// Route mouse wheel scrolling to the active UI mode.
    pub fn scroll_mouse(&mut self, delta: isize, viewport_height: usize) {
        if self.dialog_mode != DialogMode::None {
            return;
        }

        match self.app_mode {
            AppMode::List => self.scroll_list(delta.signum()),
            AppMode::View(_) => self.scroll_view(delta, viewport_height),
        }
    }

    fn view_line_at_row(&self, row: u16, frame_area: Rect) -> Option<usize> {
        let AppMode::View(state) = &self.app_mode else {
            return None;
        };
        if self.dialog_mode != DialogMode::None {
            return None;
        }
        let layout = ui::view_layout_rects(frame_area, self, state);
        if row < layout.content.y || row >= layout.content.y.saturating_add(layout.content.height) {
            return None;
        }
        Some(state.scroll_offset + (row - layout.content.y) as usize)
    }

    fn message_idx_at_line(ranges: &[MessageRange], line_idx: usize) -> Option<usize> {
        let idx = ranges.partition_point(|m| m.end_line <= line_idx);
        ranges
            .get(idx)
            .is_some_and(|m| line_idx >= m.start_line && line_idx < m.end_line)
            .then_some(idx)
    }

    fn view_tool_output_at_line(&self, line_idx: usize) -> Option<ToolOutputId> {
        let AppMode::View(state) = &self.app_mode else {
            return None;
        };
        state.rendered_lines.get(line_idx).and_then(|line| {
            if line.clickable {
                line.tool_output_id.clone()
            } else {
                None
            }
        })
    }

    pub fn handle_view_mouse_move(&mut self, row: u16, frame_area: Rect) -> bool {
        let next = self
            .view_line_at_row(row, frame_area)
            .and_then(|line_idx| self.view_tool_output_at_line(line_idx));
        let AppMode::View(state) = &mut self.app_mode else {
            return false;
        };
        if state.hovered_tool_output == next {
            return false;
        }
        state.hovered_tool_output = next;
        true
    }

    pub fn handle_view_click(
        &mut self,
        row: u16,
        frame_area: Rect,
        viewport_height: usize,
    ) -> bool {
        let Some(line_idx) = self.view_line_at_row(row, frame_area) else {
            return false;
        };
        let tool_output = self.view_tool_output_at_line(line_idx);
        let message_idx = if let AppMode::View(state) = &self.app_mode {
            Self::message_idx_at_line(&state.message_ranges, line_idx)
        } else {
            None
        };
        if tool_output.is_none() && message_idx.is_none() {
            return false;
        }

        let AppMode::View(state) = &mut self.app_mode else {
            return false;
        };
        let mut changed = false;
        if let Some(idx) = message_idx
            && (!state.message_nav_active || state.focused_message != Some(idx))
        {
            state.message_nav_active = true;
            state.focused_message = Some(idx);
            changed = true;
        }
        if let Some(id) = tool_output {
            if state.expanded_tool_outputs.contains(&id) {
                state.expanded_tool_outputs.remove(&id);
            } else {
                state.expanded_tool_outputs.insert(id.clone());
            }
            state.hovered_tool_output = Some(id);
            changed = true;
        }
        if changed {
            self.re_render_view(viewport_height);
        }
        changed
    }

    /// Handle a left-click in list mode: select the conversation under the cursor.
    /// Returns true if the click landed on a list item — the caller is expected to
    /// then transition into view mode (matching the Enter-key behavior).
    pub fn handle_list_click(&mut self, row: u16, frame_area: Rect) -> bool {
        if !matches!(self.app_mode, AppMode::List)
            || self.dialog_mode != DialogMode::None
            || self.is_loading()
        {
            return false;
        }

        // Mirror the layout in render_list_mode: outer 1px border, then split
        // [search bar (2), list (Min 1), bottom bar (1)] — or omit the bottom
        // bar when the inner area is < 4 lines tall.
        let inner_height = frame_area.height.saturating_sub(2);
        let list_y = frame_area.y.saturating_add(1).saturating_add(2);
        let list_height = if inner_height < 4 {
            inner_height.saturating_sub(2)
        } else {
            inner_height.saturating_sub(3)
        };

        if list_height == 0 || row < list_y || row >= list_y.saturating_add(list_height) {
            return false;
        }

        let lines_per_item = list_lines_per_item(self.list_search_mode, &self.query);
        let items_per_page = (list_height as usize) / lines_per_item;
        if items_per_page == 0 {
            return false;
        }

        let offset = match self.selected {
            Some(sel) => (sel / items_per_page) * items_per_page,
            None => 0,
        };
        let relative_row = (row - list_y) as usize;
        let relative_idx = relative_row / lines_per_item;
        let new_idx = offset + relative_idx;
        if new_idx < self.filtered.len() {
            self.selected = Some(new_idx);
            true
        } else {
            false
        }
    }

    /// Sync focused message to the current scroll position
    fn sync_focus_to_scroll(state: &mut ViewState, viewport_height: usize) {
        if state.message_ranges.is_empty() {
            return;
        }
        let viewport_start = state.scroll_offset;
        let viewport_end = viewport_start + viewport_height;
        let found = state
            .message_ranges
            .iter()
            .position(|m| m.end_line > viewport_start && m.start_line < viewport_end);
        if let Some(idx) = found {
            state.focused_message = Some(idx);
        }
    }

    /// Scroll viewport to make the focused message visible
    fn ensure_message_visible(state: &mut ViewState, viewport_height: usize) {
        if let Some(idx) = state.focused_message
            && let Some(msg) = state.message_ranges.get(idx)
        {
            let max_scroll = state.total_lines.saturating_sub(viewport_height);
            if msg.start_line < state.scroll_offset
                || msg.start_line >= state.scroll_offset + viewport_height
            {
                state.scroll_offset = msg.start_line.min(max_scroll);
            }
        }
    }

    /// Copy the currently focused message to clipboard
    fn copy_focused_message(&mut self, viewport_height: usize) {
        // Activate nav mode and sync focus if needed
        if let AppMode::View(ref mut state) = self.app_mode
            && !state.message_nav_active
        {
            state.message_nav_active = true;
            Self::sync_focus_to_scroll(state, viewport_height);
        }

        let (path, entry_index) = if let AppMode::View(ref state) = self.app_mode {
            if let Some(idx) = state.focused_message {
                if let Some(msg) = state.message_ranges.get(idx) {
                    (state.conversation_path.clone(), msg.entry_index)
                } else {
                    return;
                }
            } else {
                return;
            }
        } else {
            return;
        };

        let options = if let AppMode::View(ref state) = self.app_mode {
            crate::tui::export::ExportOptions {
                show_tools: state.tool_display.is_visible(),
                show_thinking: state.show_thinking,
            }
        } else {
            return;
        };

        match crate::tui::export::extract_message_text(&path, entry_index, options) {
            Ok(text) if text.is_empty() => {
                self.status_message = Some((
                    "No text content in this message".to_string(),
                    std::time::Instant::now(),
                ));
            }
            Ok(text) => match crate::tui::export::copy_to_system_clipboard(&text) {
                Ok(()) => {
                    self.status_message = Some((
                        "Message copied to clipboard".to_string(),
                        std::time::Instant::now(),
                    ));
                }
                Err(e) => {
                    self.status_message = Some((e, std::time::Instant::now()));
                }
            },
            Err(e) => {
                self.status_message = Some((e, std::time::Instant::now()));
            }
        }
    }

    /// Check if view needs re-render due to width change
    pub fn check_view_resize(&mut self, new_content_width: usize, viewport_height: usize) {
        if let AppMode::View(ref mut state) = self.app_mode
            && state.content_width != new_content_width
        {
            state.content_width = new_content_width;
            self.re_render_view(viewport_height);
        }
    }
}

/// Stable reference point for preserving scroll position across re-renders.
/// `entry_index` survives re-renders (it's the JSONL line index), and
/// `relative_row` is the message's screen row (`start_line - scroll_offset`)
/// before re-render. `isize` so it can go negative when the anchor started
/// above the viewport.
#[derive(Clone, Copy, Debug)]
struct ScrollAnchor {
    entry_index: usize,
    relative_row: isize,
}

/// Pick an anchor message for the current view state.
/// In nav mode the anchor is the focused message; otherwise it is the first
/// message at or below the viewport top (falling back to the last message if
/// the user has scrolled past the end).
fn capture_anchor(
    ranges: &[MessageRange],
    scroll_offset: usize,
    focused: Option<usize>,
    nav_active: bool,
) -> Option<ScrollAnchor> {
    if ranges.is_empty() {
        return None;
    }

    let msg = if nav_active {
        focused.and_then(|i| ranges.get(i))
    } else {
        None
    }
    .unwrap_or_else(|| {
        let i = ranges.partition_point(|m| m.start_line < scroll_offset);
        ranges.get(i).unwrap_or_else(|| ranges.last().unwrap())
    });

    Some(ScrollAnchor {
        entry_index: msg.entry_index,
        relative_row: msg.start_line as isize - scroll_offset as isize,
    })
}

/// Find the index of the message with this `entry_index`, or the closest
/// preceding surviving entry. Returns `Some(0)` when no earlier entry exists
/// but `ranges` is non-empty.
fn find_message_idx_or_prev(ranges: &[MessageRange], entry_index: usize) -> Option<usize> {
    if ranges.is_empty() {
        return None;
    }
    match ranges.binary_search_by_key(&entry_index, |m| m.entry_index) {
        Ok(idx) => Some(idx),
        Err(0) => Some(0),
        Err(idx) => Some(idx - 1),
    }
}

#[cfg(test)]
mod tests;

/// RAII guard to ensure terminal is restored on exit
struct TerminalGuard {
    terminal: Terminal<CrosstermBackend<Stdout>>,
}

/// Check if a rendered line matches the search query by concatenating all span texts.
/// This allows multi-word queries to match across span boundaries.
pub fn line_matches_query(line: &RenderedLine, query_lower: &str) -> bool {
    let full_text: String = line.spans.iter().map(|(text, _)| text.as_str()).collect();
    full_text.to_lowercase().contains(query_lower)
}

#[cfg(test)]
mod interaction_tests;

impl TerminalGuard {
    fn new() -> Result<Self> {
        terminal::enable_raw_mode().map_err(|e| AppError::Io(io::Error::other(e)))?;

        let mut stdout = io::stdout();
        if let Err(e) = crossterm::execute!(stdout, EnterAlternateScreen, EnableMouseCapture) {
            let _ = terminal::disable_raw_mode();
            return Err(AppError::Io(io::Error::other(e)));
        }

        let backend = CrosstermBackend::new(stdout);
        let terminal = match Terminal::new(backend) {
            Ok(t) => t,
            Err(e) => {
                let _ = terminal::disable_raw_mode();
                let _ =
                    crossterm::execute!(io::stdout(), DisableMouseCapture, LeaveAlternateScreen);
                return Err(AppError::Io(io::Error::other(e)));
            }
        };

        Ok(Self { terminal })
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = terminal::disable_raw_mode();
        let _ = crossterm::execute!(
            self.terminal.backend_mut(),
            DisableMouseCapture,
            LeaveAlternateScreen
        );
    }
}

/// Name column width for ledger-style display
const NAME_WIDTH: usize = 9;

/// Maximum events to drain in a single batch to avoid starving redraws
const MAX_EVENT_BATCH: usize = 256;

/// Read all immediately available events after an initial blocking wait.
///
/// When pasting text, crossterm delivers each character as a separate KeyEvent.
/// Without batching, each character triggers a full redraw before reading the next,
/// making paste visibly slow. This function drains all ready events so the caller
/// can process them all before a single redraw.
fn drain_events(wait: Duration) -> Result<Vec<Event>> {
    if !event::poll(wait).map_err(|e| AppError::Io(io::Error::other(e)))? {
        return Ok(Vec::new());
    }

    let mut events = vec![event::read().map_err(|e| AppError::Io(io::Error::other(e)))?];

    while events.len() < MAX_EVENT_BATCH
        && event::poll(Duration::ZERO).map_err(|e| AppError::Io(io::Error::other(e)))?
    {
        events.push(event::read().map_err(|e| AppError::Io(io::Error::other(e)))?);
    }

    Ok(events)
}

/// Run the TUI with background loading
/// Returns the action and the final list of conversations
#[allow(clippy::too_many_arguments)]
pub fn run_with_loader(
    rx: Receiver<LoaderMessage>,
    tool_display: ToolDisplayMode,
    show_thinking: bool,
    keys: KeyBindings,
    workspace_filter: bool,
    current_project_dir_name: Option<String>,
    exclude_projects: Vec<String>,
    search_options: TuiSearchOptions,
) -> Result<(Action, Vec<Conversation>)> {
    // Set up panic hook to restore terminal
    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        let _ = terminal::disable_raw_mode();
        let _ = crossterm::execute!(io::stdout(), DisableMouseCapture, LeaveAlternateScreen);
        original_hook(panic_info);
    }));

    let mut guard = TerminalGuard::new()?;
    let mut app = App::new_loading_with_options(
        tool_display,
        show_thinking,
        keys,
        workspace_filter,
        current_project_dir_name,
        exclude_projects,
        search_options,
    );

    loop {
        // Process all pending loader messages (non-blocking)
        loop {
            match rx.try_recv() {
                Ok(LoaderMessage::Fatal(err)) => {
                    // Fatal error - restore terminal and return error
                    drop(guard);
                    return Err(err);
                }
                Ok(LoaderMessage::ProjectError) => {
                    // Logged by loader, continue
                }
                Ok(LoaderMessage::Batch(convs)) => {
                    app.append_conversations(convs);
                }
                Ok(LoaderMessage::Done) => {
                    app.finish_loading();
                    // Check for empty conversations
                    if app.conversations().is_empty() {
                        drop(guard);
                        return Err(AppError::NoHistoryFound("selected scope".to_string()));
                    }
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    // Loader finished unexpectedly
                    if app.is_loading() {
                        app.finish_loading();
                        if app.conversations().is_empty() {
                            drop(guard);
                            return Err(AppError::NoHistoryFound("selected scope".to_string()));
                        }
                    }
                    break;
                }
            }
        }

        let frame_area = guard.terminal.get_frame().area();
        let viewport_height = frame_area.height.saturating_sub(3) as usize;
        let content_width = (frame_area.width as usize)
            .saturating_sub(NAME_WIDTH + 3 + crate::tui::viewer::GUTTER_WIDTH);

        // Check for resize in view mode
        app.check_view_resize(content_width, viewport_height);
        let viewport_height = match app.app_mode() {
            AppMode::View(state) => {
                ui::view_layout_rects(frame_area, &app, state)
                    .content
                    .height as usize
            }
            AppMode::List => viewport_height,
        };

        // Pick up any completed search results from the background worker
        app.receive_search_results();

        // Render current state
        guard.terminal.draw(|frame| ui::render(frame, &app))?;

        // Use short poll timeout while loading or search is in-flight,
        // otherwise block until input arrives (or until status message expires)
        let poll_timeout = if app.is_loading() {
            Duration::from_millis(50)
        } else if app.has_search_work_in_flight() {
            // Poll frequently so search results appear quickly (within ~8ms)
            Duration::from_millis(8)
        } else if let Some(remaining) = app.status_message_remaining() {
            remaining
        } else {
            Duration::from_secs(3600)
        };

        // Drain all currently queued events and process them, then redraw.
        // drain_events coalesces events that arrive during rendering (e.g. paste),
        // while always returning to the outer loop for a redraw after each batch.
        let events = drain_events(poll_timeout)?;
        for ev in events {
            let key = match ev {
                Event::Key(k) if k.kind == KeyEventKind::Press => k,
                Event::Mouse(m) => {
                    match m.kind {
                        MouseEventKind::ScrollDown => {
                            app.scroll_mouse(3, viewport_height);
                        }
                        MouseEventKind::ScrollUp => {
                            app.scroll_mouse(-3, viewport_height);
                        }
                        MouseEventKind::Down(MouseButton::Left) => {
                            if app.handle_view_click(m.row, frame_area, viewport_height) {
                                break;
                            }
                            if app.handle_list_click(m.row, frame_area) {
                                app.enter_view_mode(content_width);
                                break; // mode transition: redraw before processing more events
                            }
                        }
                        MouseEventKind::Moved => {
                            app.handle_view_mouse_move(m.row, frame_area);
                        }
                        _ => {}
                    }
                    continue;
                }
                _ => continue,
            };

            // Check for Enter in list mode - enter view mode (but not during dialogs)
            if matches!(app.app_mode(), AppMode::List)
                && *app.dialog_mode() == DialogMode::None
                && key.code == KeyCode::Enter
                && !app.is_loading()
                && app.selected().is_some()
            {
                app.enter_view_mode(content_width);
                break; // mode transition: redraw before processing more events
            }

            if let Some(action) = app.handle_key(key.code, key.modifiers, viewport_height) {
                match action {
                    Action::Delete(ref path) => {
                        // Extract UUID from filename and delete session
                        // (removes .jsonl + session dir with tool-results/subagents)
                        let uuid = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
                        match crate::history::delete_session_by_uuid(uuid) {
                            Ok(_) => {
                                // Only remove from list if deletion succeeded
                                app.remove_selected_from_list();
                                // If in view mode, return to list
                                app.exit_view_mode();
                            }
                            Err(e) => {
                                let _ = debug_log::log_debug(&format!(
                                    "Failed to delete session {}: {}",
                                    uuid, e
                                ));
                                // Keep item in list since file still exists
                            }
                        }
                        // Continue the loop (don't exit TUI)
                    }
                    _ => return Ok((action, app.into_conversations())),
                }
            }
        }
    }
}

/// Run the TUI for a single file (direct input mode)
pub fn run_single_file(
    path: PathBuf,
    tool_display: ToolDisplayMode,
    show_thinking: bool,
    keys: KeyBindings,
) -> Result<()> {
    // Set up panic hook to restore terminal
    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        let _ = terminal::disable_raw_mode();
        let _ = crossterm::execute!(io::stdout(), DisableMouseCapture, LeaveAlternateScreen);
        original_hook(panic_info);
    }));

    let mut guard = TerminalGuard::new()?;
    let mut app = App::new_single_file(path, tool_display, show_thinking, keys);

    loop {
        let frame_area = guard.terminal.get_frame().area();
        let viewport_height = frame_area.height.saturating_sub(3) as usize;
        let content_width = (frame_area.width as usize)
            .saturating_sub(NAME_WIDTH + 3 + crate::tui::viewer::GUTTER_WIDTH);

        // Check for resize in view mode (this triggers initial render too)
        app.check_view_resize(content_width, viewport_height);
        let viewport_height = match app.app_mode() {
            AppMode::View(state) => {
                ui::view_layout_rects(frame_area, &app, state)
                    .content
                    .height as usize
            }
            AppMode::List => viewport_height,
        };

        guard.terminal.draw(|frame| ui::render(frame, &app))?;

        let events = drain_events(Duration::from_secs(3600))?;
        for ev in events {
            let key = match ev {
                Event::Key(k) if k.kind == KeyEventKind::Press => k,
                Event::Mouse(m) => {
                    match m.kind {
                        MouseEventKind::ScrollDown => app.scroll_mouse(3, viewport_height),
                        MouseEventKind::ScrollUp => app.scroll_mouse(-3, viewport_height),
                        MouseEventKind::Down(MouseButton::Left) => {
                            app.handle_view_click(m.row, frame_area, viewport_height);
                        }
                        MouseEventKind::Moved => {
                            app.handle_view_mouse_move(m.row, frame_area);
                        }
                        _ => {}
                    }
                    continue;
                }
                _ => continue,
            };
            if let Some(Action::Quit) = app.handle_key(key.code, key.modifiers, viewport_height) {
                return Ok(());
            }
        }
    }
}
