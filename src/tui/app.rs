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
use crate::tui::viewer::ToolDisplayMode;
#[cfg(test)]
use crate::tui::viewer::ToolOutputId;
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

mod dialog_state;
mod input_controller;
mod list_state;
mod search_state;
mod types;
mod view_state;
#[allow(unused_imports)]
use search_state::{SearchCommand, SearchResponse, SemanticSearchState, spawn_search_worker};
#[allow(unused_imports)]
pub use types::{
    Action, AppMode, DialogMode, LIST_LINES_PER_ITEM, ListSearchMode, LoadingState,
    SemanticProgress, SemanticResultMetadata, TuiSearchOptions, ViewSearchMode, ViewState,
    list_lines_per_item,
};

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
}

#[cfg(test)]
mod tests;

/// RAII guard to ensure terminal is restored on exit
struct TerminalGuard {
    terminal: Terminal<CrosstermBackend<Stdout>>,
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
