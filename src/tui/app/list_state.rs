use super::{App, SearchCommand};
use crate::history::{Conversation, format_short_name_from_path};
use crate::search;
use crate::search::query::ParsedQuery;
use chrono::Local;
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

impl App {
    pub(super) fn filter_indices<I>(&self, indices: I) -> Vec<usize>
    where
        I: IntoIterator<Item = usize>,
    {
        filter_conversation_indices(
            indices,
            &self.conversations,
            &self.excluded_projects,
            self.workspace_filter,
            self.current_project_dir_name.as_deref(),
        )
    }

    pub(super) fn apply_filtered(&mut self, filtered: Vec<usize>) {
        self.filtered = filtered;
        self.selected = if self.filtered.is_empty() {
            None
        } else {
            Some(0)
        };
    }

    pub(super) fn apply_uuid_filter(&mut self, query: &str) -> bool {
        if let Some(idx) = self.find_or_load_uuid(query) {
            self.filtered = vec![idx];
            self.selected = Some(0);
            true
        } else {
            false
        }
    }

    pub(super) fn apply_lexical_filter(&mut self) {
        let now = Local::now();
        let filtered = search::search(&self.conversations, &self.searchable, &self.query, now);
        let filtered = self.filter_indices(filtered);
        self.apply_filtered(filtered);
    }

    pub(super) fn update_filter(&mut self) {
        let query = self.query.trim().to_string();

        if ParsedQuery::parse(&query).is_effectively_empty() {
            self.semantic_search.error = None;
            self.apply_lexical_filter();
            return;
        }

        if search::is_uuid(&query) {
            self.semantic_search.error = None;
            self.apply_uuid_filter(&query);
            return;
        }

        if self.list_search_mode == super::ListSearchMode::Semantic {
            return;
        }

        self.semantic_search.error = None;
        self.semantic_search.results.clear();
        self.apply_lexical_filter();
    }

    pub(super) fn find_or_load_uuid(&mut self, uuid: &str) -> Option<usize> {
        let uuid_jsonl = format!("{}.jsonl", uuid);
        for (idx, conv) in self.conversations.iter().enumerate() {
            if conv
                .path
                .file_name()
                .is_some_and(|f| f.to_string_lossy() == uuid_jsonl)
            {
                return Some(idx);
            }
        }

        let path = crate::history::find_jsonl_by_uuid(uuid).ok()??;
        let modified = path.metadata().ok().and_then(|m| m.modified().ok());
        let mut conv = crate::history::process_conversation_file(path, modified, None).ok()??;

        let fallback_path = conv
            .path
            .parent()
            .and_then(|p| p.file_name())
            .map(|n| crate::history::path::decode_project_dir_name_to_path(&n.to_string_lossy()))
            .unwrap_or_default();
        let project_path = conv.cwd.clone().unwrap_or(fallback_path);
        conv.project_name = Some(format_short_name_from_path(&project_path));
        conv.project_path = Some(project_path);

        let idx = self.conversations.len();
        self.conversations.push(conv);

        self.searchable = search::precompute_search_text(&self.conversations);
        self.conversations_snapshot = Arc::new(self.conversations.clone());
        self.rebuild_semantic_conversations_snapshot();

        let _ = self.search_tx.send(SearchCommand::UpdateData {
            conversations: self.conversations_snapshot.clone(),
            searchable: Arc::new(self.searchable.clone()),
        });

        Some(idx)
    }

    pub(super) fn select_prev(&mut self) {
        if let Some(selected) = self.selected
            && selected > 0
        {
            self.selected = Some(selected - 1);
        }
    }

    pub(super) fn select_next(&mut self) {
        if let Some(selected) = self.selected
            && selected + 1 < self.filtered.len()
        {
            self.selected = Some(selected + 1);
        }
    }

    pub(super) fn select_first(&mut self) {
        if !self.filtered.is_empty() {
            self.selected = Some(0);
        }
    }

    pub(super) fn select_last(&mut self) {
        if !self.filtered.is_empty() {
            self.selected = Some(self.filtered.len() - 1);
        }
    }

    pub(super) fn select_page_up(&mut self) {
        if let Some(selected) = self.selected {
            self.selected = Some(selected.saturating_sub(10));
        }
    }

    pub(super) fn select_page_down(&mut self) {
        if let Some(selected) = self.selected {
            let new_selected = (selected + 10).min(self.filtered.len().saturating_sub(1));
            self.selected = Some(new_selected);
        }
    }

    pub(super) fn select_half_page_down(&mut self, viewport_height: usize) {
        if let Some(selected) = self.selected {
            let half_page = viewport_height / 2;
            let new_selected = (selected + half_page).min(self.filtered.len().saturating_sub(1));
            self.selected = Some(new_selected);
        }
    }

    pub(super) fn scroll_list(&mut self, delta: isize) {
        let Some(selected) = self.selected else {
            return;
        };

        let max = self.filtered.len().saturating_sub(1);
        let new_selected = if delta >= 0 {
            selected.saturating_add(delta as usize).min(max)
        } else {
            selected.saturating_sub((-delta) as usize)
        };
        self.selected = Some(new_selected);
    }

    pub(super) fn get_selected_path(&self) -> Option<PathBuf> {
        self.selected
            .and_then(|sel| self.filtered.get(sel))
            .map(|&idx| self.conversations[idx].path.clone())
    }

    pub(super) fn get_selected_conversation_index(&self) -> Option<usize> {
        self.selected
            .and_then(|sel| self.filtered.get(sel))
            .copied()
    }

    pub(super) fn refresh_search_data(&mut self) {
        self.conversations_snapshot = Arc::new(self.conversations.clone());
        self.rebuild_semantic_conversations_snapshot();
        self.searchable = search::precompute_search_text(&self.conversations);
        let _ = self.search_tx.send(SearchCommand::UpdateData {
            conversations: self.conversations_snapshot.clone(),
            searchable: Arc::new(self.searchable.clone()),
        });
        self.invalidate_search_generation();
    }

    pub(super) fn remove_selected_from_list(&mut self) {
        let Some(selected) = self.selected else {
            return;
        };
        let Some(&conv_idx) = self.filtered.get(selected) else {
            return;
        };

        self.conversations.remove(conv_idx);

        self.searchable.retain_mut(|s| {
            if s.index == conv_idx {
                false
            } else {
                if s.index > conv_idx {
                    s.index -= 1;
                }
                true
            }
        });

        self.filtered.retain(|&idx| idx != conv_idx);
        for idx in &mut self.filtered {
            if *idx > conv_idx {
                *idx -= 1;
            }
        }

        if self.filtered.is_empty() {
            self.selected = None;
        } else if selected >= self.filtered.len() {
            self.selected = Some(self.filtered.len() - 1);
        }

        self.refresh_search_data();
    }
}

pub(super) fn filter_conversation_indices<I>(
    indices: I,
    conversations: &[Conversation],
    excluded_projects: &HashSet<String>,
    workspace_filter: bool,
    current_project_dir_name: Option<&str>,
) -> Vec<usize>
where
    I: IntoIterator<Item = usize>,
{
    indices
        .into_iter()
        .filter(|&idx| {
            conversations[idx]
                .project_name
                .as_deref()
                .is_none_or(|name| !project_is_excluded(name, excluded_projects))
        })
        .filter(|&idx| {
            let Some(project_dir_name) = current_project_dir_name.filter(|_| workspace_filter)
            else {
                return true;
            };
            conversations[idx]
                .path
                .parent()
                .and_then(|p| p.file_name())
                .is_some_and(|name| {
                    crate::history::path::is_same_project(&name.to_string_lossy(), project_dir_name)
                })
        })
        .collect()
}

fn project_is_excluded(project_name: &str, excluded_projects: &HashSet<String>) -> bool {
    excluded_projects.contains(project_name)
        || project_name
            .split_once('/')
            .is_some_and(|(parent, _)| excluded_projects.contains(parent))
}
