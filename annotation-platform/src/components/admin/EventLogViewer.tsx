import { useState, useEffect, useCallback } from 'react';
import { supabase } from '../../lib/supabase';
import type { User, EventType } from '../../lib/types';

interface EventRow {
  id: string;
  user_id: string;
  sentence_id: string | null;
  task: string | null;
  type: string;
  value: Record<string, unknown> | null;
  ts: string;
}

const EVENT_TYPES: EventType[] = [
  'keystroke_summary', 'paste_attempt', 'copy_attempt', 'cut_attempt',
  'drag_attempt', 'tab_blur', 'focus_return', 'time_to_first_keystroke',
  'edit_count', 'submit',
];

const PAGE_SIZE = 25;

export default function EventLogViewer() {
  const [events, setEvents] = useState<EventRow[]>([]);
  const [annotators, setAnnotators] = useState<User[]>([]);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(0);
  const [totalCount, setTotalCount] = useState(0);

  // Filters
  const [filterAnnotator, setFilterAnnotator] = useState('');
  const [filterType, setFilterType] = useState('');
  const [filterSentence, setFilterSentence] = useState('');
  const [filterFrom, setFilterFrom] = useState('');
  const [filterTo, setFilterTo] = useState('');

  // Summary stats
  const [summary, setSummary] = useState<Record<string, { paste: number; copy: number; tab: number }>>({});

  useEffect(() => {
    supabase.from('users').select('id, name, role, status')
      .eq('role', 'annotator')
      .then(({ data }) => setAnnotators((data as User[]) || []));
  }, []);

  const fetchEvents = useCallback(async () => {
    setLoading(true);
    let query = supabase.from('events')
      .select('*', { count: 'exact' })
      .order('ts', { ascending: false })
      .range(page * PAGE_SIZE, (page + 1) * PAGE_SIZE - 1);

    if (filterAnnotator) query = query.eq('user_id', filterAnnotator);
    if (filterType) query = query.eq('type', filterType);
    if (filterSentence) query = query.eq('sentence_id', filterSentence);
    if (filterFrom) query = query.gte('ts', filterFrom);
    if (filterTo) query = query.lte('ts', filterTo + 'T23:59:59Z');

    const { data, count } = await query;
    setEvents((data as EventRow[]) || []);
    setTotalCount(count ?? 0);
    setLoading(false);
  }, [page, filterAnnotator, filterType, filterSentence, filterFrom, filterTo]);

  useEffect(() => { fetchEvents(); }, [fetchEvents]);

  // Fetch summary stats
  useEffect(() => {
    async function fetchSummary() {
      const stats: Record<string, { paste: number; copy: number; tab: number }> = {};
      for (const a of annotators) {
        const [pasteRes, copyRes, tabRes] = await Promise.all([
          supabase.from('events').select('id', { count: 'exact', head: true })
            .eq('user_id', a.id).eq('type', 'paste_attempt'),
          supabase.from('events').select('id', { count: 'exact', head: true })
            .eq('user_id', a.id).eq('type', 'copy_attempt'),
          supabase.from('events').select('id', { count: 'exact', head: true })
            .eq('user_id', a.id).eq('type', 'tab_blur'),
        ]);
        stats[a.id] = { paste: pasteRes.count ?? 0, copy: copyRes.count ?? 0, tab: tabRes.count ?? 0 };
      }
      setSummary(stats);
    }
    if (annotators.length > 0) fetchSummary();
  }, [annotators]);

  const getAnnotatorName = (id: string) => annotators.find((a) => a.id === id)?.name || id.slice(0, 8);
  const totalPages = Math.ceil(totalCount / PAGE_SIZE);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold text-[var(--color-text-primary)]">Event Logs</h2>
        <p className="text-xs text-[var(--color-text-muted)] italic">
          Events are for review only. Nothing auto-punishes an annotator.
        </p>
      </div>

      {/* Summary stats */}
      {annotators.length > 0 && Object.keys(summary).length > 0 && (
        <div className="glass-panel p-4">
          <h3 className="mb-3 text-xs font-medium uppercase tracking-widest text-[var(--color-text-muted)]">
            Per-Annotator Summary
          </h3>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
            {annotators.map((a) => (
              <div key={a.id} className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface-2)] p-3">
                <p className="mb-2 text-sm font-medium text-[var(--color-text-primary)]">{a.name}</p>
                <div className="flex gap-3 text-xs text-[var(--color-text-secondary)]">
                  <span>Paste: <strong className="text-[var(--color-amber)]">{summary[a.id]?.paste ?? 0}</strong></span>
                  <span>Copy: <strong className="text-[var(--color-amber)]">{summary[a.id]?.copy ?? 0}</strong></span>
                  <span>Tab: <strong className="text-[var(--color-amber)]">{summary[a.id]?.tab ?? 0}</strong></span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Filters */}
      <div className="glass-panel p-4">
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
          <select
            id="filter-annotator"
            value={filterAnnotator}
            onChange={(e) => { setFilterAnnotator(e.target.value); setPage(0); }}
            className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface-2)] px-3 py-2 text-xs text-[var(--color-text-primary)] focus:outline-none"
          >
            <option value="">All Annotators</option>
            {annotators.map((a) => <option key={a.id} value={a.id}>{a.name}</option>)}
          </select>

          <select
            id="filter-event-type"
            value={filterType}
            onChange={(e) => { setFilterType(e.target.value); setPage(0); }}
            className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface-2)] px-3 py-2 text-xs text-[var(--color-text-primary)] focus:outline-none"
          >
            <option value="">All Types</option>
            {EVENT_TYPES.map((t) => <option key={t} value={t}>{t}</option>)}
          </select>

          <input
            id="filter-sentence"
            type="text"
            value={filterSentence}
            onChange={(e) => { setFilterSentence(e.target.value); setPage(0); }}
            placeholder="Sentence ID"
            className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface-2)] px-3 py-2 text-xs text-[var(--color-text-primary)] placeholder-[var(--color-text-muted)] focus:outline-none"
          />

          <input
            id="filter-from"
            type="date"
            value={filterFrom}
            onChange={(e) => { setFilterFrom(e.target.value); setPage(0); }}
            className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface-2)] px-3 py-2 text-xs text-[var(--color-text-primary)] focus:outline-none"
          />

          <input
            id="filter-to"
            type="date"
            value={filterTo}
            onChange={(e) => { setFilterTo(e.target.value); setPage(0); }}
            className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface-2)] px-3 py-2 text-xs text-[var(--color-text-primary)] focus:outline-none"
          />
        </div>
      </div>

      {/* Events table */}
      <div className="glass-panel overflow-x-auto">
        {loading ? (
          <div className="py-8 text-center text-[var(--color-text-muted)]">Loading events...</div>
        ) : events.length === 0 ? (
          <div className="py-8 text-center text-[var(--color-text-muted)]">No events found</div>
        ) : (
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-[var(--color-border)] text-[var(--color-text-muted)]">
                <th className="px-3 py-2.5 text-left font-medium">Timestamp</th>
                <th className="px-3 py-2.5 text-left font-medium">Annotator</th>
                <th className="px-3 py-2.5 text-left font-medium">Sentence ID</th>
                <th className="px-3 py-2.5 text-left font-medium">Task</th>
                <th className="px-3 py-2.5 text-left font-medium">Event Type</th>
                <th className="px-3 py-2.5 text-left font-medium">Value</th>
              </tr>
            </thead>
            <tbody>
              {events.map((ev) => (
                <tr key={ev.id} className="border-b border-[var(--color-border)]/50 text-[var(--color-text-secondary)]">
                  <td className="px-3 py-2 font-mono text-[var(--color-text-muted)]">
                    {new Date(ev.ts).toLocaleString()}
                  </td>
                  <td className="px-3 py-2">{getAnnotatorName(ev.user_id)}</td>
                  <td className="px-3 py-2 font-mono">{ev.sentence_id || '—'}</td>
                  <td className="px-3 py-2">{ev.task || '—'}</td>
                  <td className="px-3 py-2">
                    <span className={`rounded-full px-2 py-0.5 ${
                      ev.type.includes('attempt') ? 'bg-[var(--color-amber-muted)] text-[var(--color-amber)]'
                      : ev.type === 'tab_blur' ? 'bg-[var(--color-rose-muted)] text-[var(--color-rose)]'
                      : 'bg-[var(--color-surface-3)] text-[var(--color-text-secondary)]'
                    }`}>
                      {ev.type}
                    </span>
                  </td>
                  <td className="max-w-xs truncate px-3 py-2 font-mono text-[var(--color-text-muted)]">
                    {ev.value ? JSON.stringify(ev.value) : '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between text-xs text-[var(--color-text-muted)]">
          <span>Page {page + 1} of {totalPages} ({totalCount} events)</span>
          <div className="flex gap-2">
            <button
              onClick={() => setPage((p) => Math.max(0, p - 1))}
              disabled={page === 0}
              className="rounded-lg border border-[var(--color-border)] px-3 py-1.5 transition-colors hover:bg-[var(--color-surface-3)] disabled:opacity-50"
            >
              ← Prev
            </button>
            <button
              onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
              disabled={page >= totalPages - 1}
              className="rounded-lg border border-[var(--color-border)] px-3 py-1.5 transition-colors hover:bg-[var(--color-surface-3)] disabled:opacity-50"
            >
              Next →
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
