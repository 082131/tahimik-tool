import { useState } from 'react';
import ImportPanel from '../components/admin/ImportPanel';
import AssignmentPanel from '../components/admin/AssignmentPanel';
import ExportPanel from '../components/admin/ExportPanel';
import EventLogViewer from '../components/admin/EventLogViewer';

type Tab = 'import' | 'assignments' | 'export' | 'logs';

const TABS: { key: Tab; label: string }[] = [
  { key: 'import', label: 'Import' },
  { key: 'assignments', label: 'Assignments' },
  { key: 'export', label: 'Export' },
  { key: 'logs', label: 'Event Logs' },
];

export default function AdminPage() {
  const [activeTab, setActiveTab] = useState<Tab>('import');

  return (
    <div className="animate-fade-in">
      <h1 className="mb-6 text-xl font-semibold text-[var(--color-text-primary)]">
        Admin Dashboard
      </h1>

      {/* Tab navigation */}
      <div className="mb-8 flex gap-1 border-b border-[var(--color-border)]">
        {TABS.map((tab) => (
          <button
            key={tab.key}
            id={`admin-tab-${tab.key}`}
            onClick={() => setActiveTab(tab.key)}
            className={`
              relative px-4 py-2.5 text-sm font-medium transition-colors
              ${activeTab === tab.key
                ? 'text-[var(--color-accent)]'
                : 'text-[var(--color-text-muted)] hover:text-[var(--color-text-secondary)]'
              }
            `}
          >
            {tab.label}
            {activeTab === tab.key && (
              <span className="absolute bottom-0 left-0 right-0 h-0.5 rounded-full bg-[var(--color-accent)]" />
            )}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div>
        {activeTab === 'import' && <ImportPanel />}
        {activeTab === 'assignments' && <AssignmentPanel />}
        {activeTab === 'export' && <ExportPanel />}
        {activeTab === 'logs' && <EventLogViewer />}
      </div>
    </div>
  );
}
