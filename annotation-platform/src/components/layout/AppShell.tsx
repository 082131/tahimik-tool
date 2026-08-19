import { Outlet } from 'react-router-dom';
import TopNav from './TopNav';
import RulebookDrawer from '../shared/RulebookDrawer';

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

/**
 * Application shell: TopNav + main content area + global overlays.
 *
 * TabSwitchBanner is rendered here but controlled by individual annotation
 * pages via context or props drilled through Outlet context.
 */
export default function AppShell() {
  return (
    <div className="flex min-h-screen flex-col bg-[var(--color-surface-0)]">
      {/* Sticky top navigation */}
      <TopNav />

      {/* Main content area */}
      <main className="mx-auto w-full max-w-5xl flex-1 px-4 py-6 md:px-6 md:py-8">
        <Outlet />
      </main>

      {/* Global overlays */}
      <RulebookDrawer />
      {/*
        TabSwitchBanner is included here as a layout-level component.
        Individual pages pass count/visible via the annotation area context.
        A zero-count, hidden banner is the default state.
      */}
    </div>
  );
}
