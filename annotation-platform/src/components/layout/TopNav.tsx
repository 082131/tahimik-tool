import { NavLink } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';
import { useCallback } from 'react';

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

/** Dispatch the custom event the RulebookDrawer listens for. */
function toggleRulebook() {
  window.dispatchEvent(new CustomEvent('toggle-rulebook'));
}

/** NavLink class builder — active link gets teal accent underline. */
function navLinkClass({ isActive }: { isActive: boolean }): string {
  const base =
    'relative px-3 py-2 text-sm font-medium transition-colors duration-200 ' +
    'hover:text-[var(--color-accent-hover)] focus-visible:outline-offset-4';
  const active =
    'text-[var(--color-accent)] after:absolute after:inset-x-0 after:bottom-0 ' +
    'after:h-[2px] after:bg-[var(--color-accent)] after:rounded-full';
  const inactive = 'text-[var(--color-text-secondary)]';
  return `${base} ${isActive ? active : inactive}`;
}

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export default function TopNav() {
  const { user, signOut } = useAuth();

  const handleSignOut = useCallback(async () => {
    await signOut();
  }, [signOut]);

  if (!user) return null;

  const isAdmin = user.role === 'admin';

  return (
    <nav
      className={
        'sticky top-0 z-40 flex h-14 items-center justify-between border-b ' +
        'border-[var(--color-border)] bg-[var(--color-surface-1)]/95 ' +
        'px-4 backdrop-blur-sm md:px-6'
      }
    >
      {/* ---- Left: brand + links ---- */}
      <div className="flex items-center gap-6">
        {/* Brand */}
        <span className="text-lg font-bold tracking-wider text-[var(--color-accent)]">
          TAHIMIK
        </span>

        {/* Navigation links */}
        <div className="hidden items-center gap-1 sm:flex">
          {isAdmin ? (
            <NavLink to="/admin" className={navLinkClass}>
              Admin
            </NavLink>
          ) : (
            <>
              <NavLink to="/normalization" className={navLinkClass}>
                Normalization
              </NavLink>
              <NavLink to="/labeling" className={navLinkClass}>
                Labeling
              </NavLink>
            </>
          )}
        </div>
      </div>

      {/* ---- Right: rulebook + user + logout ---- */}
      <div className="flex items-center gap-3">
        {/* Rulebook button (annotators only) */}
        {!isAdmin && (
          <button
            type="button"
            onClick={toggleRulebook}
            className={
              'rounded-[var(--radius-md)] border border-[var(--color-border)] ' +
              'px-3 py-1.5 text-xs font-medium text-[var(--color-text-secondary)] ' +
              'transition-colors duration-200 hover:border-[var(--color-accent)] ' +
              'hover:text-[var(--color-accent)]'
            }
            aria-label="Open Rulebook"
          >
            📖 Rulebook
          </button>
        )}

        {/* User name */}
        <span className="hidden text-sm text-[var(--color-text-muted)] md:inline">
          {user.name}
        </span>

        {/* Logout */}
        <button
          type="button"
          onClick={handleSignOut}
          className={
            'rounded-[var(--radius-md)] bg-[var(--color-surface-2)] px-3 py-1.5 ' +
            'text-xs font-medium text-[var(--color-text-secondary)] ' +
            'transition-colors duration-200 hover:bg-[var(--color-surface-3)] ' +
            'hover:text-[var(--color-text-primary)]'
          }
        >
          Logout
        </button>
      </div>
    </nav>
  );
}
