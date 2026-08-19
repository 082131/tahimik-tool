import { Navigate, Outlet } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';

/* ------------------------------------------------------------------ */
/*  Props                                                              */
/* ------------------------------------------------------------------ */

interface ProtectedRouteProps {
  /** When true, only users with role === 'admin' may access this route. */
  requireAdmin?: boolean;
}

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export default function ProtectedRoute({ requireAdmin = false }: ProtectedRouteProps) {
  const { user, loading } = useAuth();

  /* ---- Loading state ---- */
  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-[var(--color-surface-0)]">
        <div className="flex flex-col items-center gap-3">
          <svg
            className="h-8 w-8 animate-spin text-[var(--color-accent)]"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            aria-label="Loading"
          >
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
            />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
            />
          </svg>
          <span className="text-sm text-[var(--color-text-muted)]">
            Authenticating…
          </span>
        </div>
      </div>
    );
  }

  /* ---- Not authenticated ---- */
  if (!user) {
    return <Navigate to="/login" replace />;
  }

  /* ---- Admin guard ---- */
  if (requireAdmin && user.role !== 'admin') {
    return <Navigate to="/" replace />;
  }

  /* ---- Authorized ---- */
  return <Outlet />;
}
