import { useState, useCallback, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';

export default function LoginPage() {
  const { signIn, user, loading: authLoading } = useAuth();
  const navigate = useNavigate();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  // Redirect if already logged in (in useEffect, not during render)
  useEffect(() => {
    if (!authLoading && user) {
      navigate(user.role === 'admin' ? '/admin' : '/normalization', { replace: true });
    }
  }, [user, authLoading, navigate]);

  // Safety: if signIn succeeded but profile never loads, show an error
  useEffect(() => {
    if (!loading) return;
    const timeout = setTimeout(() => {
      setLoading(false);
      setError('Sign-in succeeded but your profile was not found. Make sure you have a row in the users table.');
    }, 8000);
    return () => clearTimeout(timeout);
  }, [loading]);

  const handleSubmit = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email || !password) { setError('Please enter email and password.'); return; }

    setLoading(true);
    setError('');

    const { error: authError } = await signIn(email, password);
    if (authError) {
      setError(authError.message || 'Invalid credentials.');
      setLoading(false);
    }
    // On success, don't setLoading(false) yet — wait for useEffect redirect.
    // The 8-second safety timeout above catches cases where the profile is missing.
  }, [email, password, signIn]);

  return (
    <div className="flex min-h-screen items-center justify-center bg-[var(--color-surface-0)] p-4">
      <div className="glass-panel w-full max-w-sm p-8 animate-fade-in">
        {/* Branding */}
        <div className="mb-8 text-center">
          <h1 className="text-3xl font-bold tracking-tight text-[var(--color-accent)]">
            TAHIMIK
          </h1>
          <p className="mt-2 text-xs text-[var(--color-text-muted)] leading-relaxed">
            Filipino/Taglish Text Normalization<br />Annotation Platform
          </p>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="space-y-5">
          <div>
            <label htmlFor="login-email" className="mb-1.5 block text-xs font-medium text-[var(--color-text-secondary)]">
              Email
            </label>
            <input
              id="login-email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              autoComplete="email"
              className="w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-surface-2)] px-4 py-2.5 text-sm text-[var(--color-text-primary)] placeholder-[var(--color-text-muted)] transition-colors focus:border-[var(--color-border-focus)] focus:outline-none focus:ring-1 focus:ring-[var(--color-accent)]"
              placeholder="annotator@tahimik.study"
            />
          </div>

          <div>
            <label htmlFor="login-password" className="mb-1.5 block text-xs font-medium text-[var(--color-text-secondary)]">
              Password
            </label>
            <input
              id="login-password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete="current-password"
              className="w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-surface-2)] px-4 py-2.5 text-sm text-[var(--color-text-primary)] placeholder-[var(--color-text-muted)] transition-colors focus:border-[var(--color-border-focus)] focus:outline-none focus:ring-1 focus:ring-[var(--color-accent)]"
              placeholder="••••••••"
            />
          </div>

          {error && (
            <div className="rounded-lg bg-[var(--color-rose-muted)] px-4 py-2.5 text-xs text-[var(--color-rose)] animate-fade-in">
              {error}
            </div>
          )}

          <button
            id="login-submit"
            type="submit"
            disabled={loading}
            className="w-full rounded-lg bg-[var(--color-accent)] py-2.5 text-sm font-semibold text-[var(--color-text-inverse)] transition-all duration-200 hover:bg-[var(--color-accent-hover)] hover:shadow-lg disabled:cursor-not-allowed disabled:opacity-50"
          >
            {loading ? 'Signing in...' : 'Sign In'}
          </button>
        </form>

        <p className="mt-6 text-center text-[10px] text-[var(--color-text-muted)]">
          Closed annotation system — no public registration
        </p>
      </div>
    </div>
  );
}
