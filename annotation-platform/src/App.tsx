import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import ProtectedRoute from './components/layout/ProtectedRoute';
import AppShell from './components/layout/AppShell';
import LoginPage from './pages/LoginPage';
import NormalizationPage from './pages/NormalizationPage';
import LabelingPage from './pages/LabelingPage';
import AdminPage from './pages/AdminPage';

/* ------------------------------------------------------------------ */
/*  RoleRedirect                                                       */
/*  Sends annotators → /normalization, admins → /admin.                */
/* ------------------------------------------------------------------ */

function RoleRedirect() {
  const { user } = useAuth();

  if (!user) return <Navigate to="/login" replace />;
  if (user.role === 'admin') return <Navigate to="/admin" replace />;
  return <Navigate to="/normalization" replace />;
}

/* ------------------------------------------------------------------ */
/*  App                                                                */
/* ------------------------------------------------------------------ */

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <Routes>
          {/* Public */}
          <Route path="/login" element={<LoginPage />} />

          {/* Protected (annotator + admin) */}
          <Route element={<ProtectedRoute />}>
            <Route element={<AppShell />}>
              {/* Role-based index redirect */}
              <Route index element={<RoleRedirect />} />

              {/* Annotator pages */}
              <Route path="normalization" element={<NormalizationPage />} />
              <Route path="labeling" element={<LabelingPage />} />

              {/* Admin-only */}
              <Route element={<ProtectedRoute requireAdmin />}>
                <Route path="admin" element={<AdminPage />} />
              </Route>
            </Route>
          </Route>

          {/* Catch-all */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  );
}
