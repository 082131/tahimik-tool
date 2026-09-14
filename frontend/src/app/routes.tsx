import { createHashRouter, Navigate, useLocation, useOutlet } from "react-router";
import { AnimatePresence, motion } from "motion/react";
import Landing from "./pages/Landing";
import Root from "./Root";

function PageTransitionLayout() {
  const location = useLocation();
  const outlet = useOutlet();
  const isLanding = location.pathname === "/" || location.pathname === "";

  return (
    <div className="relative w-full min-h-screen overflow-x-hidden bg-white">
      <AnimatePresence mode="popLayout" initial={false}>
        <motion.div
          key={location.pathname}
          initial={isLanding ? false : { opacity: 1 }}
          animate={{ x: 0, opacity: 1 }}
          exit={{ x: "-100%", opacity: 0.9 }}
          transition={{ duration: 0.85, ease: [0.22, 1, 0.36, 1] }}
          className="w-full min-h-screen"
          style={{
            zIndex: isLanding ? 20 : 1,
          }}
        >
          {outlet}
        </motion.div>
      </AnimatePresence>
    </div>
  );
}

export const router = createHashRouter([
  {
    element: <PageTransitionLayout />,
    children: [
      { path: "/",    element: <Landing /> },
      { path: "/app", element: <Root /> },
      { path: "*",    element: <Navigate to="/app" replace /> },
    ],
  },
]);

