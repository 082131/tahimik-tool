import { RouterProvider } from "react-router";
import { router } from "./routes";

export type RequestStatus = "idle" | "loading" | "error";

export default function App() {
  return <RouterProvider router={router} />;
}
