import { useCallback, useState } from "react";
import { InfoAccordion } from "./components/InfoAccordion";
import { PageHeader } from "./components/PageHeader";
import { TextWorkspace, type ModelOption } from "./components/TextWorkspace";
import "./App.css";

/**
 * Base URL of the TAHIMIK inference API (FastAPI, see backend/app.py).
 * Override at build time with VITE_API_URL when the API is not on localhost.
 */
const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8100";

export type RequestStatus = "idle" | "loading" | "error";

export default function App() {
  const [inputText, setInputText] = useState("");
  const [outputText, setOutputText] = useState("");
  const [selectedModel, setSelectedModel] = useState<ModelOption>("");
  const [status, setStatus] = useState<RequestStatus>("idle");
  const [errorMessage, setErrorMessage] = useState("");
  const [latencyMs, setLatencyMs] = useState<number | null>(null);

  /**
   * Send the raw input to the backend and show the normalized result.
   * The chosen variant is passed through so the API can route to the
   * matching checkpoint.
   */
  const normalizeText = useCallback(async () => {
    const text = inputText.trim();

    if (!text) {
      setErrorMessage("Enter some text to normalize.");
      setStatus("error");
      return;
    }

    if (!selectedModel) {
      setErrorMessage("Choose a model before normalizing.");
      setStatus("error");
      return;
    }

    setStatus("loading");
    setErrorMessage("");
    setOutputText("");
    setLatencyMs(null);

    try {
      const response = await fetch(`${API_URL}/normalize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, model: selectedModel, num_beams: 4 }),
      });

      if (!response.ok) {
        const detail = await response
          .json()
          .then((body) => body.detail as string | undefined)
          .catch(() => undefined);
        throw new Error(detail ?? `Request failed (${response.status}).`);
      }

      const result = await response.json();
      setOutputText(result.normalized);
      setLatencyMs(result.inference_time_ms);
      setStatus("idle");
    } catch (error) {
      setErrorMessage(
        error instanceof Error
          ? error.message
          : "Could not reach the normalization API.",
      );
      setStatus("error");
    }
  }, [inputText, selectedModel]);

  const clearInput = useCallback(() => {
    setInputText("");
    setOutputText("");
    setErrorMessage("");
    setLatencyMs(null);
    setStatus("idle");
  }, []);

  return (
    <main className="app">
      <PageHeader />
      <TextWorkspace
        inputValue={inputText}
        outputValue={outputText}
        selectedModel={selectedModel}
        status={status}
        errorMessage={errorMessage}
        latencyMs={latencyMs}
        onInputChange={setInputText}
        onSelectModel={setSelectedModel}
        onNormalize={normalizeText}
        onClear={clearInput}
      />
      <div className="app__divider" />
      <InfoAccordion />
      <h1 className="app__title">TAHIMIK</h1>
      <p className="app__subtitle">
        Text Augmentation and Harmonization of Informal and Multilingual Input
        for Knowledge Extraction
      </p>
    </main>
  );
}
