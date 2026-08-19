import { useRef } from "react";
import type { RequestStatus } from "../App";

/** "" means nothing picked yet — the trigger then reads "Choose model". */
export type ModelOption = "" | "byt5" | "mrt5" | "tahimik";

/**
 * The three variants compared in the study, ordered as they appear in the
 * paper: the accuracy ceiling, the efficiency baseline, then our proposal.
 */
const MODEL_OPTIONS: Array<{ value: Exclude<ModelOption, "">; label: string }> = [
  { value: "byt5", label: "ByT5" },
  { value: "mrt5", label: "MrT5" },
  { value: "tahimik", label: "TAHIMIK" },
];

type TextWorkspaceProps = {
  inputValue: string;
  outputValue: string;
  selectedModel: ModelOption;
  status: RequestStatus;
  errorMessage: string;
  latencyMs: number | null;
  onInputChange: (value: string) => void;
  onSelectModel: (model: ModelOption) => void;
  onNormalize: () => void;
  onClear: () => void;
};

type TextBoxProps = {
  label: string;
  labelVariant?: "default" | "success";
  children: React.ReactNode;
};

function TextBox({
  label,
  labelVariant = "default",
  children,
}: TextBoxProps) {
  const labelClassName = [
    "text-box__label",
    labelVariant === "success" ? "text-box__label--success" : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <section className="text-box">
      <div className={labelClassName}>{label}</div>
      {children}
    </section>
  );
}

export function TextWorkspace({
  inputValue,
  outputValue,
  selectedModel,
  status,
  errorMessage,
  latencyMs,
  onInputChange,
  onSelectModel,
  onNormalize,
  onClear,
}: TextWorkspaceProps) {
  const modelMenuRef = useRef<HTMLDetailsElement>(null);
  const isLoading = status === "loading";
  const selectedModelLabel =
    MODEL_OPTIONS.find((option) => option.value === selectedModel)?.label ??
    "Choose model";

  function chooseModel(model: Exclude<ModelOption, "">) {
    onSelectModel(model);
    modelMenuRef.current?.removeAttribute("open");
  }

  /** Ctrl/Cmd+Enter submits, matching the habit from most chat inputs. */
  function handleKeyDown(event: React.KeyboardEvent<HTMLTextAreaElement>) {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      onNormalize();
    }
  }

  return (
    <div className="text-workspace">
      <TextBox label="Input">
        <div className="text-box__field text-box__field--input">
          <textarea
            aria-label="Input text"
            className="text-box__textarea"
            placeholder="Enter text..."
            value={inputValue}
            onChange={(event) => onInputChange(event.target.value)}
            onKeyDown={handleKeyDown}
          />
          <div className="text-box__controls">
            <button
              type="button"
              className="text-box__control text-box__control--normalize"
              onClick={onNormalize}
              disabled={isLoading}
            >
              {isLoading ? "Normalizing..." : "Normalize"}
            </button>
            <button
              type="button"
              className="text-box__control text-box__control--clear"
              onClick={onClear}
              disabled={isLoading}
            >
              Clear Input
            </button>
          </div>
        </div>
      </TextBox>

      <TextBox label="Output" labelVariant="success">
        <div className="text-box__field text-box__field--output">
          <output
            aria-label="Normalized output"
            aria-live="polite"
            className="text-box__result"
          >
            {status === "error" ? (
              <span className="text-box__message text-box__message--error">
                {errorMessage}
              </span>
            ) : isLoading ? (
              <span className="text-box__message">Running the model...</span>
            ) : (
              outputValue
            )}
          </output>

          <div className="text-box__controls text-box__controls--output">
            {latencyMs !== null && status === "idle" && (
              <span className="text-box__latency">
                {Math.round(latencyMs)} ms
              </span>
            )}

            <details
              className={`model-select${selectedModel ? " is-selected" : ""}`}
              ref={modelMenuRef}
            >
              <summary className="model-select__trigger">
                {selectedModelLabel}
                <span className="model-select__arrow" aria-hidden="true" />
              </summary>
              <div
                className="model-select__menu"
                role="listbox"
                aria-label="Normalization model"
              >
                {MODEL_OPTIONS.map((option) => (
                  <button
                    type="button"
                    role="option"
                    aria-selected={selectedModel === option.value}
                    className={`model-select__option${
                      selectedModel === option.value ? " is-selected" : ""
                    }`}
                    key={option.value}
                    onClick={() => chooseModel(option.value)}
                  >
                    {option.label}
                  </button>
                ))}
              </div>
            </details>
          </div>
        </div>
      </TextBox>
    </div>
  );
}
