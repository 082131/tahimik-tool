import { useRef, useState } from "react";
import { parseDatasetFile, type ParsedSentence } from "../utils/datasetParser";

export default function NormalizationEvaluation() {
  const fileRef = useRef<HTMLInputElement>(null);
  const [rows, setRows] = useState<ParsedSentence[]>([]);
  const [message, setMessage] = useState("Upload a held-out test set with noisy input and clean reference columns.");

  const loadFile = async (file?: File) => {
    if (!file) return;
    const parsed = parseDatasetFile(await file.text());
    setRows(parsed.sentences);
    const paired = parsed.sentences.filter((row) => row.reference?.trim()).length;
    setMessage(paired === parsed.sentences.length && paired > 0
      ? `${paired} paired examples are ready for evaluation.`
      : `${parsed.sentences.length} inputs found; every row needs a clean reference before evaluation can run.`);
  };

  const paired = rows.length > 0 && rows.every((row) => row.reference?.trim());
  return <div className="flex flex-col gap-5">
    <div className="flex items-center gap-3"><span className="font-medium text-[clamp(.65rem,.9vw,12px)] uppercase tracking-widest text-[#999]">Normalization Evaluation</span><span className="flex-1 h-px bg-black/10" /></div>
    <div className="rounded-[24px] bg-white p-6 sm:p-7 flex flex-col gap-4" style={{ border: "0.5px solid rgba(0,0,0,0.5)" }}>
      <p className="text-sm text-black/70">This workflow evaluates a labelled held-out test set. Single text and unlabelled batches remain in Text Normalization and show model outputs plus compression and gating information only.</p>
      <input ref={fileRef} className="hidden" type="file" accept=".csv,.tsv,.json,.jsonl,.txt" onChange={(event) => loadFile(event.target.files?.[0])} />
      <div className="flex flex-wrap items-center gap-3"><button type="button" onClick={() => fileRef.current?.click()} className="rounded-full border border-black/50 bg-white px-4 py-2 text-xs font-medium">Upload labelled test set</button><span className="text-xs text-black/50">Required fields: <code>input</code> and <code>reference</code></span></div>
      <p className={`text-xs ${paired ? "text-black/70" : "text-black/45"}`}>{message}</p>
      {rows.length > 0 && <div className="max-h-44 overflow-auto rounded-xl border border-black/10 text-xs"><div className="grid grid-cols-2 gap-3 p-3 font-medium border-b border-black/10"><span>Noisy input</span><span>Clean reference</span></div>{rows.slice(0, 10).map((row, index) => <div className="grid grid-cols-2 gap-3 p-3 border-b border-black/5" key={`${row.input}-${index}`}><span>{row.input}</span><span>{row.reference || "Missing reference"}</span></div>)}</div>}
      <button type="button" disabled className="self-start rounded-full bg-black px-4 py-2 text-xs font-medium text-white disabled:opacity-40">Run Evaluation &amp; Profile</button>
      <p className="text-[11px] text-black/45">The run button becomes available when the labelled evaluation API is connected.</p>
    </div>
  </div>;
}
