import { useRef, useState, useEffect, useCallback } from "react";
import { Pill, KpiCard } from "../shared";

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8100";
const DATA_MODE = import.meta.env.VITE_TAHIMIK_DATA_MODE ?? "demo";
const IS_LIVE_MODE = DATA_MODE === "live";

// ── Palette ──────────────────────────────────────────────────────────────

const LIME  = "#DDEF75";
const LILAC = "#D4A7E0";
const CORAL = "#F0A080";

// ── Types ────────────────────────────────────────────────────────────────

export type SentenceData = {
  input:   string;
  byt5:    string;
  mrt5:    string;
  tahimik: string;
  noise:   number;   // 0–1
  pruning: number;   // %
  latencyByt5?: string;
  latencyMrt5?: string;
  latencyTahimik?: string;
  tokens:  Token[];
  /** Fixed byte positions for the presentation scenario; live runs supply backend telemetry instead. */
  bytePrunedPositions?: number[];
  reference?: string;
  isFallback?: boolean;
  isDemo?: boolean;
};

type CompareModelResult = {
  model: "byt5" | "mrt5" | "tahimik";
  normalized: string;
  inference_time_ms: number;
  telemetry?: Record<string, unknown>;
};

const DEMO_SENTENCE: SentenceData = {
  input: "Sanaol nlng tlga sa inyo mga lodi ang ganda ng araw nyo dyan!",
  byt5: "Sana all na lang talaga sa inyo mga lodi, ang ganda ng araw ninyo diyan!",
  mrt5: "Sanaol lang sa inyo mga, ang ganda ng araw.",
  tahimik: "Sana all na lang talaga sa inyo mga lodi, ang ganda ng araw ninyo diyan!",
  noise: 0.38,
  pruning: 31,
  latencyByt5: "138 ms",
  latencyMrt5: "74 ms",
  latencyTahimik: "82 ms",
  tokens: [
    { text: "Sanaol", kept: true }, { text: "nlng", kept: false },
    { text: "tlga", kept: false }, { text: "sa", kept: true },
    { text: "inyo", kept: true }, { text: "mga", kept: true },
    { text: "lodi", kept: false }, { text: "ang", kept: true },
    { text: "ganda", kept: true }, { text: "ng", kept: true },
    { text: "araw", kept: true }, { text: "nyo", kept: false },
    { text: "dyan!", kept: false },
  ],
  // 19 of the 61 ASCII byte positions are removed: 31%, matching the target shown below.
  // The positions are intentionally scattered so the visual does not imply word deletion.
  bytePrunedPositions: [1, 4, 8, 10, 13, 15, 18, 21, 23, 26, 29, 32, 35, 39, 42, 45, 49, 53, 58],
  isDemo: true,
};

// ── Sample Presets ───────────────────────────────────────────────────────

const PRESET_BATCH: SentenceData[] = [
  {
    input:   "Sanaol nlng tlga sa inyo mga lodi ang ganda ng araw nyo dyan!",
    byt5:    "Sana all na lang talaga sa inyo mga lodi, ang ganda ng araw ninyo diyan!",
    mrt5:    "Sanaol lang sa inyo mga, ang ng araw.",
    tahimik: "Sana all na lang talaga sa inyo mga lodi, ang ganda ng araw ninyo diyan!",
    noise:   0.38,
    pruning: 31,
    latencyByt5: "138 ms",
    latencyMrt5: "74 ms",
    latencyTahimik: "82 ms",
    tokens:  [
      { text: "Sanaol", kept: true  }, { text: "nlng",   kept: false },
      { text: "tlga",   kept: false }, { text: "sa",     kept: true  },
      { text: "inyo",   kept: true  }, { text: "mga",    kept: true  },
      { text: "lodi",   kept: false }, { text: "ang",    kept: true  },
      { text: "ganda",  kept: true  }, { text: "ng",     kept: true  },
      { text: "araw",   kept: true  }, { text: "nyo",    kept: false },
      { text: "dyan!",  kept: false },
    ],
  },
  {
    input:   "Kumusta ka na besh? Matagal na tayong di nagkita!",
    byt5:    "Kumusta ka na best? Matagal na tayong hindi nagkita!",
    mrt5:    "Kumusta ka na besh? Matagal tayong di nagkita!",
    tahimik: "Kumusta ka na best? Matagal na tayong hindi nagkita!",
    noise:   0.22,
    pruning: 19,
    latencyByt5: "135 ms",
    latencyMrt5: "72 ms",
    latencyTahimik: "80 ms",
    tokens:  [
      { text: "Kumusta", kept: true  }, { text: "ka",     kept: true  },
      { text: "na",      kept: true  }, { text: "besh?",  kept: false },
      { text: "Matagal", kept: true  }, { text: "na",     kept: true  },
      { text: "tayong",  kept: true  }, { text: "di",     kept: false },
      { text: "nagkita!", kept: true },
    ],
  },
  {
    input:   "Lodi grabe ka tlga ang galing mo!",
    byt5:    "Lodi, grabe ka talaga, ang galing mo!",
    mrt5:    "Lodi grabe ka tlga, ang galing mo.",
    tahimik: "Lodi, grabe ka talaga, ang galing mo!",
    noise:   0.18,
    pruning: 14,
    latencyByt5: "128 ms",
    latencyMrt5: "70 ms",
    latencyTahimik: "78 ms",
    tokens:  [
      { text: "Lodi",   kept: false }, { text: "grabe",  kept: true  },
      { text: "ka",     kept: true  }, { text: "tlga",   kept: false },
      { text: "ang",    kept: true  }, { text: "galing", kept: true  },
      { text: "mo!",    kept: true  },
    ],
  },
  {
    input:   "Nkklk na ko sa trabaho bro, sana matapos na.",
    byt5:    "Nakakainis na ako sa trabaho bro, sana matapos na.",
    mrt5:    "Nkklk na ko sa trabaho bro, sana matapos na.",
    tahimik: "Nakakainis na ako sa trabaho bro, sana matapos na.",
    noise:   0.45,
    pruning: 36,
    latencyByt5: "142 ms",
    latencyMrt5: "76 ms",
    latencyTahimik: "84 ms",
    tokens:  [
      { text: "Nkklk",    kept: false }, { text: "na",      kept: true  },
      { text: "ko",       kept: false }, { text: "sa",      kept: true  },
      { text: "trabaho",  kept: true  }, { text: "bro,",    kept: true  },
      { text: "sana",     kept: true  }, { text: "matapos", kept: true  },
      { text: "na.",      kept: true  },
    ],
  },
  {
    input:   "Lam mo ba yung bagong resto sa may SM? Sulit daw!",
    byt5:    "Alam mo ba yung bagong resto sa may SM? Sulit daw!",
    mrt5:    "Lam mo ba yung bagong resto sa may SM? Sulit daw!",
    tahimik: "Alam mo ba yung bagong resto sa may SM? Sulit daw!",
    noise:   0.12,
    pruning: 9,
    latencyByt5: "132 ms",
    latencyMrt5: "71 ms",
    latencyTahimik: "79 ms",
    tokens:  [
      { text: "Lam",     kept: false }, { text: "mo",     kept: true  },
      { text: "ba",      kept: true  }, { text: "yung",   kept: true  },
      { text: "bagong",  kept: true  }, { text: "resto",  kept: true  },
      { text: "sa",      kept: true  }, { text: "may",    kept: true  },
      { text: "SM?",     kept: true  }, { text: "Sulit",  kept: true  },
      { text: "daw!",    kept: true  },
    ],
  },
];

import {
  type Token,
  parseDatasetFile,
  estimateSentenceGating,
} from "../utils/datasetParser";

// ── Shared UI primitives ─────────────────────────────────────────────────

function IPill({ children, bg = "white" }: { children: React.ReactNode; bg?: string }) {
  return (
    <div
      className="inline-flex items-center px-3 py-1 rounded-full shrink-0"
      style={{ border: "0.5px solid rgba(0,0,0,0.5)", backgroundColor: bg }}
    >
      <span className="font-['Inter',sans-serif] font-medium text-[clamp(0.6rem,0.85vw,11px)] text-black whitespace-nowrap">
        {children}
      </span>
    </div>
  );
}

function CardLabel({ children }: { children: React.ReactNode }) {
  return (
    <span className="font-['Inter',sans-serif] font-semibold text-[clamp(0.7rem,1.05vw,13px)] text-black tracking-tight">
      {children}
    </span>
  );
}

function SubLabel({ children }: { children: React.ReactNode }) {
  return (
    <span className="font-['Inter',sans-serif] font-medium text-[clamp(0.58rem,0.85vw,11px)] text-[#7A7A7A] uppercase tracking-wide">
      {children}
    </span>
  );
}

// ── Gating Inspector ─────────────────────────────────────────────────────

function GatingInspector({ sentence, isLoading }: { sentence: SentenceData; isLoading?: boolean }) {
  const [telemetryModel, setTelemetryModel] = useState<"tahimik" | "mrt5">("tahimik");

  if (isLoading) {
    return (
      <div className="flex flex-col gap-4 w-full min-w-0 max-w-full animate-pulse">
        {/* Card 1 — Compression Overview Skeleton */}
        <div className="w-full rounded-[24px] bg-white p-4 sm:p-6 flex flex-col gap-4 min-w-0" style={{ border: "0.5px solid rgba(0,0,0,0.5)" }}>
          <div className="flex items-center justify-between flex-wrap gap-2">
            <CardLabel>Compression Overview</CardLabel>
            <span className="flex items-center gap-1.5 font-['Inter',sans-serif] font-medium text-[11px] text-black/70">
              <span className="w-2 h-2 rounded-full bg-[#fafe45] animate-ping" />
              Computing model compression rates...
            </span>
          </div>
          <div className="grid gap-3" style={{ gridTemplateColumns: "repeat(3, minmax(180px, 1fr))", overflowX: "auto" }}>
            {[
              { name: "ByT5", tag: "BASELINE", bg: "rgba(0,0,0,0.03)" },
              { name: "MrT5", tag: "FIXED 50%", bg: "rgba(240,160,128,0.12)" },
              { name: "TAHIMIK Adaptive", tag: "ADAPTIVE", bg: "rgba(221,239,117,0.18)" },
            ].map((m) => (
              <div key={m.name} className="flex flex-col gap-3 rounded-[16px] p-4" style={{ border: "0.5px solid rgba(0,0,0,0.15)", backgroundColor: m.bg }}>
                <div className="flex items-center justify-between gap-1">
                  <span className="font-['Inter',sans-serif] font-semibold text-[clamp(0.7rem,1.1vw,13px)] text-black">{m.name}</span>
                  <IPill bg="white">{m.tag}</IPill>
                </div>
                <div className="w-full h-2 rounded-full bg-black/10 overflow-hidden relative">
                  <div className="h-full bg-black/20 rounded-full w-2/3 animate-pulse" />
                </div>
                <div className="flex justify-between">
                  <span className="h-2.5 bg-black/10 rounded w-16" />
                  <span className="h-2.5 bg-black/10 rounded w-14" />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Card 2 — Neural Noise & Gate Diagnostics Skeleton */}
        <div className="w-full rounded-[24px] bg-white p-5 sm:p-6 flex flex-col gap-4 min-w-0" style={{ border: "0.5px solid rgba(0,0,0,0.5)" }}>
          <div className="flex items-center gap-3">
            <CardLabel>Neural Noise & Gate Diagnostics</CardLabel>
            <span className="font-['Inter',sans-serif] font-normal text-[clamp(0.58rem,0.85vw,11px)] text-[#999]">
              — Estimating character noise & delete gate mechanics...
            </span>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
            <div className="flex flex-col gap-3">
              <SubLabel>Noise Estimator</SubLabel>
              <div className="w-full h-3 rounded-full bg-black/10 overflow-hidden">
                <div className="h-full bg-[#f0a080] rounded-full w-1/3 animate-pulse" />
              </div>
              <div className="h-7 bg-black/10 rounded-full w-3/4" />
            </div>
            <div className="flex flex-col gap-3">
              <SubLabel>Delete Gate Mechanics</SubLabel>
              <div className="flex flex-wrap gap-2">
                <div className="h-6 w-20 bg-black/10 rounded-full" />
                <div className="h-6 w-24 bg-black/10 rounded-full" />
                <div className="h-6 w-16 bg-black/10 rounded-full" />
              </div>
              <div className="h-6 bg-black/10 rounded-full w-2/3" />
            </div>
          </div>
        </div>

        {/* Card 3 — Byte Gating Decision Map Skeleton */}
        <div className="w-full rounded-[24px] bg-white p-5 sm:p-6 flex flex-col gap-4 min-w-0" style={{ border: "0.5px solid rgba(0,0,0,0.5)" }}>
          <div className="flex items-center justify-between flex-wrap gap-2">
            <CardLabel>Byte Gating Decision Map</CardLabel>
            <span className="font-['Inter',sans-serif] font-normal text-xs text-[#999]">Evaluating adaptive byte masks...</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {["Token", "Noise", "Evaluation", "Gate", "Shift", "Bytes", "Capacity"].map((lbl, idx) => (
              <div
                key={idx}
                className="h-7 px-3.5 rounded-full bg-black/5 flex items-center justify-center font-['Inter',sans-serif] text-xs text-black/40"
                style={{ border: "0.5px solid rgba(0,0,0,0.1)" }}
              >
                {lbl}
              </div>
            ))}
          </div>
        </div>

        {/* Card 4 — Normalized Output Comparison Skeleton */}
        <div className="w-full rounded-[24px] bg-white p-4 sm:p-6 flex flex-col gap-4 min-w-0" style={{ border: "0.5px solid rgba(0,0,0,0.5)" }}>
          <CardLabel>Normalized Output Comparison</CardLabel>
          <div className="grid gap-3" style={{ gridTemplateColumns: "repeat(3, minmax(180px, 1fr))", overflowX: "auto" }}>
            {[1, 2, 3].map((k) => (
              <div key={k} className="flex flex-col gap-2.5 rounded-[16px] p-4 bg-black/[0.02]" style={{ border: "0.5px solid rgba(0,0,0,0.1)" }}>
                <div className="h-4 bg-black/10 rounded w-1/3" />
                <div className="h-4 bg-black/10 rounded w-full" />
                <div className="h-4 bg-black/10 rounded w-4/5" />
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  const totalBytes  = new TextEncoder().encode(sentence.input).length;
  const keptBytes   = Math.round(totalBytes * (1 - sentence.pruning / 100));
  const keptTokens  = sentence.tokens.filter((t) => t.kept).length;
  const pruneTokens = sentence.tokens.filter((t) => !t.kept).length;
  const byteMapAvailable =
    telemetryModel === "tahimik" &&
    Boolean(sentence.bytePrunedPositions) &&
    totalBytes === sentence.input.length;
  const prunedBytePositions = new Set(sentence.bytePrunedPositions ?? []);
  const mappedPrunedBytes = prunedBytePositions.size;
  const mappedKeptBytes = totalBytes - mappedPrunedBytes;

  const models = [
    {
      name: "ByT5",
      tag: "BASELINE",
      pruned: 0,
      kept: totalBytes,
      total: totalBytes,
      latency: sentence.latencyByt5 || "138 ms",
      primary: false,
      output: sentence.byt5,
      telemetryKey: null as null | "tahimik" | "mrt5",
    },
    {
      name: "MrT5",
      tag: "FIXED 50%",
      pruned: 50,
      kept: Math.round(totalBytes * 0.5),
      total: totalBytes,
      latency: sentence.latencyMrt5 || "74 ms",
      primary: false,
      output: sentence.mrt5,
      telemetryKey: "mrt5" as null | "tahimik" | "mrt5",
    },
    {
      name: "TAHIMIK Adaptive",
      tag: "ADAPTIVE",
      pruned: sentence.pruning,
      kept: keptBytes,
      total: totalBytes,
      latency: sentence.latencyTahimik || "82 ms",
      primary: true,
      output: sentence.tahimik,
      telemetryKey: "tahimik" as null | "tahimik" | "mrt5",
    },
  ];

  return (
    <div className="flex flex-col gap-4 w-full min-w-0 max-w-full">

      {/* Card 1 — Compression Overview */}
      <div className="w-full rounded-[24px] bg-white p-4 sm:p-6 flex flex-col gap-4 min-w-0" style={{ border: "0.5px solid rgba(0,0,0,0.5)" }}>
        <CardLabel>Compression Overview</CardLabel>
        <div className="grid gap-3" style={{ gridTemplateColumns: "repeat(3, minmax(180px, 1fr))", overflowX: "auto" }}>
          {models.map((m) => {
            const isActive = m.telemetryKey !== null && telemetryModel === m.telemetryKey;
            return (
              <div
                key={m.name}
                onClick={() => m.telemetryKey && setTelemetryModel(m.telemetryKey)}
                className="flex flex-col gap-3 rounded-[16px] p-4 transition-all"
                style={{
                  cursor:          m.telemetryKey ? "pointer" : "default",
                  border:          isActive
                    ? "0.5px solid rgba(0,0,0,0.6)"
                    : "0.5px solid rgba(0,0,0,0.15)",
                  backgroundColor: isActive
                    ? (m.telemetryKey === "tahimik" ? "rgba(221,239,117,0.18)" : "rgba(240,160,128,0.12)")
                    : "white",
                  boxShadow: isActive ? "inset 0 0 0 1px rgba(0,0,0,0.06)" : "none",
                }}
              >
                <div className="flex items-center justify-between gap-2 flex-wrap">
                  <span className="font-['Inter',sans-serif] font-semibold text-[clamp(0.7rem,1.1vw,13px)] text-black">{m.name}</span>
                  <div className="flex items-center gap-1.5">
                    {isActive && (
                      <span className="font-['Inter',sans-serif] font-medium text-[9px] text-[#7A7A7A] uppercase tracking-wide">inspecting</span>
                    )}
                    <IPill bg={m.primary ? LIME : m.pruned === 50 ? CORAL : "white"}>{m.tag}</IPill>
                  </div>
                </div>
                <div className="flex flex-col gap-1.5">
                  <div className="w-full h-2 rounded-full overflow-hidden" style={{ backgroundColor: "rgba(0,0,0,0.07)" }}>
                    <div
                      className="h-full rounded-full"
                      style={{
                        width:           `${100 - m.pruned}%`,
                        backgroundColor: m.primary ? LIME : m.pruned === 50 ? CORAL : "#d0d0d0",
                      }}
                    />
                  </div>
                  <div className="flex justify-between">
                    <span className="font-['Inter',sans-serif] font-normal text-[10px] text-[#aaa]">{m.pruned}% pruned</span>
                    <span className="font-['Inter',sans-serif] font-normal text-[10px] text-[#aaa]">{m.kept}/{m.total} bytes</span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Unified Telemetry & Byte Gating Pipeline Card */}
      <div className="w-full rounded-[24px] bg-white p-5 sm:p-6 flex flex-col gap-6 min-w-0 shadow-xs" style={{ border: "0.5px solid rgba(0,0,0,0.5)" }}>
        {/* Header with Title and Byte Gating Summary */}
        <div className="flex items-center justify-between gap-3 flex-wrap min-w-0">
          <div className="flex items-center gap-3">
            <CardLabel>Noise Estimator & Byte Gating Mechanics</CardLabel>
            <span className="font-['Inter',sans-serif] font-normal text-[clamp(0.58rem,0.85vw,11px)] text-[#888]">
              — {telemetryModel === "tahimik" ? "TAHIMIK Adaptive" : "MrT5"}
            </span>
          </div>
          <span className="font-['Inter',sans-serif] font-medium text-[clamp(0.58rem,0.85vw,11px)] text-black/70 whitespace-nowrap">
            {byteMapAvailable
              ? `${mappedKeptBytes} retained • ${mappedPrunedBytes} removed • ${totalBytes} total bytes`
              : `${keptBytes} retained • ${totalBytes - keptBytes} removed • ${totalBytes} total bytes`}
          </span>
        </div>

        {/* Stage 1: Noise Estimation & Target Deletion Calculation */}
        {telemetryModel === "tahimik" ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="flex flex-col gap-3">
              <SubLabel>Stage 1 — Noise Density Estimation</SubLabel>
              <div className="flex flex-col gap-2">
                <div className="w-full h-3 rounded-full overflow-hidden" style={{ backgroundColor: "rgba(0,0,0,0.07)" }}>
                  <div className="h-full rounded-full transition-all duration-300" style={{ width: `${Math.max(sentence.noise * 100, 3)}%`, backgroundColor: CORAL }} />
                </div>
                <div className="flex justify-between">
                  <span className="font-['Inter',sans-serif] font-normal text-[10px] text-[#aaa]">Clean (0.0)</span>
                  <span className="font-['Inter',sans-serif] font-semibold text-[10px] text-black">η<sub>i</sub> = {sentence.noise.toFixed(2)}</span>
                  <span className="font-['Inter',sans-serif] font-normal text-[10px] text-[#aaa]">Extreme (1.0)</span>
                </div>
              </div>
              <IPill bg="rgba(221,239,117,0.55)">
                <span>Target Deletion = 0.50 × (1 − {sentence.noise.toFixed(2)}) = {sentence.pruning}%</span>
              </IPill>
            </div>
            <div className="flex flex-col gap-3">
              <SubLabel>Stage 2 — Score Each Byte</SubLabel>
              <div className="flex flex-wrap gap-2">
                <IPill bg={LILAC}>Gate scores every byte position</IPill>
                <IPill bg={LILAC}>Each score becomes retain or remove</IPill>
              </div>
              <IPill bg="white">
                Noise sets the deletion target; the gate decides individual bytes, not words.
              </IPill>
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="flex flex-col gap-3">
              <SubLabel>Stage 1 — Noise Density Estimation</SubLabel>
              <div className="flex flex-col gap-2">
                <div className="w-full h-3 rounded-full overflow-hidden" style={{ backgroundColor: "rgba(0,0,0,0.07)" }}>
                  <div className="h-full rounded-full" style={{ width: `${sentence.noise * 100}%`, backgroundColor: "#d0d0d0" }} />
                </div>
                <div className="flex justify-between">
                  <span className="font-['Inter',sans-serif] font-normal text-[10px] text-[#aaa]">Clean (0.0)</span>
                  <span className="font-['Inter',sans-serif] font-semibold text-[10px] text-[#7A7A7A]">η<sub>i</sub> = {sentence.noise.toFixed(2)} (ignored)</span>
                  <span className="font-['Inter',sans-serif] font-normal text-[10px] text-[#aaa]">Extreme (1.0)</span>
                </div>
              </div>
              <IPill bg={CORAL}>Fixed Deletion Rate = 50% (noise-agnostic)</IPill>
            </div>
            <div className="flex flex-col gap-3">
              <SubLabel>Stage 2 — Static Delete Gate</SubLabel>
              <div className="flex flex-wrap gap-2">
                <IPill bg="rgba(0,0,0,0.05)">Target Rate = 0.50</IPill>
                <IPill bg="rgba(0,0,0,0.05)"><span>Noise Input = Disabled</span></IPill>
                <IPill bg="rgba(0,0,0,0.05)">Masking = Static</IPill>
              </div>
              <IPill bg="white">No adaptive shift — uniform pruning regardless of noise</IPill>
            </div>
          </div>
        )}

        {/* Visual Pipeline Connector */}
        <div className="flex items-center gap-3 w-full">
          <span className="flex-1 h-px bg-black/10 min-w-[20px]" />
          <span className="font-['Inter',sans-serif] font-medium text-[10px] uppercase tracking-wider text-[#888] px-2">
            Stage 3 — Per-byte Gate Decisions
          </span>
          <span className="flex-1 h-px bg-black/10 min-w-[20px]" />
        </div>

        {/* Stage 3: Byte Gating Decision Map */}
        <div className="flex flex-col gap-3.5">
          {byteMapAvailable ? (
            <>
              <div className="flex flex-wrap gap-x-3.5 gap-y-3 items-center w-full min-w-0">
                {(() => {
                  const words = sentence.input.split(/(\s+)/);
                  let bytePosition = 0;
                  return words.map((word, wordIndex) => {
                    if (!word) return null;
                    if (/^\s+$/.test(word)) {
                      bytePosition += word.length;
                      return null;
                    }

                    const byteElements = Array.from(word).map((character, characterIndex) => {
                      const position = bytePosition++;
                      const retained = !prunedBytePositions.has(position);
                      return (
                        <span
                          key={characterIndex}
                          className={`inline-flex items-center justify-center min-w-[19px] h-[26px] px-1 rounded-[4px] font-['Inter',sans-serif] text-[13px] transition-colors ${
                            retained
                              ? "bg-[#ddf075] text-black font-medium border border-black/25"
                              : "bg-black/[0.04] text-black/35 line-through border border-black/10"
                          }`}
                          title={`Byte position ${position}: ${retained ? "RETAINED" : "REMOVED"}`}
                        >
                          {character}
                        </span>
                      );
                    });
                    return (
                      <div key={wordIndex} className="inline-flex items-center gap-[2px] p-1 rounded-[8px] bg-black/[0.02] border border-black/10">
                        {byteElements}
                      </div>
                    );
                  });
                })()}
              </div>
              <div className="flex flex-wrap gap-2">
                <IPill bg="rgba(221,239,117,0.55)">
                  η<sub>i</sub> = {sentence.noise.toFixed(2)} → {sentence.pruning}% target → {mappedPrunedBytes}/{totalBytes} bytes removed
                </IPill>
              </div>
            </>
          ) : (
            <div className="rounded-[12px] border border-black/10 bg-black/[0.02] px-3 py-2.5 font-['Inter',sans-serif] text-[11px] text-[#666]">
              Per-byte decisions appear here when byte-position telemetry is available for the selected run.
            </div>
          )}

          {/* Legend and Flow Note */}
          <div className="flex items-center justify-between flex-wrap gap-2 pt-1">
            <div className="flex items-center gap-2">
              <IPill bg={LIME}>● KEPT</IPill>
              <IPill bg="white">○ PRUNED</IPill>
            </div>
            <span className="font-['Inter',sans-serif] text-[10px] text-[#888]">
              Word outlines are reading guides only; the gate acts on individual bytes.
            </span>
          </div>
        </div>
      </div>

      {/* Card 4 — Normalized Output Comparison */}
      <div className="w-full rounded-[24px] bg-white p-4 sm:p-6 flex flex-col gap-4 min-w-0" style={{ border: "0.5px solid rgba(0,0,0,0.5)" }}>
        <CardLabel>Normalized Output Comparison</CardLabel>
        <div className="grid gap-3" style={{ gridTemplateColumns: "repeat(3, minmax(180px, 1fr))", overflowX: "auto" }}>
          {models.map((m) => (
            <div
              key={m.name}
              className="flex flex-col gap-2.5 rounded-[16px] p-4 min-w-0"
              style={{
                border:          m.primary ? "0.5px solid rgba(0,0,0,0.5)" : "0.5px solid rgba(0,0,0,0.15)",
                backgroundColor: m.primary ? "rgba(221,239,117,0.09)" : "white",
              }}
            >
              <div className="flex items-center gap-2">
                <SubLabel>{m.name}</SubLabel>
                {m.primary && (
                  <div className="w-1.5 h-1.5 rounded-full shrink-0" style={{ backgroundColor: LIME, border: "0.5px solid rgba(0,0,0,0.4)" }} />
                )}
              </div>
              <p className="font-['Inter',sans-serif] font-normal text-[clamp(0.78rem,1.3vw,15px)] text-black leading-relaxed break-words [overflow-wrap:anywhere]">
                {m.output}
              </p>
            </div>
          ))}
        </div>
      </div>

    </div>
  );
}

// ── Batch Summary ────────────────────────────────────────────────────────

const MODEL_META = {
  byt5:    { label: "ByT5",          pruneOf: (_s: SentenceData) => 0,     color: "white",  tagColor: "white", tag: "0% pruned"   },
  mrt5:    { label: "MrT5",          pruneOf: (_s: SentenceData) => 50,    color: CORAL,    tagColor: CORAL,   tag: "fixed 50%"   },
  tahimik: { label: "TAHIMIK",       pruneOf: (s: SentenceData) => s.pruning, color: LIME,  tagColor: LIME,    tag: "adaptive"    },
} as const;

type ModelKey = keyof typeof MODEL_META;

function BatchSummary({ rows, isLoading }: { rows: SentenceData[]; isLoading?: boolean }) {
  const [model, setModel] = useState<ModelKey>("tahimik");

  const meta       = MODEL_META[model];
  const total      = rows.length;
  const avgNoise   = total > 0 ? (rows.reduce((a, b) => a + b.noise, 0) / total).toFixed(2) : "0.00";
  const avgPrune   = total > 0 ? Math.round(rows.reduce((a, b) => a + meta.pruneOf(b), 0) / total) : 0;
  const totalBytes = rows.reduce((a, b) => a + new TextEncoder().encode(b.input).length, 0);

  const noiseBuckets = [
    { label: "Low (< 0.2)",    count: rows.filter((s) => s.noise < 0.2).length,  color: LIME  },
    { label: "Mid (0.2–0.35)", count: rows.filter((s) => s.noise >= 0.2 && s.noise < 0.35).length, color: "#fafe45" },
    { label: "High (≥ 0.35)",  count: rows.filter((s) => s.noise >= 0.35).length, color: CORAL },
  ];

  if (isLoading) {
    return (
      <div className="flex flex-col gap-4 w-full min-w-0 animate-pulse">
        <div className="w-full rounded-[24px] bg-white p-5 sm:p-6 flex flex-col gap-5 min-w-0" style={{ border: "0.5px solid rgba(0,0,0,0.5)" }}>
          <div className="flex items-center justify-between gap-3 flex-wrap">
            <CardLabel>Batch Summary — Processing {total} Sentences</CardLabel>
            <span className="flex items-center gap-1.5 font-['Inter',sans-serif] text-xs text-black/70">
              <span className="w-2 h-2 rounded-full bg-[#fafe45] animate-ping" />
              Aggregating batch telemetry...
            </span>
          </div>
          <div className="flex flex-wrap gap-4">
            {[1, 2, 3, 4].map((k) => (
              <div key={k} className="h-24 w-40 bg-black/5 rounded-[16px] animate-pulse" />
            ))}
          </div>
          <div className="h-20 bg-black/5 rounded-[16px] animate-pulse" />
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4 w-full min-w-0">
      <div className="w-full rounded-[24px] bg-white p-5 sm:p-6 flex flex-col gap-5 min-w-0" style={{ border: "0.5px solid rgba(0,0,0,0.5)" }}>

        {/* Header + model selector */}
        <div className="flex items-center justify-between gap-3 flex-wrap">
          <CardLabel>Batch Summary — {total} Sentences</CardLabel>
          <div className="flex items-center gap-1.5 flex-wrap">
            {(Object.keys(MODEL_META) as ModelKey[]).map((m) => (
              <button
                key={m}
                onClick={() => setModel(m)}
                className="cursor-pointer px-3 py-1 rounded-full font-['Inter',sans-serif] font-medium text-[clamp(0.55rem,0.85vw,11px)] text-black transition-all hover:opacity-80"
                style={{
                  border:          "0.5px solid rgba(0,0,0,0.35)",
                  backgroundColor: model === m ? MODEL_META[m].color : "white",
                  color:           model === m ? "black" : "#aaa",
                }}
              >
                {MODEL_META[m].label}
              </button>
            ))}
          </div>
        </div>

        {/* KPI row */}
        <div className="flex flex-wrap gap-4">
          <KpiCard label="Sentences"         value={`${total}`}        tag="batch size"    tagColor="#fafe45"      />
          <KpiCard label="Avg Noise Density" value={avgNoise}          tag={<span>η<sub>avg</sub></span>} tagColor={CORAL}        />
          <KpiCard label={`Avg ${meta.label} Prune`} value={`${avgPrune}%`} tag={meta.tag} tagColor={meta.tagColor} />
          <KpiCard label="Total Bytes"       value={`${totalBytes} B`} tag="raw input"     tagColor={LILAC}        />
        </div>

        {/* Noise distribution */}
        <div className="flex flex-col gap-3">
          <SubLabel>Noise Distribution Across Batch</SubLabel>
          <div className="flex flex-col gap-2">
            {noiseBuckets.map((b) => (
              <div key={b.label} className="flex items-center gap-3">
                <span className="font-['Inter',sans-serif] font-normal text-[clamp(0.6rem,0.9vw,11px)] text-[#7A7A7A] w-[120px] shrink-0">{b.label}</span>
                <div className="flex-1 h-2.5 rounded-full overflow-hidden" style={{ backgroundColor: "rgba(0,0,0,0.07)" }}>
                  <div
                    className="h-full rounded-full"
                    style={{ width: total > 0 ? `${(b.count / total) * 100}%` : "0%", backgroundColor: b.color }}
                  />
                </div>
                <span className="font-['Inter',sans-serif] font-semibold text-[clamp(0.6rem,0.9vw,11px)] text-black w-5 text-right shrink-0">{b.count}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Per-sentence mini breakdown */}
        <div className="flex flex-col gap-2">
          <SubLabel>Per-Sentence Pruning — {meta.label}</SubLabel>
          <div className="flex flex-col gap-1.5 max-h-56 overflow-y-auto">
            {rows.map((s, i) => {
              const prune = meta.pruneOf(s);
              return (
                <div key={i} className="flex items-center gap-3 min-w-0">
                  <span className="font-['Inter',sans-serif] font-normal text-[10px] text-[#aaa] w-4 text-right shrink-0">{i + 1}</span>
                  <div className="flex-1 h-2 rounded-full overflow-hidden" style={{ backgroundColor: "rgba(0,0,0,0.07)" }}>
                    <div
                      className="h-full rounded-full"
                      style={{ width: `${100 - prune}%`, backgroundColor: meta.color === "white" ? "#d0d0d0" : meta.color }}
                    />
                  </div>
                  <span className="font-['Inter',sans-serif] font-medium text-[10px] text-black w-14 text-right shrink-0 whitespace-nowrap">{prune}% prune</span>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Main Engine Page ─────────────────────────────────────────────────────

export default function Engine({
  onBatchModeChange,
  onLoadingChange,
}: {
  onBatchModeChange?: (isBatch: boolean) => void;
  onLoadingChange?: (loading: boolean) => void;
}) {
  const [inputText, setInputText] = useState(DEMO_SENTENCE.input);
  const [batchRows, setBatchRows] = useState<SentenceData[]>([DEMO_SENTENCE]);
  const [isBatch, setIsBatch] = useState(false);
  const [loading, setLoading] = useState(false);
  const [apiStatusMessage, setApiStatusMessage] = useState<string | null>(null);
  const [apiIsOnline, setApiIsOnline] = useState<boolean | null>(null);
  const [copied, setCopied] = useState(false);
  const [batchNormalized, setBatchNormalized] = useState(false);

  // Notify parent component of loading state
  useEffect(() => {
    onLoadingChange?.(loading);
  }, [loading, onLoadingChange]);

  // Pipeline state
  const [fileMeta, setFileMeta] = useState<{
    name: string;
    column: string;
    totalRows: number;
    hasReference: boolean;
  } | null>(null);
  const [batchProgress, setBatchProgress] = useState<{
    current: number;
    total: number;
  } | null>(null);

  // null = summary view, number = specific sentence index
  const [selected, setSelected] = useState<number | null>(0);
  const [showSummary, setShowSummary] = useState(false);

  const fileRef = useRef<HTMLInputElement>(null);
  const modelOutputsRef = useRef<HTMLDivElement>(null);

  // Derived values
  const byteCount  = new TextEncoder().encode(inputText).length;
  const isFallback = Boolean(batchRows[0]?.isFallback);
  const isDemo = Boolean(batchRows[0]?.isDemo);
  const outputText = isFallback ? batchRows[0]?.input ?? "" : batchRows[0]?.tahimik ?? "";
  const displayLatency = batchRows[0]?.latencyTahimik ?? "0 ms";

  // Check health on mount
  useEffect(() => {
    fetch(`${API_URL}/health`)
      .then((res) => {
        if (res.ok) {
          setApiIsOnline(true);
        } else {
          setApiIsOnline(false);
        }
      })
      .catch(() => setApiIsOnline(false));
  }, []);

  // Presentation mode always restores the complete configured scenario.
  // Set VITE_TAHIMIK_DATA_MODE=live to call the backend instead.
  const handleNormalize = useCallback(async () => {
    const text = inputText.trim();
    if (!text) return;

    if (!IS_LIVE_MODE) {
      const demo = { ...DEMO_SENTENCE };
      setInputText(demo.input);
      setBatchRows([demo]);
      setIsBatch(false);
      onBatchModeChange?.(false);
      setSelected(0);
      setShowSummary(false);
      setApiStatusMessage(null);
      return;
    }

    setLoading(true);
    setApiStatusMessage(null);

    try {
      const response = await fetch(`${API_URL}/compare`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, num_beams: 4 }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || `Comparison failed (${response.status}).`);
      const resultByModel = new Map<string, CompareModelResult>(
        (payload.results as CompareModelResult[]).map((result) => [result.model, result])
      );
      const byt5 = resultByModel.get("byt5");
      const mrt5 = resultByModel.get("mrt5");
      const tahimik = resultByModel.get("tahimik");
      if (!byt5 || !mrt5 || !tahimik) throw new Error("Comparison did not return all three models.");

      const updatedRow: SentenceData = {
        input: text,
        byt5: byt5.normalized,
        mrt5: mrt5.normalized,
        tahimik: tahimik.normalized,
        noise: Number(tahimik.telemetry?.noise_score ?? 0),
        pruning: Math.round(Number(tahimik.telemetry?.deletion_rate ?? 0) * 100),
        latencyByt5: `${Math.round(byt5.inference_time_ms)} ms`,
        latencyMrt5: `${Math.round(mrt5.inference_time_ms)} ms`,
        latencyTahimik: `${Math.round(tahimik.inference_time_ms)} ms`,
        tokens: [],
      };

      setBatchRows([updatedRow]);
      setIsBatch(false);
      onBatchModeChange?.(false);
      setSelected(0);
      setShowSummary(false);
      setApiIsOnline(true);

      // Smooth scroll back to Model Outputs
      setTimeout(() => {
        modelOutputsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 100);
    } catch (error) {
      const message = error instanceof Error ? error.message : `Could not connect to FastAPI backend at ${API_URL}.`;
      setApiStatusMessage(message);
      setBatchRows([{
        input: text,
        byt5: "Model unavailable",
        mrt5: "Model unavailable",
        tahimik: "Unprocessed input",
        noise: 0,
        pruning: 0,
        latencyByt5: "0 ms",
        latencyMrt5: "0 ms",
        latencyTahimik: "0 ms",
        tokens: [],
        isFallback: true,
      }]);
      setSelected(0);
      setShowSummary(false);
    } finally {
      setLoading(false);
    }
  }, [inputText, apiIsOnline]);

  const loadDemo = useCallback(() => {
    const demo = { ...DEMO_SENTENCE };
    setInputText(demo.input);
    setBatchRows([demo]);
    setIsBatch(false);
    onBatchModeChange?.(false);
    setSelected(0);
    setShowSummary(false);
    setApiStatusMessage(null);
  }, [onBatchModeChange]);

  // Robust Chunked Batch Normalization Pipeline
  const runBatchNormalization = useCallback(async (sentences: { input: string; reference?: string }[]) => {
    setLoading(true);
    setBatchProgress({ current: 0, total: sentences.length });

    const CHUNK_SIZE = 25;
    const initialRows: SentenceData[] = sentences.map((item) => {
      return {
        input: item.input,
        byt5: "", mrt5: "", tahimik: "", noise: 0, pruning: 0,
        latencyByt5: "—", latencyMrt5: "—", latencyTahimik: "—", tokens: [],
        reference: item.reference,
      };
    });

    const isBatchInput = sentences.length > 1;
    setBatchRows(initialRows);
    setIsBatch(isBatchInput);
    onBatchModeChange?.(isBatchInput);
    setSelected(0);
    setShowSummary(false);

    // Process in chunks of CHUNK_SIZE
    for (let i = 0; i < sentences.length; i += CHUNK_SIZE) {
      const chunk = sentences.slice(i, i + CHUNK_SIZE);
      const chunkTexts = chunk.map((s) => s.input);

      try {
        const res = await fetch(`${API_URL}/compare/batch`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ texts: chunkTexts, num_beams: 4 }),
        });

        if (res.ok) {
          const batchRes = await res.json();
          const results = batchRes.results as { results: { model: string; normalized: string; inference_time_ms: number; telemetry?: Record<string, unknown> }[] }[];

          setBatchRows((prev) => {
            const next = [...prev];
            results.forEach((comparison, idx) => {
              const targetIdx = i + idx;
              if (next[targetIdx]) {
                const model = new Map(comparison.results.map((result) => [result.model, result]));
                const byt5 = model.get("byt5"); const mrt5 = model.get("mrt5"); const tahimik = model.get("tahimik");
                if (!byt5 || !mrt5 || !tahimik) return;
                next[targetIdx] = {
                  ...next[targetIdx],
                  byt5: byt5.normalized, mrt5: mrt5.normalized, tahimik: tahimik.normalized,
                  latencyByt5: `${Math.round(byt5.inference_time_ms)} ms`,
                  latencyMrt5: `${Math.round(mrt5.inference_time_ms)} ms`,
                  latencyTahimik: `${Math.round(tahimik.inference_time_ms)} ms`,
                  noise: Number(tahimik.telemetry?.noise_score ?? 0),
                  pruning: Math.round(Number(tahimik.telemetry?.deletion_rate ?? 0) * 100),
                  tokens: [],
                };
              }
            });
            return next;
          });
          setApiIsOnline(true);
        }
      } catch {
        // Backend offline or error; fallback remains in place
      }

      setBatchProgress({ current: Math.min(i + CHUNK_SIZE, sentences.length), total: sentences.length });
    }

    setLoading(false);
    setBatchNormalized(true);

    // Smooth scroll back to Model Outputs
    setTimeout(() => {
      modelOutputsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 100);

    setTimeout(() => setBatchProgress(null), 3500);
  }, []);

  // Handle CSV / JSON / JSONL file upload
  function handleCSV(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = async (evt) => {
      const content = evt.target?.result as string;
      if (!content || !content.trim()) return;

      const { sentences, detectedColumn, format } = parseDatasetFile(content);

      if (sentences.length === 0) {
        setApiStatusMessage("No valid text sentences found in uploaded file.");
        return;
      }

      const columnDisplay =
        format === "jsonl"
          ? "JSONL 'input' field"
          : format === "json"
          ? "JSON 'input' property"
          : `"${detectedColumn}"`;

      setFileMeta({
        name: file.name,
        column: columnDisplay,
        totalRows: sentences.length,
        hasReference: sentences.some((s) => Boolean(s.reference)),
      });

      // Populate preview rows so user can inspect each row before running
      const initialRows: SentenceData[] = sentences.map((item) => {
        const gating = estimateSentenceGating(item.input);
        return {
          input: item.input,
          byt5: "",
          mrt5: "",
          tahimik: "",
          noise: gating.noise,
          pruning: gating.pruning,
          latencyByt5: "—",
          latencyMrt5: "—",
          latencyTahimik: "—",
          tokens: gating.tokens,
          reference: item.reference,
        };
      });

      setBatchRows(initialRows);
      setIsBatch(true);
      onBatchModeChange?.(true);
      setSelected(0);
      setShowSummary(false);
      setBatchNormalized(false);
    };

    reader.readAsText(file);
    e.target.value = "";
  }

  // Clear handler: resets back to empty/clean slate regardless of mode
  function clearBatch() {
    setInputText("");
    setBatchRows([]);
    setSelected(0);
    setShowSummary(false);
    setApiStatusMessage(null);
    setFileMeta(null);
    setBatchProgress(null);
    setBatchNormalized(false);
  }

  function downloadModelCSV(model: "byt5" | "mrt5" | "tahimik") {
    const header = `input,${model}\n`;
    const body = batchRows
      .map((r) => `"${r.input.replace(/"/g, '""')}","${r[model].replace(/"/g, '""')}"`)
      .join("\n");
    const blob = new Blob([header + body], { type: "text/csv" });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href     = url;
    a.download = `tahimik-${model}-outputs.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function handleRowClick(idx: number) {
    setShowSummary(false);
    setSelected((prev) => (prev === idx ? null : idx));
  }

  const handleCopyOutput = useCallback(() => {
    const textToCopy = isBatch
      ? batchRows.map((r) => r.tahimik).filter(Boolean).join("\n")
      : (outputText || batchRows[0]?.tahimik || "");

    if (!textToCopy) return;

    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(textToCopy).then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      }).catch(() => {
        fallbackCopy(textToCopy);
      });
    } else {
      fallbackCopy(textToCopy);
    }
  }, [isBatch, batchRows, outputText]);

  function fallbackCopy(text: string) {
    const ta = document.createElement("textarea");
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand("copy");
    document.body.removeChild(ta);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  const activeSentence = selected !== null ? batchRows[selected] : null;

  return (
    <div className="flex flex-col gap-8 w-full min-w-0 max-w-full overflow-hidden">

      {/* Section label & Mode Selector */}
      <div className="flex items-center gap-3 w-full min-w-0">
        <span className="font-['Inter',sans-serif] font-medium text-[clamp(0.55rem,0.85vw,11px)] text-[#bbb] uppercase tracking-widest shrink-0">
          Text Normalization
        </span>
        <span className="flex-1 h-px bg-black/10 min-w-[20px]" />

        {/* Mode Selector: Normalize Text / Normalize Batch */}
        <div className="flex items-center gap-1.5 shrink-0">
          <button
            onClick={() => {
              setIsBatch(false);
              onBatchModeChange?.(false);
            }}
            className="cursor-pointer flex items-center gap-1.5 px-3 py-1 rounded-full transition-opacity hover:opacity-80 font-['Inter',sans-serif] font-medium text-[clamp(0.55rem,0.85vw,11px)] shrink-0"
            style={{
              border: "0.5px solid rgba(0,0,0,0.35)",
              backgroundColor: !isBatch ? "#fafe45" : "white",
              color: !isBatch ? "black" : "#7A7A7A",
            }}
          >
            Normalize Text
          </button>
          <button
            onClick={() => {
              setIsBatch(true);
              onBatchModeChange?.(true);
            }}
            className="cursor-pointer flex items-center gap-1.5 px-3 py-1 rounded-full transition-opacity hover:opacity-80 font-['Inter',sans-serif] font-medium text-[clamp(0.55rem,0.85vw,11px)] shrink-0"
            style={{
              border: "0.5px solid rgba(0,0,0,0.35)",
              backgroundColor: isBatch ? "#fafe45" : "white",
              color: isBatch ? "black" : "#7A7A7A",
            }}
          >
            Normalize Batch
          </button>
        </div>
      </div>


      {/* ── NORMALIZE TEXT MODE (Side-by-side 2-column frame) ── */}
      {!isBatch ? (
        <div
          className="flex flex-col sm:flex-row items-stretch w-full min-w-0 max-w-full rounded-[24px] bg-white overflow-hidden shadow-xs"
          style={{ border: "0.5px solid rgba(0,0,0,0.5)" }}
        >
          {/* Input panel (Left side on desktop, top on mobile) */}
          <div
            className="flex flex-col gap-3 w-full sm:flex-1 min-w-0 max-w-full bg-white p-5 overflow-hidden border-b sm:border-b-0 sm:border-r border-black/20"
          >
            <div className="flex items-center justify-between gap-2 flex-wrap min-w-0">
              <Pill>RAW INPUT TEXT</Pill>
            </div>

            <textarea
              className="w-full min-w-0 bg-transparent resize-none font-['Inter',sans-serif] font-normal text-[clamp(0.9rem,1.8vw,20px)] text-black placeholder-[#d3d3d3] outline-none leading-relaxed break-words [overflow-wrap:anywhere]"
              style={{ height: "clamp(120px, 20vw, 260px)" }}
              placeholder="Type or paste informal Tagalog/Taglish social media text..."
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={(e) => {
                if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
                  handleNormalize();
                }
              }}
            />

            {/* Bottom actions */}
            <div className="flex items-center justify-between pt-2 flex-wrap gap-2 min-w-0">
              <span className="font-['Inter',sans-serif] font-normal text-[clamp(0.6rem,0.9vw,11px)] text-[#777]">
                {byteCount} UTF-8 bytes
              </span>

              <div className="flex items-center gap-2">
                <button
                  onClick={clearBatch}
                  className="cursor-pointer flex items-center justify-center px-3.5 py-1.5 rounded-full transition-opacity duration-200 hover:opacity-80 font-['Inter',sans-serif] font-medium text-[clamp(0.65rem,1.1vw,12px)] text-black"
                  style={{ border: "0.5px solid rgba(0,0,0,0.5)", backgroundColor: "#f09662" }}
                  title="Clear input and reset engine"
                >
                  Clear
                </button>

                <button
                  onClick={handleNormalize}
                  disabled={loading}
                  className="cursor-pointer flex items-center gap-2 justify-center px-4 py-1.5 rounded-full transition-opacity duration-200 hover:opacity-80 font-['Inter',sans-serif] font-medium text-[clamp(0.65rem,1.1vw,12px)] text-black disabled:opacity-50"
                  style={{ border: "0.5px solid rgba(0,0,0,0.5)", backgroundColor: "#f6c1f7" }}
                >
                  {loading ? (
                    <>
                      <svg className="animate-spin h-3.5 w-3.5 text-black" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      <span>Normalizing</span>
                    </>
                  ) : (
                    <span>Normalize</span>
                  )}
                </button>
              </div>
            </div>
          </div>

          {/* Output panel (Right side on desktop, bottom on mobile) */}
          <div className="flex flex-col gap-3 w-full sm:flex-1 min-w-0 max-w-full bg-white p-5 overflow-hidden">
            <div className="flex items-center justify-between gap-3 flex-wrap min-w-0">
              <Pill>NORMALIZED OUTPUT</Pill>
              {loading && (
                <span className="flex items-center gap-1.5 font-['Inter',sans-serif] font-medium text-[clamp(0.65rem,1.1vw,13px)] text-black/70">
                  <span className="w-1.5 h-1.5 rounded-full bg-black animate-ping" />
                  <span>Processing</span>
                </span>
              )}
            </div>

            {loading ? (
              <div className="flex flex-col gap-3 w-full py-4 animate-pulse justify-center" style={{ height: "clamp(120px, 20vw, 260px)" }}>
                <div className="h-4 bg-black/10 rounded-md w-4/5"></div>
                <div className="h-4 bg-black/10 rounded-md w-3/5"></div>
                <div className="h-4 bg-black/10 rounded-md w-2/3"></div>
              </div>
            ) : (
              <div
                className="w-full min-w-0 max-w-full font-['Inter',sans-serif] font-normal text-[clamp(0.9rem,1.8vw,20px)] text-black leading-relaxed overflow-auto break-words [overflow-wrap:anywhere]"
                style={{ height: "clamp(120px, 20vw, 260px)" }}
              >
                {outputText || <span className="text-[#ccc] italic">Normalized result will appear here...</span>}
              </div>
            )}

            {isFallback && (
              <p className="rounded-lg bg-[#fff4e8] px-3 py-2 font-['Inter',sans-serif] text-xs text-[#7a4b1e]" style={{ border: "0.5px solid rgba(122,75,30,0.35)" }}>
                Model unavailable — showing the original, unprocessed input. This is not a TAHIMIK prediction and is excluded from benchmarking.
              </p>
            )}

            {/* Output Bottom actions */}
            <div className="flex items-center justify-between pt-2 flex-wrap gap-2 min-w-0">
              <span className="font-['Inter',sans-serif] font-normal text-[clamp(0.6rem,0.9vw,11px)] text-[#999]">
                {displayLatency}
              </span>

              <button
                onClick={handleCopyOutput}
                className="cursor-pointer flex items-center gap-1.5 px-3.5 py-1.5 rounded-full transition-all duration-200 hover:opacity-80 font-['Inter',sans-serif] font-medium text-[clamp(0.65rem,1.1vw,12px)] text-black"
                style={{ border: "0.5px solid rgba(0,0,0,0.5)", backgroundColor: copied ? "#86E992" : "white" }}
                title="Copy normalized output text"
              >
                {copied ? (
                  <>
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                      <polyline points="20 6 9 17 4 12" />
                    </svg>
                    Copied!
                  </>
                ) : (
                  <>
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                    </svg>
                    Copy Output
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      ) : (
        /* ── NORMALIZE BATCH MODE (One unified section with 2-column table) ── */
        <div
          className="flex flex-col w-full min-w-0 max-w-full rounded-[24px] bg-white overflow-hidden shadow-xs"
          style={{ border: "0.5px solid rgba(0,0,0,0.5)" }}
        >
          {/* Header Bar */}
          <div className="flex items-center justify-between px-5 pt-4 pb-3 flex-wrap gap-2 min-w-0" style={{ borderBottom: "0.5px solid rgba(0,0,0,0.1)" }}>
            <div className="flex items-center gap-2.5 flex-wrap min-w-0">
              <Pill>{fileMeta ? `DATASET PREVIEW — ${batchRows.length} SENTENCES` : "UPLOAD BATCH DATASET"}</Pill>
              {fileMeta && (
                <span className="font-['Inter',sans-serif] font-normal text-xs text-[#555] bg-black/5 px-2.5 py-0.5 rounded-full truncate max-w-xs">
                  {fileMeta.name}
                </span>
              )}
            </div>
          </div>

          {/* Hidden File Input */}
          <input
            ref={fileRef}
            type="file"
            accept=".csv,.tsv,.txt,.json,.jsonl"
            className="hidden"
            onChange={handleCSV}
          />

          {/* Content: Upload Dropzone when empty, or 2-Column Table when dataset loaded */}
          {!fileMeta && batchRows.length <= 1 ? (
            /* Upload Dropzone */
            <div
              onClick={() => fileRef.current?.click()}
              className="cursor-pointer flex flex-col items-center justify-center gap-3 p-10 min-h-[220px] bg-black/[0.01] hover:bg-black/[0.03] transition-colors text-center m-4 rounded-[16px] border border-dashed border-black/25"
            >
              <div className="w-12 h-12 rounded-full bg-black/5 flex items-center justify-center text-black/70">
                <svg width="22" height="22" viewBox="0 0 12 12" fill="none">
                  <path d="M6 1v7M3 5l3-3 3 3M1 10h10" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </div>
              <div className="flex flex-col gap-1 max-w-md">
                <span className="font-['Inter',sans-serif] font-semibold text-sm text-black">
                  Upload Dataset File (CSV, TSV, JSON, JSONL)
                </span>
                <span className="font-['Inter',sans-serif] font-normal text-xs text-[#777]">
                  File should include informal/noisy sentences and optional clean reference sentences.
                </span>
              </div>
              <button
                type="button"
                className="mt-1 flex items-center gap-1.5 px-4 py-1.5 rounded-full font-['Inter',sans-serif] font-medium text-xs text-black transition-opacity hover:opacity-80"
                style={{ border: "0.5px solid rgba(0,0,0,0.5)", backgroundColor: "#f6c1f7" }}
              >
                Choose Dataset File
              </button>
            </div>
          ) : (
            /* 2-Column Table: Noisy Sentence & Clean Reference */
            <div className="flex flex-col w-full min-w-0">
              {/* Table Column Headers */}
              <div
                className="grid grid-cols-[44px_minmax(0,1fr)_minmax(0,1fr)] bg-[#f8f8f8] px-4 py-2.5 font-['Inter',sans-serif] font-medium text-[clamp(0.6rem,0.8vw,11px)] text-[#888] uppercase tracking-wider"
                style={{ borderBottom: "0.5px solid rgba(0,0,0,0.1)" }}
              >
                <span className="text-center">#</span>
                <span className="pr-4 border-r border-black/10">Noisy Sentence (Input)</span>
                <span className="pl-4">Clean Reference Sentence (Target)</span>
              </div>

              {/* Scrollable rows */}
              <div
                className="w-full min-w-0 max-w-full overflow-y-auto overflow-x-hidden divide-y divide-black/5"
                style={{ height: "clamp(220px, 30vw, 380px)" }}
              >
                {batchRows.map((row, i) => (
                  <div key={i} className="grid grid-cols-[44px_minmax(0,1fr)_minmax(0,1fr)] px-4 py-3 min-w-0 items-baseline hover:bg-black/[0.015] transition-colors">
                    <span className="font-['Inter',sans-serif] font-normal text-[11px] text-[#bbb] text-center shrink-0">
                      {i + 1}
                    </span>
                    <p className="font-['Inter',sans-serif] font-normal text-[clamp(0.8rem,1.2vw,14px)] text-black leading-relaxed break-words [overflow-wrap:anywhere] min-w-0 pr-4 border-r border-black/10">
                      {row.input}
                    </p>
                    <p className="font-['Inter',sans-serif] font-normal text-[clamp(0.8rem,1.2vw,14px)] text-[#444] leading-relaxed break-words [overflow-wrap:anywhere] min-w-0 pl-4">
                      {row.reference || <span className="text-[#bbb] italic text-xs">(No reference provided)</span>}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Bottom actions: File / Processing Status, Clear, Upload Different File, Run Normalization */}
          <div className="flex items-center justify-between px-5 py-3.5 flex-wrap gap-3 min-w-0 bg-[#fafafa]" style={{ borderTop: "0.5px solid rgba(0,0,0,0.1)" }}>
            {batchProgress ? (
              /* Visual Animated Progress Bar */
              <div className="flex items-center gap-3 min-w-0">
                <div className="w-36 sm:w-56 h-2.5 rounded-full bg-black/10 overflow-hidden relative p-0.5">
                  <div
                    className="h-full bg-[#f6c1f7] rounded-full transition-all duration-300"
                    style={{
                      width: `${Math.round((batchProgress.current / batchProgress.total) * 100)}%`,
                      border: "0.5px solid rgba(0,0,0,0.3)",
                    }}
                  />
                </div>
                <div className="flex items-center gap-2">
                  <span className="font-['Inter',sans-serif] font-semibold text-xs text-black">
                    {Math.round((batchProgress.current / batchProgress.total) * 100)}%
                  </span>
                  <span className="font-['Inter',sans-serif] font-normal text-[11px] text-[#888]">
                    ({batchProgress.current}/{batchProgress.total})
                  </span>
                </div>
              </div>
            ) : (
              <span className="font-['Inter',sans-serif] font-normal text-[clamp(0.65rem,0.9vw,12px)] text-[#666]">
                {fileMeta || batchRows.length > 1
                  ? `${batchRows.length} sentences loaded • Ready to normalize`
                  : "No dataset loaded"}
              </span>
            )}

            <div className="flex items-center gap-2">
              {(fileMeta || batchRows.length > 1) && (
                <>
                  <button
                    onClick={clearBatch}
                    disabled={loading}
                    className="cursor-pointer flex items-center justify-center px-3.5 py-1.5 rounded-full transition-opacity duration-200 hover:opacity-80 font-['Inter',sans-serif] font-medium text-[clamp(0.65rem,1.1vw,12px)] text-black disabled:opacity-40"
                    style={{ border: "0.5px solid rgba(0,0,0,0.5)", backgroundColor: "#f09662" }}
                    title="Clear uploaded dataset"
                  >
                    Clear
                  </button>

                  <button
                    onClick={() => fileRef.current?.click()}
                    disabled={loading}
                    className="cursor-pointer flex items-center gap-1.5 px-3.5 py-1.5 rounded-full transition-opacity duration-200 hover:opacity-80 font-['Inter',sans-serif] font-medium text-[clamp(0.65rem,1.1vw,12px)] text-black disabled:opacity-40"
                    style={{ border: "0.5px solid rgba(0,0,0,0.5)", backgroundColor: "white" }}
                  >
                    <svg width="11" height="11" viewBox="0 0 12 12" fill="none" className="shrink-0">
                      <path d="M6 1v7M3 5l3-3 3 3M1 10h10" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                    Upload Different File
                  </button>

                  <button
                    onClick={() => runBatchNormalization(batchRows.map((r) => ({ input: r.input, reference: r.reference })))}
                    disabled={loading}
                    className="cursor-pointer flex items-center gap-2 px-5 py-1.5 rounded-full transition-opacity duration-200 hover:opacity-80 font-['Inter',sans-serif] font-medium text-[clamp(0.7rem,1.1vw,13px)] text-black disabled:opacity-50"
                    style={{ border: "0.5px solid rgba(0,0,0,0.5)", backgroundColor: "#f6c1f7" }}
                  >
                    {loading ? (
                      <>
                        <svg className="animate-spin h-3.5 w-3.5 text-black" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        <span>Normalizing</span>
                      </>
                    ) : (
                      <>
                        <svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor" className="shrink-0">
                          <polygon points="5 3 19 12 5 21 5 3" />
                        </svg>
                        <span>Run Normalization</span>
                      </>
                    )}
                  </button>
                </>
              )}

            </div>
          </div>
        </div>
      )}

      {/* Model Outputs Table */}
      <div
        ref={modelOutputsRef}
        className="flex flex-col gap-3 w-full min-w-0 max-w-full scroll-mt-[80px]"
      >
        <div className="flex items-center gap-3 flex-wrap min-w-0">
          <span className="font-['Inter',sans-serif] font-medium text-[clamp(0.55rem,0.85vw,11px)] text-[#bbb] uppercase tracking-widest shrink-0">
            Model Outputs
          </span>
          {loading && (
            <span className="flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-black/5 text-black font-medium text-[10px]">
              <span className="w-1.5 h-1.5 rounded-full bg-[#f6c1f7] animate-ping" />
              Processing Inference...
            </span>
          )}
          <span className="flex-1 h-px bg-black/10 min-w-[20px]" />
          {(["byt5", "mrt5", "tahimik"] as const).map((m) => (
            <button
              key={m}
              onClick={() => downloadModelCSV(m)}
              disabled={loading}
              className="cursor-pointer flex items-center gap-1.5 px-3 py-1 rounded-full bg-white transition-opacity hover:opacity-70 font-['Inter',sans-serif] font-medium text-[clamp(0.55rem,0.85vw,11px)] text-[#7A7A7A] shrink-0 disabled:opacity-40"
              style={{
                border:          "0.5px solid rgba(0,0,0,0.35)",
                backgroundColor: m === "tahimik" ? "#fafe45" : m === "mrt5" ? CORAL : "white",
                color:           m === "byt5" ? "#7A7A7A" : "black",
              }}
            >
              <svg width="11" height="11" viewBox="0 0 12 12" fill="none" className="shrink-0">
                <path d="M6 1v7M3 8l3 3 3-3M1 11h10" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              {m === "byt5" ? "BYT5" : m === "mrt5" ? "MrT5" : "TAHIMIK"}
            </button>
          ))}
        </div>

        {/* Horizontal scroll wrapper with bounded columns */}
        <div className="w-full max-w-full overflow-x-auto rounded-[24px] bg-white shadow-xs" style={{ border: "0.5px solid rgba(0,0,0,0.5)" }}>
          <div className="w-full" style={{ minWidth: 460 }}>

            <div className="grid grid-cols-[minmax(0,1fr)_minmax(0,1fr)] bg-[#fafafa] w-full" style={{ borderBottom: "0.5px solid rgba(0,0,0,0.1)" }}>
              <div className="px-5 py-3 min-w-0" style={{ borderRight: "0.5px solid rgba(0,0,0,0.1)" }}>
                <span className="font-['Inter',sans-serif] font-medium text-[clamp(0.55rem,0.8vw,11px)] text-[#888] uppercase tracking-wider">Input Text</span>
              </div>
              <div className="px-5 py-3 min-w-0">
                <span className="font-['Inter',sans-serif] font-medium text-[clamp(0.55rem,0.8vw,11px)] text-[#888] uppercase tracking-wider">Model Outputs (ByT5 • MrT5 • TAHIMIK)</span>
              </div>
            </div>

            <div className="overflow-y-auto w-full divide-y divide-black/5" style={{ maxHeight: "clamp(320px, 55vw, 520px)" }}>
              {/* If single text mode is actively loading */}
              {loading && !isBatch ? (
                <div className="grid grid-cols-[minmax(0,1fr)_minmax(0,1fr)] w-full">
                  <div className="flex gap-3 px-5 py-4 h-full min-w-0 overflow-hidden bg-black/[0.01]" style={{ borderRight: "0.5px solid rgba(0,0,0,0.1)" }}>
                    <span className="shrink-0 font-['Inter',sans-serif] font-normal text-[10px] text-[#bbb] pt-[3px] w-6 text-right select-none">1</span>
                    <p className="font-['Inter',sans-serif] font-normal text-[clamp(0.78rem,1.3vw,15px)] text-black leading-relaxed break-words [overflow-wrap:anywhere] min-w-0 flex-1">
                      {inputText}
                    </p>
                  </div>
                  <div className="flex flex-col min-w-0 overflow-hidden">
                    <div className="flex flex-col gap-1.5 px-5 py-3 min-w-0 bg-black/[0.01]">
                      <span className="font-['Inter',sans-serif] font-normal text-[10px] text-[#bbb] uppercase tracking-wide">BYT5</span>
                      <div className="flex items-center gap-2">
                        <span className="w-1.5 h-1.5 rounded-full bg-black/40 animate-ping" />
                        <span className="font-['Inter',sans-serif] text-xs text-[#888] italic animate-pulse">Computing ByT5 output...</span>
                      </div>
                    </div>
                    <div className="flex flex-col gap-1.5 px-5 py-3 min-w-0 border-t border-black/5 bg-black/[0.01]">
                      <span className="font-['Inter',sans-serif] font-normal text-[10px] text-[#bbb] uppercase tracking-wide">MrT5</span>
                      <div className="flex items-center gap-2">
                        <span className="w-1.5 h-1.5 rounded-full bg-[#f0a080] animate-ping" />
                        <span className="font-['Inter',sans-serif] text-xs text-[#888] italic animate-pulse">Applying 50% fixed byte deletion...</span>
                      </div>
                    </div>
                    <div className="flex flex-col gap-1.5 px-5 py-3 min-w-0 border-t border-black/5 bg-[#fafe45]/10">
                      <span className="self-start inline-flex items-center px-1.5 py-px rounded-full font-['Inter',sans-serif] font-semibold text-[9px] text-black uppercase tracking-wide" style={{ border: "0.5px solid rgba(0,0,0,0.5)", backgroundColor: "#fafe45" }}>TAHIMIK</span>
                      <div className="flex items-center gap-2">
                        <span className="w-1.5 h-1.5 rounded-full bg-[#fafe45] animate-ping" />
                        <span className="font-['Inter',sans-serif] text-xs text-black/80 font-medium italic animate-pulse">Estimating noise & executing adaptive gate...</span>
                      </div>
                    </div>
                  </div>
                </div>
              ) : batchRows.length === 0 ? (
                <div className="px-5 py-8 text-center">
                  <p className="font-['Inter',sans-serif] text-sm text-[#666]">
                    No model outputs yet.
                  </p>
                  <p className="mt-1 font-['Inter',sans-serif] text-xs text-[#999]">
                    Enter text above and select Normalize, or upload a batch dataset to compare ByT5, MrT5, and TAHIMIK.
                  </p>
                </div>
              ) : (
                batchRows.map((row, i) => {
                  const isActive = selected === i;
                  const hasOutput = Boolean(row.tahimik);
                  const isItemLoading = loading && !hasOutput;

                  return (
                    <div
                      key={i}
                      className="grid grid-cols-[minmax(0,1fr)_minmax(0,1fr)] cursor-pointer transition-colors w-full"
                      style={{
                        backgroundColor: isActive ? "rgba(221,239,117,0.12)" : "transparent",
                      }}
                      onClick={() => handleRowClick(i)}
                    >
                      <div className="flex gap-3 px-5 py-4 h-full min-w-0 overflow-hidden" style={{ borderRight: "0.5px solid rgba(0,0,0,0.1)" }}>
                        <span className="shrink-0 font-['Inter',sans-serif] font-normal text-[10px] text-[#d0d0d0] pt-[3px] w-6 text-right select-none">{i + 1}</span>
                        <p className="font-['Inter',sans-serif] font-normal text-[clamp(0.78rem,1.3vw,15px)] text-black leading-relaxed break-words [overflow-wrap:anywhere] min-w-0 flex-1">
                          {row.input}
                        </p>
                        {isActive && (
                          <span className="shrink-0 self-start mt-1 text-[10px] font-['Inter',sans-serif] font-medium text-[#7A7A7A]">● inspecting</span>
                        )}
                      </div>
                      <div className="flex flex-col min-w-0 overflow-hidden">
                        {isItemLoading ? (
                          <div className="flex flex-col gap-2 p-4 animate-pulse">
                            <div className="flex items-center gap-2">
                              <span className="w-1.5 h-1.5 rounded-full bg-[#f6c1f7] animate-ping" />
                              <span className="font-['Inter',sans-serif] text-xs text-[#888] italic">Queued in batch pipeline...</span>
                            </div>
                            <div className="h-3 bg-black/5 rounded w-3/4" />
                            <div className="h-3 bg-black/5 rounded w-1/2" />
                          </div>
                        ) : (
                          [
                            { name: "BYT5",    text: row.byt5,    highlight: false },
                            { name: "MrT5",    text: row.mrt5,    highlight: false },
                            { name: "TAHIMIK", text: row.tahimik, highlight: true  },
                          ].map((m, mi) => (
                            <div
                              key={m.name}
                              className="flex flex-col gap-0.5 px-5 py-3 min-w-0 overflow-hidden"
                              style={{
                                borderTop:       mi > 0 ? "0.5px solid rgba(0,0,0,0.08)" : "none",
                                backgroundColor: m.highlight ? "rgba(250,254,69,0.06)" : "transparent",
                              }}
                            >
                              {m.highlight ? (
                                <span
                                  className="self-start inline-flex items-center px-1.5 py-px rounded-full font-['Inter',sans-serif] font-semibold text-[9px] text-black uppercase tracking-wide"
                                  style={{ border: "0.5px solid rgba(0,0,0,0.5)", backgroundColor: "#fafe45" }}
                                >
                                  {m.name}
                                </span>
                              ) : (
                                <span className="font-['Inter',sans-serif] font-normal text-[10px] text-[#c0c0c0] uppercase tracking-wide">{m.name}</span>
                              )}
                              <p className="font-['Inter',sans-serif] font-normal text-[clamp(0.78rem,1.3vw,15px)] text-black leading-relaxed break-words [overflow-wrap:anywhere] min-w-0">
                                {m.text || <span className="text-[#ccc] italic text-xs">Waiting for model normalization...</span>}
                              </p>
                            </div>
                          ))
                        )}
                      </div>
                    </div>
                  );
                })
              )}
            </div>

          </div>
        </div>
      </div>

      {/* Inspector / Summary */}
      <div className="flex flex-col gap-3 w-full min-w-0">
        <div className="flex items-center gap-3">
          <span className="font-['Inter',sans-serif] font-medium text-[clamp(0.55rem,0.85vw,11px)] text-[#bbb] uppercase tracking-widest shrink-0">
            Compression & Gating Information
          </span>
          <span className="flex-1 h-px bg-black/10 min-w-[20px]" />

          {/* Summary toggle */}
          <button
            onClick={() => { setShowSummary((s) => !s); if (!showSummary) setSelected(null); }}
            className="shrink-0 cursor-pointer flex items-center gap-1.5 px-3 py-1 rounded-full transition-opacity hover:opacity-70 font-['Inter',sans-serif] font-medium text-[clamp(0.55rem,0.85vw,11px)] text-black"
            style={{
              border:          "0.5px solid rgba(0,0,0,0.5)",
              backgroundColor: showSummary ? LIME : "white",
            }}
          >
            {showSummary ? "✕ Close Summary" : "Batch Summary"}
          </button>
        </div>

        {!showSummary && (
          <p className="font-['Inter',sans-serif] font-normal text-[clamp(0.58rem,0.85vw,11px)] text-[#bbb]">
            {activeSentence
              ? `Showing details for sentence ${(selected ?? 0) + 1} — click row again to toggle`
              : "Click any sentence row above to inspect its gating details"}
          </p>
        )}

        {showSummary ? (
          <BatchSummary rows={batchRows} isLoading={loading} />
        ) : activeSentence ? (
          <GatingInspector sentence={activeSentence} isLoading={loading} />
        ) : (
          <div
            className="w-full rounded-[24px] bg-white p-8 flex flex-col items-center justify-center gap-3"
            style={{ border: "0.5px solid rgba(0,0,0,0.15)", minHeight: 160 }}
          >
            <span className="font-['Inter',sans-serif] font-normal text-[clamp(0.8rem,1.4vw,16px)] text-[#777]">
              Compression and gating details will appear here
            </span>
            <span className="max-w-xl text-center font-['Inter',sans-serif] font-normal text-[clamp(0.65rem,1vw,13px)] text-[#999]">
              Run a comparison first. This panel then shows the selected sentence's noise estimate, deletion rate, retained bytes, and output from each model. Batch Summary aggregates those values across uploaded sentences.
            </span>
          </div>
        )}
      </div>

    </div>
  );
}
