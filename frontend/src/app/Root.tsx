import { useState, useEffect, useRef } from "react";
import { Card, GROUP_MEMBERS } from "./shared";
import systemArchImg from "@/assets/system_architecture.png";
import Engine from "./pages/Engine";
import Benchmark from "./pages/Benchmark";
import Efficiency from "./pages/Efficiency";
import Validation from "./pages/Validation";

// ── Section definitions ──────────────────────────────────────────────────

const SECTIONS = [
  { id: "engine",     label: "Text Normalization",    accordion: false },
  { id: "benchmark",  label: "Accuracy Evaluation",   accordion: true  },
  { id: "efficiency", label: "Runtime Efficiency",    accordion: true  },
  { id: "validation", label: "Statistical Validation", accordion: true },
  { id: "about",      label: "About",                 accordion: false },
];

const ACCORDION_IDS = new Set(SECTIONS.filter((s) => s.accordion).map((s) => s.id));

// ── Shared components ────────────────────────────────────────────────────

function TahimikTab() {
  return (
    <div className="flex flex-col gap-6 w-full">
      {/* Basic info card */}
      <Card>
        <p className="font-['Inter',sans-serif] font-light text-[clamp(0.82rem,1.05vw,14px)] text-black/90 text-justify leading-relaxed">
          <span className="font-bold text-black">TAHIMIK</span>
          {" (Text Augmentation and Harmonization of Informal and Multilingual Input for Knowledge Extraction) is a text normalization system built on "}
          <span className="font-bold text-black">ByT5 (Byte-level T5)</span>
          {" and "}
          <span className="font-bold text-black">MrT5</span>
          {". By operating directly on raw UTF-8 bytes rather than subword vocabularies, TAHIMIK eliminates out-of-vocabulary failures on noisy Tagalog/Taglish informal spelling, abbreviations, character elongations, and slang. Its core contribution is an adaptive byte-level gating mechanism that conditions pruning thresholds on estimated sentence noise density ("}
          <span className="font-math font-semibold text-black">η<sub>i</sub></span>
          {") to recover high downstream accuracy while maintaining runtime efficiency."}
        </p>
      </Card>

      {/* Technical Information Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5 w-full">
        {/* 1. Two-Stage Training Curriculum */}
        <div className="rounded-[24px] bg-white p-6 sm:p-7 flex flex-col gap-4" style={{ border: "0.5px solid rgba(0,0,0,0.5)" }}>
          <div className="flex items-center justify-between gap-2">
            <span className="font-['Inter',sans-serif] font-bold text-[clamp(0.95rem,1.4vw,18px)] text-black">
              Two-Stage Training Curriculum
            </span>
            <span className="px-2.5 py-0.5 rounded-full text-[10px] font-semibold tracking-wider uppercase bg-[#DDEF75] text-black border border-black/30">
              Training Pipeline
            </span>
          </div>

          <div className="flex flex-col gap-3 text-justify">
            <div className="flex flex-col gap-1.5">
              <span className="font-['Inter',sans-serif] font-semibold text-[13px] text-black">
                Stage 1 — Denoising Pretraining (Synthetic Noise Curriculum)
              </span>
              <p className="font-['Inter',sans-serif] font-light text-[clamp(0.75rem,1vw,13.5px)] text-black/85 leading-relaxed">
                Trained on ~1,000,000 synthetic noisy-to-clean sentence pairs (90/10 train/val split) derived from clean Tagalog corpora using 9 controlled netspeak noise categories from the paper: abbreviations & shortenings (slmt → salamat), orthographic variation (pede → puwede), character elongation (soooobra → sobra), punctuation variation, capitalization variation, slang & netspeak (d2 → dito), Taglish morphology patterns (sanba → saan ba), emoji sentiment markers, and code-switching.
              </p>
            </div>

            <div className="flex flex-col gap-1.5 pt-1">
              <span className="font-['Inter',sans-serif] font-semibold text-[13px] text-black">
                Stage 2 — Supervised Fine-Tuning (Gold Pair Alignment)
              </span>
              <p className="font-['Inter',sans-serif] font-light text-[clamp(0.75rem,1vw,13.5px)] text-black/85 leading-relaxed">
                The second stage is designed for gold-standard noisy-to-clean pairs using an 80% train / 10% validation / 10% test split. Dataset collection, annotation reliability, and final experimental results are reported only after they have been completed and validated.
              </p>
            </div>
          </div>
        </div>

        {/* 2. Adaptive Byte-Level Gating */}
        <div className="rounded-[24px] bg-white p-6 sm:p-7 flex flex-col gap-4" style={{ border: "0.5px solid rgba(0,0,0,0.5)" }}>
          <div className="flex items-center justify-between gap-2">
            <span className="font-['Inter',sans-serif] font-bold text-[clamp(0.95rem,1.4vw,18px)] text-black">
              Noise-Adaptive Gating Mechanism
            </span>
            <span className="px-2.5 py-0.5 rounded-full text-[10px] font-semibold tracking-wider uppercase bg-[#FAFE45] text-black border border-black/30">
              Core Innovation
            </span>
          </div>

          <div className="flex flex-col gap-3 text-justify">
            <p className="font-['Inter',sans-serif] font-light text-[clamp(0.75rem,1vw,13.5px)] text-black/85 leading-relaxed">
              Unlike rigid pruning approaches (such as MrT5 with fixed 50% byte deletion) that degrade accuracy on informal text, TAHIMIK dynamically scales deletion pressure based on the estimated sentence noise density:
            </p>

            <div className="py-2.5 flex items-center justify-center text-center">
              <span className="font-math font-semibold text-[clamp(1.05rem,1.5vw,22px)] text-black tracking-wide">
                d<sub>target</sub>(x<sub>i</sub>) = d<sub>max</sub> × (1 − η<sub>i</sub>)
              </span>
            </div>

            <p className="font-['Inter',sans-serif] font-light text-[clamp(0.75rem,1vw,13.5px)] text-black/85 leading-relaxed">
              Where <span className="font-math">η<sub>i</sub> ∈ [0, 1]</span> is predicted by an early Noise Estimator MLP (encoder layer 2) supervised on normalized byte Levenshtein distance (<span className="font-math">n*</span>). The joint optimization objective balances normalization quality and dynamic compression:
            </p>

            <div className="py-1 flex items-center justify-center text-center">
              <span className="font-math font-semibold text-[clamp(0.85rem,1.15vw,16px)] text-black tracking-wide">
                <span className="italic">L</span> = <span className="italic">L</span><sub>CE</sub> + <span className="italic">w</span><sub>rate</sub>·<span className="italic">L</span><sub>rate</sub> + <span className="italic">w</span><sub>attn</sub>·<span className="italic">L</span><sub>attn</sub> + <span className="italic">L</span><sub>NE</sub>
              </span>
            </div>
          </div>
        </div>

        {/* 3. Model Architecture & Byte Tokenization */}
        <div className="rounded-[24px] bg-white p-6 sm:p-7 flex flex-col gap-4" style={{ border: "0.5px solid rgba(0,0,0,0.5)" }}>
          <div className="flex items-center justify-between gap-2">
            <span className="font-['Inter',sans-serif] font-bold text-[clamp(0.95rem,1.4vw,18px)] text-black">
              Tokenizer-Free Byte Architecture
            </span>
            <span className="px-2.5 py-0.5 rounded-full text-[10px] font-semibold tracking-wider uppercase bg-[#F0A080] text-black border border-black/30">
              Backbone
            </span>
          </div>

          <p className="font-['Inter',sans-serif] font-light text-[clamp(0.75rem,1vw,13.5px)] text-black/85 text-justify leading-relaxed">
            Standard subword tokenizers (WordPiece, SentencePiece, BPE) split informal slang and misspelled Filipino words into long sequences of uninformative subword fragments. TAHIMIK processes raw UTF-8 byte streams with a compact vocabulary of <span className="font-math">V = 256 + 3</span> special tokens (PAD, EOS, UNK). This guarantees 100% vocabulary coverage, total resilience against novel slang or character repetition, and zero out-of-vocabulary (OOV) unknown token errors.
          </p>
        </div>

        {/* 4. Statistical Validation Protocol */}
        <div className="rounded-[24px] bg-white p-6 sm:p-7 flex flex-col gap-4" style={{ border: "0.5px solid rgba(0,0,0,0.5)" }}>
          <div className="flex items-center justify-between gap-2">
            <span className="font-['Inter',sans-serif] font-bold text-[clamp(0.95rem,1.4vw,18px)] text-black">
              Evaluation & Statistical Protocol
            </span>
            <span className="px-2.5 py-0.5 rounded-full text-[10px] font-semibold tracking-wider uppercase bg-[#D4A7E0] text-black border border-black/30">
              Methodology
            </span>
          </div>

          <p className="font-['Inter',sans-serif] font-light text-[clamp(0.75rem,1vw,13.5px)] text-black/85 text-justify leading-relaxed">
            The evaluation protocol uses GLEU+ (source-aware), chrF (character 6-gram overlap), Error Reduction Rate (ERR), and Word Accuracy (WA). Planned significance testing uses 1,000 paired-bootstrap resamples for accuracy and latency, Wilcoxon signed-rank testing for GPU memory, and Holm–Bonferroni correction. Values are displayed only from validated experiment outputs.
          </p>
        </div>
      </div>
    </div>
  );
}

function ArchitectureDiagram() {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [inlineScale, setInlineScale] = useState(1);
  const [modalScale, setModalScale] = useState(1);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setIsModalOpen(false);
        setModalScale(1);
        setPosition({ x: 0, y: 0 });
      }
    };
    if (isModalOpen) {
      window.addEventListener("keydown", handleKeyDown);
      document.body.style.overflow = "hidden";
    }
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      document.body.style.overflow = "";
    };
  }, [isModalOpen]);

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY < 0 ? 0.25 : -0.25;
    setModalScale((prev) => Math.min(Math.max(0.6, prev + delta), 5));
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX - position.x, y: e.clientY - position.y });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging) {
      setPosition({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y,
      });
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const openModal = () => {
    setModalScale(1);
    setPosition({ x: 0, y: 0 });
    setIsModalOpen(true);
  };

  return (
    <>
      <div className="flex flex-col gap-3 w-full pt-4 border-t border-black/10">
        <div className="flex items-center justify-between gap-3 flex-wrap">
          <div className="flex items-center gap-2">
            <span className="px-2.5 py-0.5 rounded-full text-[10px] font-semibold uppercase bg-[#DDEF75] text-black border border-black/30">
              Diagram
            </span>
          </div>

          <div className="flex items-center gap-1.5">
            <button
              onClick={() => setInlineScale((s) => Math.max(0.7, s - 0.2))}
              className="cursor-pointer px-2.5 py-1 rounded-full border border-black/20 bg-white text-xs font-semibold text-black hover:bg-black/5"
              title="Zoom out"
            >
              −
            </button>
            <span className="text-xs text-black/60 font-mono w-12 text-center select-none">
              {Math.round(inlineScale * 100)}%
            </span>
            <button
              onClick={() => setInlineScale((s) => Math.min(3, s + 0.2))}
              className="cursor-pointer px-2.5 py-1 rounded-full border border-black/20 bg-white text-xs font-semibold text-black hover:bg-black/5"
              title="Zoom in"
            >
              +
            </button>
            <button
              onClick={() => setInlineScale(1)}
              className="cursor-pointer px-2.5 py-1 rounded-full border border-black/20 bg-white text-xs font-semibold text-black hover:bg-black/5"
              title="Reset zoom"
            >
              Reset
            </button>
            <button
              onClick={openModal}
              className="cursor-pointer flex items-center gap-1 px-3 py-1 rounded-full bg-black text-white text-xs font-medium hover:bg-black/85 ml-1 transition-colors"
            >
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
              </svg>
              Fullscreen
            </button>
          </div>
        </div>

        <div
          onClick={openModal}
          className="relative w-full rounded-[18px] bg-[#fafafa] border border-black/15 p-4 overflow-hidden cursor-zoom-in group transition-all hover:border-black/35"
        >
          <div className="w-full flex items-center justify-center overflow-auto max-h-[520px]">
            <img
              src={systemArchImg}
              alt="TAHIMIK Architecture Diagram: Pre-training, Fine-tuning, Forward Pass, and Decoder"
              className="object-contain transition-transform duration-200"
              style={{
                transform: `scale(${inlineScale})`,
                transformOrigin: "center top",
                maxWidth: inlineScale > 1 ? "none" : "100%",
              }}
            />
          </div>

          <div className="absolute bottom-3 right-3 pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity bg-black/80 backdrop-blur-sm text-white text-[11px] font-medium px-3 py-1.5 rounded-full flex items-center gap-1.5 shadow-lg">
            <span>Click to open zoom & pan viewer</span>
          </div>
        </div>
      </div>

      {/* Lightbox Modal */}
      {isModalOpen && (
        <div
          className="fixed inset-0 z-50 bg-black/90 backdrop-blur-md flex flex-col justify-between"
          onClick={(e) => {
            if (e.target === e.currentTarget) {
              setIsModalOpen(false);
            }
          }}
        >
          {/* Top Bar */}
          <div className="flex items-center justify-between px-6 py-4 bg-black/60 border-b border-white/10 z-10">
            <div className="flex items-center gap-3">
              <span className="font-['Inter',sans-serif] font-bold text-white text-[clamp(0.9rem,1.2vw,16px)]">
                TAHIMIK System Architecture
              </span>
              <span className="text-white/50 text-xs hidden sm:inline">
                Pre-training, Fine-tuning, Forward Pass & Decoder Modules
              </span>
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={() => setModalScale((s) => Math.max(0.5, s - 0.25))}
                className="cursor-pointer w-8 h-8 rounded-full bg-white/10 hover:bg-white/20 text-white font-bold text-base flex items-center justify-center transition-colors"
                title="Zoom Out"
              >
                −
              </button>
              <span className="text-white font-mono text-xs w-14 text-center select-none">
                {Math.round(modalScale * 100)}%
              </span>
              <button
                onClick={() => setModalScale((s) => Math.min(5, s + 0.25))}
                className="cursor-pointer w-8 h-8 rounded-full bg-white/10 hover:bg-white/20 text-white font-bold text-base flex items-center justify-center transition-colors"
                title="Zoom In"
              >
                +
              </button>
              <button
                onClick={() => {
                  setModalScale(1);
                  setPosition({ x: 0, y: 0 });
                }}
                className="cursor-pointer px-3 py-1 rounded-full bg-white/10 hover:bg-white/20 text-white text-xs font-medium transition-colors ml-1"
                title="Reset Zoom & Pan"
              >
                Reset
              </button>
              <button
                onClick={() => setIsModalOpen(false)}
                className="cursor-pointer w-8 h-8 rounded-full bg-white/20 hover:bg-white/30 text-white font-bold text-sm flex items-center justify-center transition-colors ml-3"
                title="Close (Esc)"
              >
                ✕
              </button>
            </div>
          </div>

          {/* Canvas Viewport */}
          <div
            className="flex-1 w-full overflow-hidden flex items-center justify-center relative select-none"
            style={{ cursor: isDragging ? "grabbing" : modalScale > 1 ? "grab" : "default" }}
            onWheel={handleWheel}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
          >
            <img
              src={systemArchImg}
              alt="TAHIMIK Architecture Diagram Fullscreen"
              className="max-w-none transition-transform duration-75 select-none"
              draggable={false}
              style={{
                transform: `translate(${position.x}px, ${position.y}px) scale(${modalScale})`,
                maxHeight: "85vh",
              }}
            />
          </div>

          {/* Bottom Bar Hints */}
          <div className="px-6 py-3 bg-black/60 border-t border-white/10 flex items-center justify-between text-white/60 text-xs">
            <span className="hidden sm:inline">
              Scroll mouse wheel to zoom • Click and drag to pan • Click Reset to re-center
            </span>
            <span className="ml-auto font-mono text-[11px] text-white/40">
              Press [ESC] to close
            </span>
          </div>
        </div>
      )}
    </>
  );
}

function SystemTab() {
  return (
    <Card>
      <div className="flex flex-col gap-6 text-justify">
        <p className="font-['Inter',sans-serif] font-light text-[clamp(0.82rem,1.05vw,14px)] text-black/90 leading-relaxed">
          <span className="font-bold text-black">System Architecture</span>
          {" — TAHIMIK's end-to-end pipeline consists of a byte encoding module that maps raw UTF-8 text into byte IDs, an early encoder stage (layers 0–2), an integrated lightweight Noise Estimator that predicts sentence noise density ("}
          <span className="font-math font-semibold text-black">η<sub>i</sub></span>
          {"), an Adaptive Delete Gate that dynamically assigns token retention masks according to "}
          <span className="font-math font-semibold text-black">d<sub>target</sub> = d<sub>max</sub> × (1 − η<sub>i</sub>)</span>
          {", and the deep encoder-decoder backbone. This selectively prunes bytes while preserving contextual information needed for normalization. Comparative accuracy and efficiency are reported only after controlled evaluation."}
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 pt-4 border-t border-black/10">
          <div className="flex flex-col gap-1">
            <span className="font-['Inter',sans-serif] font-semibold text-[clamp(0.78rem,1.1vw,14px)] text-black">
              1. Byte Embedding & Encoder (L0–2)
            </span>
            <span className="font-['Inter',sans-serif] font-light text-[clamp(0.72rem,0.95vw,13px)] text-black/75 leading-relaxed">
              Maps UTF-8 byte stream into 1472-dim representations and extracts early contextual embeddings.
            </span>
          </div>
          <div className="flex flex-col gap-1">
            <span className="font-['Inter',sans-serif] font-semibold text-[clamp(0.78rem,1.1vw,14px)] text-black">
              2. Noise Estimator & Delete Gate
            </span>
            <span className="font-['Inter',sans-serif] font-light text-[clamp(0.72rem,0.95vw,13px)] text-black/75 leading-relaxed">
              Computes <span className="font-math font-semibold">η<sub>i</sub></span> and masks uninformative character representations before deep sequence processing.
            </span>
          </div>
          <div className="flex flex-col gap-1">
            <span className="font-['Inter',sans-serif] font-semibold text-[clamp(0.78rem,1.1vw,14px)] text-black">
              3. Deep Encoder & Autoregressive Decoder
            </span>
            <span className="font-['Inter',sans-serif] font-light text-[clamp(0.72rem,0.95vw,13px)] text-black/75 leading-relaxed">
              Processes pruned sequences through deep cross-attention, generating standard, clean Filipino text.
            </span>
          </div>
        </div>

        {/* Zoomable Architecture Diagram */}
        <ArchitectureDiagram />
      </div>
    </Card>
  );
}

function ProjectTab() {
  return (
    <Card>
      <div className="flex items-start justify-between mb-4 gap-3">
        <span className="font-['Inter',sans-serif] font-bold text-[clamp(1rem,2.5vw,36px)] text-black whitespace-nowrap">
          Group 9 — Members
        </span>
        <div className="flex items-center justify-center px-4 py-2 rounded-full border-[0.5px] border-black/60 bg-white shrink-0">
          <span className="font-['Inter',sans-serif] font-medium text-[clamp(0.75rem,1.5vw,18px)] text-black whitespace-nowrap leading-normal">
            BSCS 4-1
          </span>
        </div>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-6 sm:gap-8 lg:gap-10 w-full items-center justify-items-center py-4">
        {GROUP_MEMBERS.map((member) => (
          <div key={member.name} className="flex flex-col gap-3 sm:gap-4 items-center w-full">
            <div className="relative w-full max-w-[170px] sm:max-w-[190px] lg:max-w-[210px] aspect-square shrink-0 rounded-full border border-black overflow-hidden bg-white">
              <img
                alt={member.name}
                className="absolute block inset-0 size-full object-cover"
                style={member.name.includes("Julius") ? { objectPosition: "center 30%" } : {}}
                src={member.photo}
              />
            </div>
            <span className="font-['Inter',sans-serif] font-light text-[clamp(0.8rem,1.25vw,17px)] text-black text-center whitespace-nowrap leading-normal">
              {member.name}
            </span>
          </div>
        ))}
      </div>
    </Card>
  );
}

type AboutTab = "tahimik" | "system" | "project";

const ABOUT_TABS: { id: AboutTab; label: string }[] = [
  { id: "tahimik", label: "TAHIMIK"             },
  { id: "system",  label: "System Architecture" },
  { id: "project", label: "Project Information" },
];

function AboutSection() {
  const [tab, setTab] = useState<AboutTab>("tahimik");
  return (
    <div className="flex flex-col gap-6">
      <h2 className="font-['Inter',sans-serif] font-semibold text-[clamp(1.1rem,2.8vw,36px)] text-black tracking-tight">
        About
      </h2>
      <nav
        className="flex items-center gap-5 sm:gap-7 overflow-x-auto"
        style={{ scrollbarWidth: "none" } as React.CSSProperties}
      >
        {ABOUT_TABS.map(({ id, label }) => (
          <button
            key={id}
            onClick={() => setTab(id)}
            className="bg-transparent border-none cursor-pointer p-0 transition-colors duration-200 whitespace-nowrap"
            style={{
              fontFamily:    "Inter, sans-serif",
              fontSize:      "clamp(0.65rem, 1.1vw, 13px)",
              fontWeight:    tab === id ? 700 : 400,
              color:         tab === id ? "#000000" : "#9a9a9a",
              letterSpacing: "0.01em",
            }}
          >
            {label}
          </button>
        ))}
      </nav>
      {tab === "tahimik" && <TahimikTab />}
      {tab === "system"  && <SystemTab  />}
      {tab === "project" && <ProjectTab />}
    </div>
  );
}

function NavBtn({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="bg-transparent border-none cursor-pointer p-0 transition-colors duration-200 whitespace-nowrap shrink-0"
      style={{
        fontFamily:    "Inter, sans-serif",
        fontSize:      "clamp(0.65rem, 1.1vw, 13px)",
        fontWeight:    active ? 700 : 400,
        color:         active ? "#000000" : "#9a9a9a",
        letterSpacing: "0.01em",
      }}
    >
      {label}
    </button>
  );
}

function Chevron({ open }: { open: boolean }) {
  return (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      className="shrink-0 transition-transform duration-300"
      style={{ transform: open ? "rotate(180deg)" : "rotate(0deg)" }}
    >
      <path
        d="M6 9l6 6 6-6"
        stroke="#000"
        strokeWidth="1.75"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function Collapse({ open, children }: { open: boolean; children: React.ReactNode }) {
  const ref = useRef<HTMLDivElement>(null);
  const [height, setHeight] = useState<number | undefined>(undefined);

  useEffect(() => {
    if (!ref.current) return;
    const observer = new ResizeObserver(([entry]) => {
      setHeight(entry.contentRect.height);
    });
    observer.observe(ref.current);
    return () => observer.disconnect();
  }, []);

  return (
    <div
      className="overflow-hidden transition-[max-height,opacity] duration-400 ease-in-out"
      style={{
        maxHeight: open ? (height != null ? height + "px" : "9999px") : "0px",
        opacity:   open ? 1 : 0,
      }}
    >
      <div ref={ref}>{children}</div>
    </div>
  );
}

// ── Root Page Component ──────────────────────────────────────────────────

const FOOTER_H = 34; // px

export default function Root() {
  const [hasBatchData, setHasBatchData] = useState(false);
  const [isNormalizing, setIsNormalizing] = useState(false);

  const [activeSection, setActiveSection] = useState("engine");
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  function toggleExpanded(id: string) {
    setExpanded((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  }

  useEffect(() => {
    const observers = SECTIONS.map(({ id }) => {
      const el = document.getElementById(id);
      if (!el) return null;
      const obs = new IntersectionObserver(
        ([entry]) => {
          if (entry.isIntersecting) setActiveSection(id);
        },
        { rootMargin: "-20% 0px -70% 0px", threshold: 0 }
      );
      obs.observe(el);
      return { el, obs };
    });

    return () => {
      observers.forEach((o) => o?.obs.unobserve(o.el));
    };
  }, []);

  function scrollToSection(id: string) {
    if (ACCORDION_IDS.has(id)) {
      setExpanded((prev) => new Set([...prev, id]));
    }
    const el = document.getElementById(id);
    if (!el) return;
    el.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  const PAD_X = "px-4 sm:px-8 lg:px-16";

  return (
    <div className="bg-white min-h-screen w-full min-w-0 overflow-x-hidden">

      {/* Fixed compact top navbar */}
      <header
        className="fixed top-0 left-0 right-0 z-50 bg-white/95 backdrop-blur-sm"
        style={{ borderBottom: "0.5px solid rgba(0,0,0,0.15)" }}
      >
        <div
          className={`flex items-center gap-3 sm:gap-6 ${PAD_X} max-w-[1376px] mx-auto`}
          style={{ height: "52px" }}
        >
          <div className="flex items-center gap-2.5 sm:gap-3 shrink-0">
            <span className="font-['Inter',sans-serif] font-bold text-sm tracking-tight">
              TAHIMIK
            </span>
            <span className="text-[#d0d0d0] font-light select-none">|</span>
          </div>
          <nav
            className="flex items-center gap-4 sm:gap-7 overflow-x-auto min-w-0"
            style={{ scrollbarWidth: "none", msOverflowStyle: "none" } as React.CSSProperties}
          >
            {SECTIONS.map(({ id, label }) => (
              <NavBtn
                key={id}
                label={label}
                active={activeSection === id}
                onClick={() => scrollToSection(id)}
              />
            ))}
          </nav>
        </div>
      </header>

      {/* Scrollable content */}
      <main className="pt-[65px] w-full min-w-0 overflow-x-hidden" style={{ paddingBottom: FOOTER_H + 20 + "px" }}>

        {/* Engine */}
        <section id="engine" className={`scroll-mt-[50px] ${PAD_X} pt-6 sm:pt-8 pb-10 sm:pb-16 max-w-[1376px] mx-auto w-full min-w-0 overflow-hidden`}>
          <Engine onBatchModeChange={setHasBatchData} onLoadingChange={setIsNormalizing} />
        </section>

        {/* Evaluation group label */}
        <div className={`max-w-[1376px] mx-auto ${PAD_X} w-full min-w-0`}>
          <div className="flex items-center gap-3 py-4">
            <span className="font-['Inter',sans-serif] font-medium text-[clamp(0.55rem,0.85vw,11px)] text-[#bbb] uppercase tracking-widest shrink-0">
              Evaluation
            </span>
            <span className="flex-1 h-px bg-black/10" />
          </div>
        </div>

        {/* Accordion sections */}
        {SECTIONS.filter((s) => s.accordion).map(({ id, label }) => {
          const isOpen = expanded.has(id);
          return (
            <section key={id} id={id} className="scroll-mt-[50px] max-w-[1376px] mx-auto">
              <div className={PAD_X}>
                <button
                  onClick={() => toggleExpanded(id)}
                  className="w-full flex items-center justify-between gap-4 py-5 cursor-pointer bg-transparent text-left border-none"
                  style={{ borderTop: "0.5px solid rgba(0,0,0,0.15)" }}
                >
                  <h2 className="font-['Inter',sans-serif] font-semibold text-[clamp(1.1rem,2.8vw,36px)] text-black tracking-tight leading-none">
                    {label}
                  </h2>
                  <Chevron open={isOpen} />
                </button>
              </div>

              <Collapse open={isOpen}>
                <div className={`${PAD_X} pb-10`}>
                  {id === "benchmark"  && <Benchmark hasBatchData={hasBatchData} isLoading={isNormalizing} />}
                  {id === "efficiency" && <Efficiency hasBatchData={hasBatchData} isLoading={isNormalizing} />}
                  {id === "validation" && <Validation hasBatchData={hasBatchData} isLoading={isNormalizing} />}
                </div>
              </Collapse>
            </section>
          );
        })}

        {/* About */}
        <section id="about" className="scroll-mt-[50px] max-w-[1376px] mx-auto">
          <div className={PAD_X}>
            <div className="flex items-center gap-3 py-4">
              <span className="font-['Inter',sans-serif] font-medium text-[clamp(0.55rem,0.85vw,11px)] text-[#bbb] uppercase tracking-widest shrink-0">
                The Platform
              </span>
              <span className="flex-1 h-px bg-black/10" />
            </div>
            <div className="pb-10">
              <AboutSection />
            </div>
          </div>
        </section>

      </main>

      {/* Footer */}
      <footer
        className="fixed bottom-0 left-0 right-0 z-50 bg-white/95 backdrop-blur-sm"
        style={{ borderTop: "0.5px solid rgba(0,0,0,0.15)" }}
      >
        <div
          className={`flex items-center gap-x-3 gap-y-0.5 flex-wrap ${PAD_X} max-w-[1376px] mx-auto`}
          style={{ minHeight: FOOTER_H + "px", paddingTop: 6, paddingBottom: 6 }}
        >
          <span className="font-['Inter',sans-serif] font-semibold text-[11px] text-black tracking-tight shrink-0">
            TAHIMIK
          </span>
          <span className="text-[#ddd] select-none shrink-0">•</span>
          <span className="font-['Inter',sans-serif] font-normal text-[11px] text-[#9a9a9a] shrink-0">Group 9</span>
          <span className="text-[#ddd] select-none shrink-0 hidden xs:inline">•</span>
          <span className="font-['Inter',sans-serif] font-normal text-[11px] text-[#9a9a9a] shrink-0 hidden sm:inline">BSCS 4-1</span>
          <span className="text-[#ddd] select-none shrink-0 hidden sm:inline">•</span>
          <span className="font-['Inter',sans-serif] font-normal text-[11px] text-[#9a9a9a] shrink-0 hidden sm:inline">2025–2026</span>
          <span className="flex-1" />
          <span className="font-['Inter',sans-serif] font-normal text-[11px] text-[#bbb] shrink-0">© 2026</span>
        </div>
      </footer>

    </div>
  );
}
