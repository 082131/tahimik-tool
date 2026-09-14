import { Pill } from "../shared";

type Row = {
  hyp: string;
  base: string;
  tah: string;
  diff: string;
  ci: string;
  rawp: string;
  holm: string;
  dec: string;
  winner: "tahimik" | "baseline";
};

function TH({
  children,
  right = false,
  center = false,
}: {
  children: React.ReactNode;
  right?: boolean;
  center?: boolean;
}) {
  return (
    <th
      className={`pb-2.5 pr-4 last:pr-0 font-['Inter',sans-serif] font-semibold text-[clamp(0.55rem,0.8vw,11px)] text-[#7A7A7A] uppercase tracking-wider border-b border-black/30 whitespace-nowrap ${
        center ? "text-center" : right ? "text-right" : "text-left"
      }`}
    >
      {children}
    </th>
  );
}

function TD({
  children,
  right = false,
  center = false,
  variant = "default",
}: {
  children: React.ReactNode;
  right?: boolean;
  center?: boolean;
  variant?: "default" | "winner" | "loser" | "dim" | "neutral" | "reject" | "retain";
}) {
  const styles: Record<string, React.CSSProperties> = {
    default: {},
    winner:  { fontWeight: 700, color: "#000000" },
    loser:   { color: "#9a9a9a" },
    dim:     { color: "#777777" },
    neutral: { fontWeight: 500, color: "#000000" },
    reject:  { fontWeight: 600, color: "#dc2626" },
    retain:  { fontWeight: 600, color: "#16a34a" },
  };
  return (
    <td
      className={`py-3 pr-4 last:pr-0 font-['Inter',sans-serif] font-normal text-[clamp(0.68rem,1.05vw,13.5px)] ${
        center ? "text-center" : right ? "text-right" : "text-left"
      }`}
      style={styles[variant]}
    >
      {children}
    </td>
  );
}

function ComparisonLabel({ children }: { children: React.ReactNode }) {
  return (
    <tr>
      <td
        colSpan={8}
        className="pt-5 pb-2 font-['Inter',sans-serif] font-semibold text-[clamp(0.6rem,0.9vw,11.5px)] text-[#7A7A7A] uppercase tracking-wider"
      >
        {children}
      </td>
    </tr>
  );
}

function DataRow({ r, last = false, disabled = false }: { r: Row; last?: boolean; disabled?: boolean }) {
  const isTahWinner = r.winner === "tahimik";
  const isReject = r.dec.toLowerCase().includes("reject");
  return (
    <tr className={last ? "" : "border-b border-black/8"}>
      {/* Hypothesis: always bold, always prominent */}
      <TD variant="winner">{r.hyp}</TD>
      {/* Baseline */}
      <TD right variant={!isTahWinner ? "winner" : "loser"}>{disabled ? "0.00" : r.base}</TD>
      {/* TAHIMIK */}
      <TD right variant={isTahWinner ? "winner" : "loser"}>{disabled ? "0.00" : r.tah}</TD>
      {/* Diff (Delta) */}
      <TD right variant="winner">{disabled ? "0.00" : r.diff}</TD>
      {/* CI / Effect */}
      <TD variant="loser">{disabled ? "0.00" : r.ci}</TD>
      {/* Raw p: background info */}
      <TD right variant="loser">{disabled ? "0.00" : r.rawp}</TD>
      {/* Holm adj. p: decision criterion */}
      <TD center variant="dim">{disabled ? "0.00" : r.holm}</TD>
      {/* Decision: conclusion */}
      <TD variant={disabled ? "dim" : isReject ? "reject" : "retain"}>{disabled ? "0.00" : r.dec}</TD>
    </tr>
  );
}

function SkeletonRow({ last = false }: { last?: boolean }) {
  return (
    <tr className={`animate-pulse ${last ? "" : "border-b border-black/8"}`}>
      <td className="py-3.5 pr-4"><div className="h-4 bg-black/10 rounded w-48" /></td>
      <td className="py-3.5 pr-4"><div className="h-4 bg-black/10 rounded w-14 ml-auto" /></td>
      <td className="py-3.5 pr-4"><div className="h-4 bg-black/10 rounded w-14 ml-auto" /></td>
      <td className="py-3.5 pr-4"><div className="h-4 bg-black/10 rounded w-12 ml-auto" /></td>
      <td className="py-3.5 pr-4"><div className="h-4 bg-black/10 rounded w-28" /></td>
      <td className="py-3.5 pr-4"><div className="h-4 bg-black/10 rounded w-12 ml-auto" /></td>
      <td className="py-3.5 pr-4"><div className="h-4 bg-black/10 rounded w-14 mx-auto" /></td>
      <td className="py-3.5 pr-4"><div className="h-4 bg-black/10 rounded w-20" /></td>
    </tr>
  );
}

function Family1({ disabled, isLoading }: { disabled: boolean; isLoading?: boolean }) {
  const unavailable: Row[] = [
    { hyp: "Validated results required", base: "0.00", tah: "0.00", diff: "0.00", ci: "0.00", rawp: "0.00", holm: "0.00", dec: "Awaiting evaluation", winner: "baseline" },
  ];

  return (
    <div className="w-full flex flex-col gap-6" style={{ opacity: disabled && !isLoading ? 0.5 : 1 }}>
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center gap-3">
          <Pill bg="#fafe45" bold>FAMILY 1: NORMALIZATION ACCURACY</Pill>
          <span className="flex-1 h-px bg-black/15 min-w-[20px]" />
        </div>
        {isLoading && (
          <span className="flex items-center gap-1.5 font-['Inter',sans-serif] text-xs text-black/70">
            <span className="w-2 h-2 rounded-full bg-[#fafe45] animate-ping" />
            Computing paired bootstrap tests & significance metrics...
          </span>
        )}
      </div>
      <div className="w-full overflow-x-auto">
        <table className="w-full border-collapse" style={{ minWidth: 760 }}>
          <thead>
            <tr>
              <TH>Hypothesis</TH>
              <TH right>Baseline</TH>
              <TH right>TAHIMIK</TH>
              <TH right>Diff (<span className="font-math font-semibold">Δ</span>)</TH>
              <TH>95% Bootstrap CI</TH>
              <TH right>Raw <span className="font-math">p</span></TH>
              <TH center>Holm Adj. <span className="font-math">p</span></TH>
              <TH>Decision (<span className="font-math">α = 0.05</span>)</TH>
            </tr>
          </thead>
          <tbody>
            <ComparisonLabel>Comparison A — TAHIMIK vs. ByT5</ComparisonLabel>
            {isLoading ? (
              [1, 2, 3, 4].map((i) => <SkeletonRow key={i} />)
            ) : (
              unavailable.map((r) => <DataRow key={r.hyp} r={r} disabled={disabled} />)
            )}
            <ComparisonLabel>Comparison B — TAHIMIK vs. MrT5</ComparisonLabel>
            {isLoading ? (
              [1, 2, 3, 4].map((i) => <SkeletonRow key={i} last={i === 4} />)
            ) : (
              unavailable.map((r) => <DataRow key={r.hyp} r={r} last disabled={disabled} />)
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Family2({ disabled, isLoading }: { disabled: boolean; isLoading?: boolean }) {
  const unavailable: Row[] = [
    { hyp: "Validated results required", base: "0.00", tah: "0.00", diff: "0.00", ci: "0.00", rawp: "0.00", holm: "0.00", dec: "Awaiting evaluation", winner: "baseline" },
  ];

  return (
    <div className="w-full flex flex-col gap-6" style={{ opacity: disabled && !isLoading ? 0.5 : 1 }}>
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center gap-3">
          <Pill bg="#f6c1f7" bold>FAMILY 2: COMPUTATIONAL EFFICIENCY</Pill>
          <span className="flex-1 h-px bg-black/15 min-w-[20px]" />
        </div>
        {isLoading && (
          <span className="flex items-center gap-1.5 font-['Inter',sans-serif] text-xs text-black/70">
            <span className="w-2 h-2 rounded-full bg-[#f6c1f7] animate-ping" />
            Computing Wilcoxon tests & effect sizes...
          </span>
        )}
      </div>
      <div className="w-full overflow-x-auto">
        <table className="w-full border-collapse" style={{ minWidth: 760 }}>
          <thead>
            <tr>
              <TH>Hypothesis</TH>
              <TH right>Baseline</TH>
              <TH right>TAHIMIK</TH>
              <TH right>Diff (<span className="font-math font-semibold">Δ</span>)</TH>
              <TH>Confidence / Effect</TH>
              <TH right>Raw <span className="font-math">p</span></TH>
              <TH center>Holm Adj. <span className="font-math">p</span></TH>
              <TH>Decision (<span className="font-math">α = 0.05</span>)</TH>
            </tr>
          </thead>
          <tbody>
            <ComparisonLabel>Comparison A — TAHIMIK vs. ByT5</ComparisonLabel>
            {isLoading ? (
              [1, 2].map((i) => <SkeletonRow key={i} />)
            ) : (
              unavailable.map((r) => <DataRow key={r.hyp} r={r} disabled={disabled} />)
            )}
            <ComparisonLabel>Comparison B — TAHIMIK vs. MrT5</ComparisonLabel>
            {isLoading ? (
              [1, 2].map((i) => <SkeletonRow key={i} last={i === 2} />)
            ) : (
              unavailable.map((r) => <DataRow key={r.hyp} r={r} last disabled={disabled} />)
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default function Validation({ hasBatchData, isLoading }: { hasBatchData: boolean; isLoading?: boolean }) {
  const disabled = !hasBatchData;
  return (
    <div className="flex flex-col gap-10 w-full">
      <Family1 disabled={disabled} isLoading={isLoading} />
      <Family2 disabled={disabled} isLoading={isLoading} />
    </div>
  );
}
