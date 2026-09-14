import { Pill, KpiCard, DataTable } from "../shared";

export default function Benchmark({ hasBatchData, isLoading }: { hasBatchData: boolean; isLoading?: boolean }) {
  const unavailable = "Awaiting validated evaluation";

  return (
    <div className="flex flex-col gap-8 w-full" style={{ opacity: hasBatchData || isLoading ? 1 : 0.5 }}>
      <div className="flex flex-wrap gap-4">
        <KpiCard label="GLEU+" value="0.00" tag={unavailable} tagColor="#fafe45" isLoading={isLoading} />
        <KpiCard label="chrF" value="0.00" tag={unavailable} tagColor="#6DC85A" isLoading={isLoading} />
        <KpiCard label="ERR" value="0.00" tag={unavailable} tagColor="#6898F8" isLoading={isLoading} />
        <KpiCard label="Alpha-Word" value="0.00%" tag={unavailable} tagColor="#f09662" isLoading={isLoading} />
      </div>
      <div className="w-full flex flex-col gap-5">
        {isLoading && (
          <div className="flex items-center justify-end">
            <span className="flex items-center gap-1.5 font-['Inter',sans-serif] text-xs text-black/70">
              <span className="w-2 h-2 rounded-full bg-[#fafe45] animate-ping" />
              Computing benchmark scores across test splits...
            </span>
          </div>
        )}
        <DataTable
          headers={[
            "EVALUATION METRIC",
            "BYT5",
            "MRT5",
            "TAHIMIK (ADAPTIVE)",
            <span>SUPERIORITY <span className="font-math font-semibold">Δ</span></span>,
          ]}
          rows={[
            ["GLEU+ (Source-Aware)", "0.00", "0.00", "0.00", "Not measured"],
            ["chrF (Char 6-Gram)", "0.00", "0.00", "0.00", "Not measured"],
            ["Error Reduction Rate", "0.00", "0.00", "0.00", "Not measured"],
            ["Alpha-Word Accuracy", "0.00", "0.00", "0.00", "Not measured"],
          ]}
          highlightCol={3}
          isLoading={isLoading}
        />
      </div>
    </div>
  );
}
