import { useState } from "react";
import { Pill, KpiCard, DataTable } from "../shared";

export default function Efficiency({ hasBatchData, isLoading }: { hasBatchData: boolean; isLoading?: boolean }) {
  const isSingleInput = !hasBatchData;
  // Configurable profiling metadata (non-hardcoded state/data structure)
  const [profiling] = useState({
    runs: 20,
    batchSize: 1,
    warmupRuns: 5,
  });

  return (
    <div className="flex flex-col gap-8 w-full">
      <div className="flex flex-wrap gap-4" style={{ opacity: isSingleInput && !isLoading ? 0.5 : 1 }}>
        <KpiCard
          label="LATENCY / SENTENCE"
          value="0.00 ms"
          tag="Awaiting benchmark run"
          tagColor="#fafe45"
          isLoading={isLoading}
        />
        <KpiCard
          label="PEAK GPU MEMORY"
          value="0.00 GB"
          tag="Awaiting benchmark run"
          tagColor="#6DC85A"
          isLoading={isLoading}
        />
        <KpiCard
          label="PRUNING RATIO"
          value="0.00%"
          tag="Awaiting benchmark run"
          tagColor="#6898F8"
          isLoading={isLoading}
        />
      </div>
      <div className="w-full flex flex-col gap-5">
        <div className="flex flex-col gap-5" style={{ opacity: isSingleInput && !isLoading ? 0.5 : 1 }}>
          {isLoading && (
            <div className="flex items-center justify-end">
              <span className="flex items-center gap-1.5 font-['Inter',sans-serif] text-xs text-black/70">
                <span className="w-2 h-2 rounded-full bg-[#fafe45] animate-ping" />
                Profiling GPU memory and latency benchmarks...
              </span>
            </div>
          )}
          <DataTable
            headers={[
              "RUNTIME METRIC",
              "BYT5",
              "MRT5",
              "TAHIMIK (ADAPTIVE)",
              "ADVANTAGE",
            ]}
            rows={[
              ["Inference Latency", "0.00 ms", "0.00 ms", "0.00 ms", "Not measured"],
              ["Peak GPU Memory during Inference", "0.00 GB", "0.00 GB", "0.00 GB", "Not measured"],
            ]}
            highlightCol={3}
            isLoading={isLoading}
          />
        </div>
      </div>
    </div>
  );
}
