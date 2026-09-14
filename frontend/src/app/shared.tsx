import { useRef } from "react";
import aimeePhoto from "@/assets/aimeephoto.png";
import jacePhoto from "@/assets/jacephoto.jpg";
import juliusPhoto from "@/assets/julsphoto.png";
import richardPhoto from "@/assets/richardphoto.png";
import rjayPhoto from "@/assets/rjayphoto.png";

export { aimeePhoto as memberPhoto };

export interface GroupMember {
  name: string;
  initials: string;
  photo: string;
}

export const GROUP_MEMBERS: GroupMember[] = [
  { name: "Cerene, Rjay B.", initials: "RC", photo: rjayPhoto },
  { name: "Divinagracia, Julius F. II.", initials: "JD", photo: juliusPhoto },
  { name: "Federico, John Richard J.", initials: "JF", photo: richardPhoto },
  { name: "Layesa, John Carlo C.", initials: "JL", photo: jacePhoto },
  { name: "Maniego, Aimee C.", initials: "AM", photo: aimeePhoto },
];

export const BYTE_CHIPS = [
  { text: "Sanaol", keep: true },
  { text: "nlng", keep: false },
  { text: "tlga", keep: false },
  { text: "sa", keep: true },
  { text: "inyo", keep: true },
  { text: "mga", keep: true },
  { text: "lodi", keep: false },
];

export function PillButton({
  children,
  active,
  activeColor,
  onClick,
}: {
  children: React.ReactNode;
  active?: boolean;
  activeColor?: string;
  onClick?: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className="cursor-pointer flex items-center justify-center px-4 sm:px-6 py-2 sm:py-3 rounded-full border-[0.5px] border-black/60 transition-colors duration-300 shrink-0"
      style={{ backgroundColor: active ? activeColor : "white" }}
    >
      <span className="font-['Inter',sans-serif] font-medium text-[clamp(0.75rem,1.8vw,20px)] text-black whitespace-nowrap leading-normal">
        {children}
      </span>
    </button>
  );
}

export function Pill({
  children,
  bg = "white",
  bold,
}: {
  children: React.ReactNode;
  bg?: string;
  bold?: boolean;
}) {
  return (
    <div
      className="flex items-center self-start px-3 py-1 rounded-full border-[0.5px] border-black/60 shrink-0"
      style={{ backgroundColor: bg }}
    >
      <span
        className={`font-['Inter',sans-serif] text-[clamp(0.65rem,1.1vw,13px)] text-black whitespace-nowrap tracking-wider uppercase ${bold ? "font-semibold" : "font-medium"}`}
      >
        {children}
      </span>
    </div>
  );
}

export function SlidingPanel({
  isOpen,
  children,
}: {
  isOpen: boolean;
  children: React.ReactNode;
}) {
  const contentRef = useRef<HTMLDivElement>(null);
  return (
    <div
      className="w-full overflow-hidden transition-[max-height,opacity] duration-500 ease-in-out"
      style={{
        maxHeight: isOpen
          ? contentRef.current
            ? contentRef.current.scrollHeight + "px"
            : "9999px"
          : "0px",
        opacity: isOpen ? 1 : 0,
      }}
    >
      <div ref={contentRef} className="w-full pt-2">
        {children}
      </div>
    </div>
  );
}

export function Card({
  children,
  className = "",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={`w-full rounded-[24px] border-[0.5px] border-black/60 bg-white p-5 sm:p-6 flex flex-col gap-4 ${className}`}
    >
      {children}
    </div>
  );
}

export function KpiCard({
  label,
  value,
  tag,
  tagColor,
  isLoading,
}: {
  label: React.ReactNode;
  value: string;
  tag?: React.ReactNode;
  tagColor?: string;
  isLoading?: boolean;
}) {
  if (isLoading) {
    return (
      <div className="flex flex-col gap-3 flex-1 min-w-[120px] rounded-[24px] border-[0.5px] border-black/60 bg-white p-3 sm:p-5 animate-pulse">
        <span className="font-['Inter',sans-serif] font-medium text-[clamp(0.6rem,1vw,13px)] text-[#7A7A7A] uppercase tracking-wider">
          {label}
        </span>
        <div className="h-10 bg-black/10 rounded-md w-3/4 animate-pulse my-1" />
        <div className="h-6 w-24 bg-black/10 rounded-full animate-pulse" />
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3 flex-1 min-w-[120px] rounded-[24px] border-[0.5px] border-black/60 bg-white p-3 sm:p-5">
      <span className="font-['Inter',sans-serif] font-medium text-[clamp(0.6rem,1vw,13px)] text-[#7A7A7A] uppercase tracking-wider">
        {label}
      </span>
      <span className="font-['Inter',sans-serif] font-bold text-[clamp(1.6rem,3.5vw,52px)] text-black leading-none">
        {value}
      </span>
      {tag && (
        <div
          className="self-start flex items-center px-3 py-1 rounded-full border-[0.5px] border-black/60"
          style={{ backgroundColor: tagColor || "#fafe45" }}
        >
          <span className="font-['Inter',sans-serif] font-medium text-[clamp(0.6rem,0.9vw,12px)] text-black whitespace-nowrap">
            {tag}
          </span>
        </div>
      )}
    </div>
  );
}

export function DataTable({
  headers,
  rows,
  highlightCol,
  isLoading,
}: {
  headers: (string | React.ReactNode)[];
  rows: (string | React.ReactNode)[][];
  highlightCol?: number;
  isLoading?: boolean;
}) {
  return (
    <div className="w-full overflow-x-auto">
      <table className="w-full border-collapse">
        <thead>
          <tr>
            {headers.map((h, i) => (
              <th
                key={i}
                className="text-left pb-2.5 pr-4 last:pr-0 font-['Inter',sans-serif] font-semibold text-[clamp(0.55rem,0.8vw,11px)] text-[#7A7A7A] uppercase tracking-wider border-b border-black/30 whitespace-nowrap"
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {isLoading ? (
            [1, 2, 3, 4].map((ri) => (
              <tr key={ri} className="border-b border-black/10 animate-pulse">
                {headers.map((_, ci) => (
                  <td key={ci} className="py-3.5 pr-4 last:pr-0">
                    <div className="h-4 bg-black/10 rounded w-4/5" />
                  </td>
                ))}
              </tr>
            ))
          ) : (
            rows.map((row, ri) => (
              <tr
                key={ri}
                className={ri < rows.length - 1 ? "border-b border-black/10" : ""}
              >
                {row.map((cell, ci) => (
                  <td
                    key={ci}
                    className="py-3 pr-4 last:pr-0 font-['Inter',sans-serif] font-normal text-[clamp(0.7rem,1.1vw,14px)] text-black"
                    style={ci === highlightCol ? { fontWeight: 700 } : {}}
                  >
                    {cell}
                  </td>
                ))}
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}
