import { useNavigate } from "react-router";

const P = {
  starBR: "M274.035 137V274H137.018C137.018 198.332 198.358 137 274.035 137Z",
  starTR: "M274.035 0V137C198.358 137 137.018 75.6678 137.018 0H274.035Z",
  starTL: "M0 0H137.018C137.018 75.6678 75.6775 137 0 137V0Z",
  starBL: "M137.018 274H0V137C75.6775 137 137.018 198.332 137.018 274Z",
  leafR:  "M216.902 1.55303C216.902 61.3522 169.034 109.816 109.975 109.816C109.453 109.816 108.959 109.816 108.451 109.801C108.466 49.1747 157.002 0.0290287 216.888 0C216.902 0.522517 216.902 1.03051 216.902 1.55303Z",
  leafL:  "M0 1.55303C0 61.3522 47.8683 109.816 106.927 109.816C107.45 109.816 107.943 109.816 108.451 109.801C108.437 49.1747 59.9007 0.0290287 0.0145147 0C3.32531e-07 0.522517 0 1.03051 0 1.55303Z",
  fan:    "M159 0V159H79.5C79.5 115.091 43.9094 79.5 0 79.5C43.9094 79.5 79.5 43.9094 79.5 0H159Z",
  tri1:   "M80.5 0L54 51H107L80.5 0Z",
  tri2:   "M26.9914 0L0 51H54L26.9914 0Z",
  tri3:   "M133.492 0L107 51H160L133.492 0Z",
  tri4:   "M187.009 0L160 51H214L187.009 0Z",
  tri5:   "M240.5 0L214 51H267L240.5 0Z",
  tri6:   "M294 0L267 51H321L294 0Z",
  arc:    "M217 217H152.473C152.473 132.856 84.2654 64.666 0.138857 64.666V217H0V0C119.836 0 217 97.1466 217 217Z",
};

function TulipShape() {
  return (
    <svg viewBox="0 0 216.902 109.816" fill="none" className="w-full h-auto" style={{ transform: "scaleX(-1)" }}>
      <path d={P.leafR} fill="#F09662" />
      <path d={P.leafL} fill="#F09662" />
    </svg>
  );
}

export default function Landing() {
  const navigate = useNavigate();

  function handleEnter() {
    navigate("/app");
  }

  return (
    <div
      className="relative w-full h-screen overflow-hidden flex cursor-pointer select-none bg-white"
      onClick={handleEnter}
    >
      {/* Left content */}
      <div className="flex-1 relative flex flex-col justify-between p-[6%] sm:p-[5%] overflow-hidden min-w-0">
        {/* Top: class info */}
        <p
          className="font-['Poppins',sans-serif] font-normal text-black"
          style={{ fontSize: "clamp(0.75rem, 1.8vw, 26px)", letterSpacing: "-0.03em" }}
        >
          Group 9 • BSCS 4-1
        </p>

        {/* Bottom: title block */}
        <div className="flex flex-col" style={{ gap: "clamp(6px, 1.2vw, 20px)" }}>
          <h1
            className="font-['Poppins',sans-serif] font-bold text-black"
            style={{
              fontSize:      "clamp(3.2rem, 14vw, 210px)",
              letterSpacing: "-0.06em",
              lineHeight:    0.88,
            }}
          >
            TAHIMIK
          </h1>
          <p
            className="font-['Poppins',sans-serif] font-normal text-black"
            style={{
              fontSize:   "clamp(0.72rem, 1.4vw, 22px)",
              letterSpacing: "-0.03em",
              lineHeight: 1.4,
              maxWidth:   "52ch",
            }}
          >
            Text Augmentation and Harmonization of Informal and Multilingual Input for Knowledge Extraction
          </p>
          <span
            className="font-['Inter',sans-serif] font-medium text-[#bbb] uppercase tracking-widest mt-1"
            style={{ fontSize: "clamp(0.5rem, 0.85vw, 11px)" }}
          >
            Tap anywhere to continue →
          </span>
        </div>
      </div>

      {/* Right decorative column */}
      <div
        className="hidden sm:flex flex-col shrink-0 overflow-hidden"
        style={{ width: "clamp(140px, 22.3%, 321px)" }}
      >
        {/* 1 — Green star on purple */}
        <div
          className="relative overflow-hidden flex items-center justify-center"
          style={{ flex: "2.65", backgroundColor: "#867CFF" }}
        >
          <div style={{ width: "85%", aspectRatio: "1" }}>
            <svg viewBox="0 0 274.035 274" fill="none" className="w-full h-full">
              <path d={P.starTL} fill="#86E992" />
              <path d={P.starTR} fill="#86E992" />
              <path d={P.starBL} fill="#86E992" />
              <path d={P.starBR} fill="#86E992" />
            </svg>
          </div>
        </div>

        {/* 2 — Orange + blue/pink + fan + circle */}
        <div className="relative overflow-hidden flex" style={{ flex: "1.55" }}>
          <div style={{ width: "53.9%", backgroundColor: "#F09662" }} />
          <div className="flex flex-col" style={{ flex: 1 }}>
            <div style={{ flex: "1.28", backgroundColor: "#7098FA" }} />
            <div style={{ flex: "1",    backgroundColor: "#F6C1F7" }} />
          </div>
          <div className="absolute inset-y-0" style={{ left: "4.4%", height: "100%", aspectRatio: "1" }}>
            <svg viewBox="0 0 159 159" fill="none" className="w-auto h-full">
              <path d={P.fan} fill="#AFA7FF" />
            </svg>
          </div>
          <div className="absolute inset-y-0 overflow-hidden" style={{ left: "52%", right: "-2%", height: "100%", aspectRatio: "1" }}>
            <svg viewBox="0 0 159 159" fill="none" className="w-auto h-full">
              <circle cx="79.5" cy="79.5" r="79.5" fill="#AFA7FF" />
            </svg>
          </div>
        </div>

        {/* 3 — Yellow with green triangles */}
        <div
          className="relative overflow-hidden"
          style={{ flex: "0.5", backgroundColor: "#FAFE45" }}
        >
          <svg
            viewBox="0 0 321 51"
            fill="none"
            preserveAspectRatio="xMidYMid slice"
            className="absolute inset-0 w-full h-full"
          >
            <path d={P.tri2} fill="#86E992" />
            <path d={P.tri1} fill="#86E992" />
            <path d={P.tri3} fill="#86E992" />
            <path d={P.tri4} fill="#86E992" />
            <path d={P.tri5} fill="#86E992" />
            <path d={P.tri6} fill="#86E992" />
          </svg>
        </div>

        {/* 4 — Pink strip + blue panel with tulips */}
        <div className="flex overflow-hidden" style={{ flex: "4.17" }}>
          <div style={{ width: "14%", backgroundColor: "#F6C1F7" }} />
          <div
            className="flex flex-col items-center justify-around"
            style={{ flex: 1, backgroundColor: "#7098FA", paddingBlock: "8%" }}
          >
            <TulipShape />
            <TulipShape />
            <TulipShape />
          </div>
        </div>

        {/* 5 — Green with blue arcs */}
        <div className="relative overflow-hidden" style={{ flex: "1.13", backgroundColor: "#86E992" }}>
          <div className="absolute top-0 left-0 pointer-events-none" style={{ width: "67.6%", aspectRatio: "1" }}>
            <svg viewBox="0 0 217 217" fill="none" className="w-full h-full">
              <path d={P.arc} fill="#7098FA" />
            </svg>
          </div>
          <div className="absolute top-0 right-0 pointer-events-none" style={{ width: "67.6%", aspectRatio: "1", transform: "rotate(-90deg)" }}>
            <svg viewBox="0 0 217 217" fill="none" className="w-full h-full">
              <path d={P.arc} fill="#7098FA" />
            </svg>
          </div>
        </div>
      </div>

      {/* Mobile bottom accent strip */}
      <div
        className="sm:hidden absolute bottom-0 left-0 right-0 flex overflow-hidden"
        style={{ height: "clamp(48px, 12vw, 80px)" }}
      >
        <div style={{ flex: "2.65", backgroundColor: "#867CFF" }} />
        <div style={{ flex: "1.55", backgroundColor: "#F09662" }} />
        <div style={{ flex: "0.5",  backgroundColor: "#FAFE45" }} />
        <div style={{ flex: "2",    backgroundColor: "#7098FA" }} />
        <div style={{ flex: "1.13", backgroundColor: "#86E992" }} />
      </div>
    </div>
  );
}
