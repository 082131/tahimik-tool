import { useState } from "react";
import {
  AboutTahimikPanel,
  ProjectInfoPanel,
  SystemArchitecturePanel,
} from "./InfoPanels";

type PanelId = "about" | "system" | "project";
type PanelColor = "yellow" | "pink" | "orange";

type AccordionButtonProps = {
  active: boolean;
  color: PanelColor;
  controls: string;
  onClick: () => void;
  children: React.ReactNode;
};

type AccordionPanelProps = {
  id: string;
  open: boolean;
  children: React.ReactNode;
};

function AccordionButton({
  active,
  color,
  controls,
  onClick,
  children,
}: AccordionButtonProps) {
  return (
    <button
      type="button"
      aria-controls={controls}
      aria-expanded={active}
      className={`pill-button pill-button--${color}${active ? " is-active" : ""}`}
      onClick={onClick}
    >
      {children}
    </button>
  );
}

function AccordionPanel({ id, open, children }: AccordionPanelProps) {
  return (
    <div
      id={id}
      aria-hidden={!open}
      className={`accordion-panel${open ? " is-open" : ""}`}
    >
      <div className="accordion-panel__overflow">
        <div className="accordion-panel__content">{children}</div>
      </div>
    </div>
  );
}

export function InfoAccordion() {
  const [activePanel, setActivePanel] = useState<PanelId | null>(null);

  function togglePanel(panel: PanelId) {
    setActivePanel((currentPanel) =>
      currentPanel === panel ? null : panel,
    );
  }

  return (
    <section className="info-accordion" aria-label="Project details">
      <div className="info-accordion__controls">
        <div className="info-accordion__primary-controls">
          <AccordionButton
            active={activePanel === "about"}
            color="yellow"
            controls="about-panel"
            onClick={() => togglePanel("about")}
          >
            About TAHIMIK
          </AccordionButton>
          <AccordionButton
            active={activePanel === "system"}
            color="pink"
            controls="system-panel"
            onClick={() => togglePanel("system")}
          >
            System Architecture
          </AccordionButton>
        </div>

        <AccordionButton
          active={activePanel === "project"}
          color="orange"
          controls="project-panel"
          onClick={() => togglePanel("project")}
        >
          About Us
        </AccordionButton>
      </div>

      <AccordionPanel id="about-panel" open={activePanel === "about"}>
        <AboutTahimikPanel />
      </AccordionPanel>
      <AccordionPanel id="system-panel" open={activePanel === "system"}>
        <SystemArchitecturePanel />
      </AccordionPanel>
      <AccordionPanel id="project-panel" open={activePanel === "project"}>
        <ProjectInfoPanel />
      </AccordionPanel>
    </section>
  );
}
