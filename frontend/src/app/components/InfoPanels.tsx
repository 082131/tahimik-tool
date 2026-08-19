import aimeePhoto from "@/assets/aimeephoto.png";
import jacePhoto from "@/assets/jacephoto.jpg";
import juliusPhoto from "@/assets/julsphoto.png";
import richardPhoto from "@/assets/richardphoto.png";
import rjayPhoto from "@/assets/rjayphoto.png";

const GROUP_MEMBERS = [
  { name: "Cerene, Rjay B.", initials: "RC", photo: rjayPhoto },
  {
    name: "Divinagracia, Julius F. II.",
    initials: "JD",
    photo: juliusPhoto,
    photoClassName: "member-card__image--julius",
  },
  {
    name: "Federico, John Richard J.",
    initials: "JF",
    photo: richardPhoto,
    photoClassName: "member-card__image--richard",
  },
  {
    name: "Layesa, John Carlo C.",
    initials: "JL",
    photo: jacePhoto,
    photoClassName: "member-card__image--jace",
  },
  {
    name: "Maniego, Aimee C.",
    initials: "AM",
    photo: aimeePhoto,
    photoClassName: "member-card__image--aimee",
  },
];

type InformationPanelProps = {
  children: React.ReactNode;
  compact?: boolean;
};

function InformationPanel({
  children,
  compact = false,
}: InformationPanelProps) {
  return (
    <div
      className={`information-panel${compact ? " information-panel--compact" : ""}`}
    >
      {children}
    </div>
  );
}

export function AboutTahimikPanel() {
  return (
    <InformationPanel>
      <p className="information-panel__text">
        <strong>TAHIMIK</strong> — Text Augmentation and Harmonization of
        Informal and Multilingual Input for Knowledge Extraction — normalizes
        noisy Tagalog and Taglish social media text back into clean, readable
        Filipino. It handles the nine noise categories common in Philippine
        online writing: abbreviations, orthographic variation, character
        elongation, punctuation and capitalization noise, slang, Taglish
        morphology, emoji, and code-switching.
        {" "}
        The system builds on <em>ByT5</em> and <em>MrT5</em>, which read raw
        UTF-8 bytes instead of subword tokens, so misspellings and invented
        spellings never fall outside the vocabulary. Its contribution is a{" "}
        <strong className="information-panel__emphasis">
          noise-adaptive delete gate
        </strong>{" "}
        that decides how aggressively to compress each sentence based on how
        noisy that sentence actually is.
      </p>
    </InformationPanel>
  );
}

export function SystemArchitecturePanel() {
  return (
    <InformationPanel>
      <p className="information-panel__text">
        <strong>System Architecture</strong>
        {" "}
        Input text is encoded as raw UTF-8 bytes and passed through the first
        encoder layers. A small <em>noise estimator</em> mean-pools those
        hidden states and predicts a per-sentence noise score between 0 and 1.
        That score shifts a learned <em>delete gate</em>, which drops redundant
        bytes before the remaining encoder layers and the decoder run — clean
        sentences get compressed aggressively for speed, while noisy sentences
        are kept intact so the decoder has enough signal to correct them.
        {" "}
        Training runs in two stages: synthetic pretraining on generated noisy
        and clean pairs, then fine-tuning on gold-standard annotations. Three
        variants are compared under identical settings — a ByT5 baseline with
        no compression, MrT5 with a fixed deletion rate, and TAHIMIK with
        noise-adaptive deletion — and evaluated on GLEU+, chrF, error
        reduction rate, inference time, and peak GPU memory.
      </p>
    </InformationPanel>
  );
}

export function ProjectInfoPanel() {
  return (
    <InformationPanel compact>
      <div className="project-info__header">
        <h2 className="project-info__title">Group 10 - Members</h2>
        <span className="project-info__course">BSCS 3-1</span>
      </div>

      <div className="member-list">
        {GROUP_MEMBERS.map((member) => (
          <article className="member-card" key={member.name}>
            <div className="member-card__photo">
              {member.photo ? (
                <img
                  className={`member-card__image${
                    member.photoClassName ? ` ${member.photoClassName}` : ""
                  }`}
                  src={member.photo}
                  alt={member.name}
                />
              ) : (
                <span
                  className="member-card__placeholder"
                  aria-label={`${member.name} photo placeholder`}
                >
                  {member.initials}
                </span>
              )}
            </div>
            <span className="member-card__name">{member.name}</span>
          </article>
        ))}
      </div>
    </InformationPanel>
  );
}
