// TAHIMIK annotation constants — rulebook rules and label dictionary.
// SOURCE OF TRUTH: annotator workbook, sheets 02_Rulebook and 03_Label-Dictionary.
// Text is reproduced VERBATIM from the workbook. Do not paraphrase, invent, or
// reword. If the rulebook changes, re-export from the workbook and regenerate.

export interface Rule {
  id: string;
  rule: string;
  decision: string;
  doThis: string;
  doNot: string;
  example: string;
}

export const RULEBOOK_TITLE = "TAHIMIK Rulebook";
export const RULEBOOK_NOTE =
  "Living rulebook during calibration; freeze after final calibration if α ≥ 0.80.";

export const RULES: Rule[] = [
  {
    id: "R00",
    rule: "Task Definition",
    decision:
      "Normalize noisy Filipino/Taglish social media text into a cleaner form while preserving meaning and original language choice.",
    doThis: "Correct noise; preserve sentence meaning.",
    doNot: "Do not translate, paraphrase, formalize, or summarize.",
    example: "grabe ang inittt today 😭 → Grabe ang init today 😭",
  },
  {
    id: "R01",
    rule: "No Translation",
    decision: "English words remain English; Filipino words remain Filipino.",
    doThis: "food stays food; today stays today; dog stays dog.",
    doNot: "food→pagkain; today→ngayon; dog→aso.",
    example: "ang sarap ng food ditooo → Ang sarap ng food dito.",
  },
  {
    id: "R02",
    rule: "Preserve Code-Switching",
    decision:
      "Code-switching is a valid feature of Filipino/Taglish social media text.",
    doThis: "Retain natural English insertions and Taglish structure.",
    doNot: "Force monolingual Filipino or English output.",
    example: "Need ko mag review → Need ko mag-review.",
  },
  {
    id: "R03",
    rule: "Preserve Emojis",
    decision:
      "Keep emojis exactly as they appear when they carry sentiment, humor, emphasis, sarcasm, or social tone.",
    doThis: "Keep 😭 😂 🥹 ❤️ in output.",
    doNot: "Delete emojis or convert them into words.",
    example: "late ako 😭 → Late ako 😭",
  },
  {
    id: "R04",
    rule: "Character Elongation",
    decision: "Reduce repeated expressive letters to base spelling.",
    doThis: "inittt→init; sarappp→sarap; cuteee→cute.",
    doNot: "Change the word choice or sentiment.",
    example: "Ang cuteee mo → Ang cute mo.",
  },
  {
    id: "R05",
    rule: "Abbreviations",
    decision: "Expand clear abbreviations and shortened forms.",
    doThis: "di→hindi; rn→right now; pls→please; tmrw→tomorrow.",
    doNot: "Guess unclear abbreviations.",
    example: "di pa ako ready rn → Hindi pa ako ready right now.",
  },
  {
    id: "R06",
    rule: "Orthographic Variation",
    decision:
      "Correct misspellings, phonetic spellings, vowel omissions, and informal spellings when clear.",
    doThis: "nangyare→nangyari; byahe→biyahe; aq→ako.",
    doNot: "Overcorrect slang or named entities.",
    example: "nasa byahe pa ko → Nasa biyahe pa ako.",
  },
  {
    id: "R07",
    rule: "Taglish Morphology",
    decision:
      "Standardize Filipino affix + English root forms without translating the English root.",
    doThis: "mag-review; i-send; na-download; nagcha-charge.",
    doNot: "Translate the English root.",
    example: "paki send → Paki-send.",
  },
  {
    id: "R08",
    rule: "Slang and Netspeak",
    decision:
      "Retain slang if it carries social meaning; normalize only spelling noise.",
    doThis: "Keep legit, keri, huhu, eme if meaningful.",
    doNot: "Convert all slang into formal Filipino.",
    example: "Huhu di pa ako ready → Huhu, hindi pa ako ready.",
  },
  {
    id: "R09",
    rule: "Punctuation/Capitalization",
    decision: "Apply light capitalization and punctuation only for readability.",
    doThis: "Capitalize first word, acronyms, proper nouns.",
    doNot: "Over-edit style or add formal syntax.",
    example: "edsa traffic ulit → EDSA traffic ulit.",
  },
  {
    id: "R10",
    rule: "Hashtags",
    decision:
      "Preserve hashtags if they carry topic, joke, campaign, or discourse meaning.",
    doThis: "Keep #WalangPasok, #studentlife.",
    doNot: "Delete hashtags by default.",
    example: "late na naman #studentlife → Late na naman #studentlife.",
  },
  {
    id: "R11",
    rule: "Mentions/Usernames",
    decision: "Preserve or anonymize consistently based on ethics protocol.",
    doThis: "Use @user or <USER> depending on data policy.",
    doNot: "Normalize usernames as ordinary words.",
    example: "@user thank u → @user thank you.",
  },
  {
    id: "R12",
    rule: "URLs",
    decision: "Preserve URLs or replace with <URL> consistently.",
    doThis: "Use <URL> if anonymization is required.",
    doNot: "Edit URL text.",
    example: "check this link → Check this <URL>.",
  },
  {
    id: "R13",
    rule: "Sensitive/Censored Text",
    decision:
      "Do not automatically uncensor sensitive content; preserve surface meaning.",
    doThis: "Mark NEEDS_REVIEW when uncertain.",
    doNot: "Uncensor profanity by default.",
    example: "p*ta naman → p*ta naman.",
  },
  {
    id: "R14",
    rule: "Ambiguity",
    decision: "If meaning cannot be confidently inferred, mark NEEDS_REVIEW.",
    doThis: "Use Notes column.",
    doNot: "Invent a normalized output.",
    example: "nmn aq d2 x → NEEDS_REVIEW",
  },
  {
    id: "R15",
    rule: "Final Output Style",
    decision: "Output should be cleaner but still natural for social media.",
    doThis: "Meaning-preserving, code-switch-preserving, emoji-preserving.",
    doNot: "Formal Filipino translation.",
    example: "OMG di ko gets → OMG, hindi ko gets.",
  },
];

export type LabelKey =
  | "ABBR"
  | "ORTHO"
  | "ELONG"
  | "CS"
  | "EMOJI"
  | "SLANG"
  | "MORPH"
  | "PUNC"
  | "CAPS"
  | "LAUGH_MARKER"
  | "REACTION_MARKER"
  | "HASHTAG"
  | "MENTION"
  | "URL";

export type LabelGroup = "Noise" | "Function" | "Metadata";

export interface LabelDef {
  key: LabelKey;
  category: string;
  definition: string;
  examples: string;
  decision: string;
  coding: string;
  group: LabelGroup;
}

// Grouping follows the section headers in 03_Label-Dictionary:
// NOISE LABELS (9), SENTENCE-LEVEL FUNCTION LABELS (2), METADATA AND WORKFLOW LABELS (3).
export const LABELS: LabelDef[] = [
  {
    key: "ABBR",
    category: "Abbreviation / Shortening",
    definition:
      "Shortened token, acronym, clipped form, or compressed expression.",
    examples: " rn, reqs, gc, tix",
    decision: `ABBR = 1
- Initialism 
(e.g. UMID, ID, BBM, PH, BTS, NBA, EDSA)
- Acronym 
(e.g. NASA, ASEAN — pronounced as words)
- Clipping / truncation 
(e.g. pass for password, tom for tomorrow, prof for professor, info forinformation, fb for Facebook, gc for groupchat, tix for tickets, reqs for requirements)
- Informal social-media abbreviation 
(e.g. rn, omg, btw, idk, tyl, wtf, asap, smtix for SM Tickets)

ABBR = 0
- No abbreviation`,
    coding: "Binary 1/0",
    group: "Noise",
  },
  {
    key: "ORTHO",
    category: "Orthographic Variation",
    definition:
      "Nonstandard spelling, typo, phonetic spelling, vowel omission, or informal spelling.",
    examples: "nangyare, byahe, aq, slmt, di, skin",
    decision: `ORTHO = 1
- Misspelling or typo 
(e.g. nangyare for nangyari)
- Phonetic respelling 
(e.g. byahe for biyahe, kumsin for kumain)
- Vowel omission 
(e.g. slmt for salamat, bkt for bakit)
- Numeric-character substitution 
(e.g. gus2, d2, p4)
- Informal spelling 
(e.g. aq for ako, sha for siya)
- Grammar-orthography confusion 
(e.g. ng vs nang when wrong; din vs rin when wrong)

ORTHO = 1, SLANG = 1
- Phonetic respelling that is also slang 
(e.g. ferson, forda — ALSO label SLANG=1)

ORTHO = 0
- Already-standard Filipino spelling`,
    coding: "Binary 1/0",
    group: "Noise",
  },
  {
    key: "ELONG",
    category: "Character Elongation",
    definition: "Repeated letters used for emphasis or emotion.",
    examples: "inittt, sarappp, cuteee, ano ba!!, bakit😭😭, :((((",
    decision: `ELONG = 1
- Repeated letters for emphasis (inittt, sarappp, cuteee, grabehhhh)
- Repeated punctuation (!!!, ???, .....)
- Repeated emojis (😭😭😭, 🤣🤣🤣)

ELONG = 0
- No repetitions`,
    coding: "Binary 1/0",
    group: "Noise",
  },
  {
    key: "CS",
    category: "Code-Switching",
    definition:
      "Filipino and English/Taglish occur in the same sentence or word formation.",
    examples: "Balak ko mag-review today.",
    decision: `CS = 1
- An English common noun, verb, adjective, adverb, conjunction, or phrase embedded in aFilipino-matrix clause (intra-sentential)
- An English clause inserted between Filipino clauses (inter-sentential)
- An English tag or filler ("you know," "I mean," "for real") inserted in Filipino text (extra-sentential)

CS = 0
- Proper nouns/named entities:
- Place names: United States, Manila, Quezon City
- Person names: Jimin, BBM, Wonu
- Brand names: iPhone, Yahoo, Shopee, BTS, Lesserafim
- Organization names: NBA, ASEAN
- Titles: Avengers, Every Wonwoo`,
    coding: "Binary 1/0",
    group: "Noise",
  },
  {
    key: "EMOJI",
    category: "Emoji / Emotive Marker",
    definition: "Emoji or visual sentiment marker.",
    examples: "😭 😂 🥹 ❤️",
    decision: `EMOJI = 1
- 😭 😂 🥹 ❤️
- :>  ;-;  ;)  :)

EMOJI = 1, ELONG = 1
- :)) 
- 😭😭😭`,
    coding: "Binary 1/0",
    group: "Noise",
  },
  {
    key: "SLANG",
    category: "Slang / Netspeak",
    definition: "Informal lexical item or youth/social-media expression.",
    examples: "keri, legit, lodi, chika, nhay'ed, shibal, horanghae",
    decision: `SLANG = 1
- Filipino netspeak (lodi, idol, petmalu, werpa, sanaol, charot, eme, keri, sana all)
- Foreign-derived slang adopted in Filipino discourse (shibal, horanghae)
- Phonetic respellings of standard words used as slang markers (ferson, forda, beh, teh)
- Filipino-specific lexical innovations (chika, gora)`,
    coding: "Binary 1/0",
    group: "Noise",
  },
  {
    key: "MORPH",
    category: "Taglish Morphology",
    definition: "Filipino affixation attached to English or borrowed root.",
    examples: "mag-review, i-send, na-download",
    decision: `MORPH = 1
- mag-review, i-send

MORPH = 1, ORTHO = 1
- magreview, isend 

MORPH = 0
- Tagalog morphology: magsulat, mag-angkat`,
    coding: "Binary 1/0",
    group: "Noise",
  },
  {
    key: "PUNC",
    category: "Punctuation",
    definition: "Missing punctuation, wrong punctuation",
    examples: "missing period",
    decision: `PUNC = 1
- Missing terminal punctuation (period, question mark, exclamation point) where thesentence is complete
- Missing comma where it separates clauses for readability
- Wrong punctuation in context (period where question mark needed)

PUNC = 1, ELONG = 1
- Repeated punctuation reduced to one mark

PUNC = 1, CAPS = 1
- Run-on segmentation requiring a new period/sentence boundary, also triggers CAPS=1 forthe new sentence start

PUNC = 0 
- Correct punctuation`,
    coding: "Binary 1/0",
    group: "Noise",
  },
  {
    key: "CAPS",
    category: "Capitalization",
    definition: "Wrong casing, acronym casing, or proper noun casing",
    examples: "edsa→EDSA, id -> ID, umid -> UMID",
    decision: `CAPS = 1

- First word of sentence is lowercase in input but should be uppercase
- Proper noun is lowercase in input but should be uppercase
- Acronym is in mixed case but should be all-caps (umid → UMID; edsa → EDSA)
- ALL CAPS

CAPS = 1, PUNC = 1
- Run-on segmentation introduces a new sentence boundary requiring capitalization 

CAPS = 0 
- Proper capitalization`,
    coding: "Binary 1/0",
    group: "Noise",
  },
  {
    key: "LAUGH_MARKER",
    category: "Sentence-Level Function Label",
    definition: "",
    examples: "haha, hdsjshhfhdhd, jkjkjkjkj, hshshshsh",
    decision: `LAUGH_MARKER = 1
- haha, hahaha, hehe, hihi
- Keysmash representing laughter (hskfhsdfgkj, jajaja)

LAUGH_MARKER = 1, ABBR = 1
- LOL, lmao, lmfao `,
    coding: "Binary 1/0",
    group: "Function",
  },
  {
    key: "REACTION_MARKER",
    category: "Sentence-Level Function Label",
    definition: "",
    examples: "diba, hala, uy, gagi, omg, wow, gago, tangina, noh, huhu, ackk, ay, lol, amf",
    decision: `REACTION_MARKER = 1
- lol, omg, hala, uy, jusko, hala, gagi, ackk, huhu (in reaction context)
- Filipino interjections: aba, aray, uy, hoy, naku
- Profanity used as reaction marker: amf, tangina, putangina, bwiset (when expressive, not literal)`,
    coding: "Binary 1/0",
    group: "Function",
  },
  {
    key: "HASHTAG",
    category: "Hashtag / Topic Marker",
    definition: "Hashtag used as topic, joke, campaign, or discourse marker.",
    examples: "#WalangPasok #studentlife",
    decision: "Preserve unless ethics policy says otherwise.",
    coding: "Optional 1/0",
    group: "Metadata",
  },
  {
    key: "MENTION",
    category: "Mention / Username",
    definition: "User handle or platform mention.",
    examples: "@username",
    decision: "Preserve or anonymize consistently.",
    coding: "Optional 1/0",
    group: "Metadata",
  },
  {
    key: "URL",
    category: "URL / Link",
    definition: "Web address, shortened link, platform reference.",
    examples: "bit.ly, fb.com, shopee.ph",
    decision: "Preserve or replace with <URL> consistently.",
    coding: "Optional 1/0",
    group: "Metadata",
  },
];

// Workflow flag — not one of the 14 categorical labels, handled as its own toggle.
export const NEEDS_REVIEW = {
  key: "NEEDS_REVIEW" as const,
  category: "Unclear / Ambiguous",
  definition: "Cannot be confidently normalized or labeled.",
  examples: "Ambiguous compressed text",
  decision: "Do not guess; send to adjudication.",
  coding: "Workflow flag",
};

// CRITICAL: this order MUST match the label columns in sheet 05_Reliability-Subset
// (each of the Annotator_A / _B / _C blocks) so the exported TSV pastes into the
// sheet and the in-sheet Krippendorff's α formula reads the correct columns.
export const LABEL_KEYS_IN_SHEET_ORDER: LabelKey[] = [
  "ABBR",
  "ORTHO",
  "ELONG",
  "CS",
  "EMOJI",
  "SLANG",
  "MORPH",
  "LAUGH_MARKER",
  "REACTION_MARKER",
  "CAPS",
  "PUNC",
  "HASHTAG",
  "MENTION",
  "URL",
];
