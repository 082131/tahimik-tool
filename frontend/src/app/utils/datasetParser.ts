export interface Token {
  text: string;
  kept: boolean;
}

export interface ParsedSentence {
  input: string;
  reference?: string;
}

export interface ParseResult {
  sentences: ParsedSentence[];
  detectedColumn: string;
  format: "jsonl" | "json" | "csv" | "text";
}

// ── Extract specific field from relaxed JSON or dictionary string ────────────

function extractNamedField(str: string, fieldName: string): string {
  const regex = new RegExp(
    `["']?${fieldName}["']?\\s*:\\s*(?:"((?:[^"\\\\]|\\\\.)*)"|'((?:[^'\\\\]|\\\\.)*)'|([^,}\\s]+))`,
    "i"
  );
  const match = str.match(regex);
  if (match) {
    const rawVal = match[1] ?? match[2] ?? match[3] ?? "";
    return rawVal
      .replace(/\\"/g, '"')
      .replace(/\\'/g, "'")
      .replace(/\\\\/g, "\\")
      .trim();
  }
  return "";
}

// ── Sanitize sentence: strip residual JSON wrappers or metadata tags ─────────

export function sanitizeSentence(raw: string): string {
  if (!raw) return "";
  const trimmed = raw.trim();

  // If the string is or contains a JSON-like object (e.g. {sentence_id: ..., input: ...})
  if (trimmed.startsWith("{") || trimmed.includes("sentence_id") || trimmed.includes('"input"') || trimmed.includes("'input'")) {
    try {
      const obj = JSON.parse(trimmed);
      const val = obj.input || obj.text || obj.sentence || obj.source || obj.raw || obj.noisy;
      if (val) return String(val).trim();
    } catch {
      // Relaxed regex extraction
      const extracted =
        extractNamedField(trimmed, "input") ||
        extractNamedField(trimmed, "text") ||
        extractNamedField(trimmed, "sentence") ||
        extractNamedField(trimmed, "source") ||
        extractNamedField(trimmed, "raw");
      if (extracted) return extracted;
    }

    // Strip leading `{sentence_id: ...}` if it's prepended
    const stripped = trimmed.replace(/^\{?sentence_id\s*:\s*[^,]+,?\s*/i, "").replace(/\}$/, "").trim();
    if (stripped.length > 0) return stripped;
  }

  // Remove surrounding unescaped quotes if any
  if ((trimmed.startsWith('"') && trimmed.endsWith('"')) || (trimmed.startsWith("'") && trimmed.endsWith("'"))) {
    return trimmed.slice(1, -1).trim();
  }

  return trimmed;
}

// ── Parse single line as JSONL object ────────────────────────────────────────

function parseJsonLine(line: string): ParsedSentence | null {
  const trimmed = line.trim();
  if (!trimmed) return null;

  // 1. Strict JSON parse
  try {
    const obj = JSON.parse(trimmed);
    const input = String(obj.input || obj.text || obj.sentence || obj.source || obj.raw || obj.noisy || "").trim();
    const reference = obj.target || obj.reference || obj.clean || obj.normalized || obj.gold
      ? String(obj.target || obj.reference || obj.clean || obj.normalized || obj.gold).trim()
      : undefined;
    if (input) return { input, reference };
  } catch {
    // 2. Relaxed regex extraction
    const input =
      extractNamedField(trimmed, "input") ||
      extractNamedField(trimmed, "text") ||
      extractNamedField(trimmed, "sentence") ||
      extractNamedField(trimmed, "raw");
    const reference =
      extractNamedField(trimmed, "target") ||
      extractNamedField(trimmed, "clean") ||
      extractNamedField(trimmed, "reference") ||
      extractNamedField(trimmed, "gold") ||
      undefined;

    if (input) return { input, reference: reference || undefined };
  }

  return null;
}

// ── Quote-Aware CSV Parser ───────────────────────────────────────────────────

export function parseCSV(text: string): string[][] {
  const rows: string[][] = [];
  let currentRow: string[] = [];
  let currentField = "";
  let inQuotes = false;

  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    const nextChar = text[i + 1];

    if (inQuotes) {
      if (char === '"') {
        if (nextChar === '"') {
          currentField += '"';
          i++; // skip escaped quote
        } else {
          inQuotes = false;
        }
      } else {
        currentField += char;
      }
    } else {
      if (char === '"') {
        inQuotes = true;
      } else if (char === "," || char === "\t" || char === ";") {
        currentRow.push(currentField.trim());
        currentField = "";
      } else if (char === "\r") {
        // ignore CR
      } else if (char === "\n") {
        currentRow.push(currentField.trim());
        if (currentRow.some((field) => field.length > 0)) {
          rows.push(currentRow);
        }
        currentRow = [];
        currentField = "";
      } else {
        currentField += char;
      }
    }
  }

  if (currentField || currentRow.length > 0) {
    currentRow.push(currentField.trim());
    if (currentRow.some((field) => field.length > 0)) {
      rows.push(currentRow);
    }
  }

  return rows;
}

// ── Column Identification for Tabular Data ───────────────────────────────────

const IGNORE_COLUMNS = new Set([
  "sentence_id", "sentenceid", "id", "idx", "index", "source_platform",
  "sourceplatform", "platform", "label", "category", "split", "date",
  "created_at", "user_id", "author", "batch", "source_id"
]);

const INPUT_KEYWORDS = [
  "input", "raw", "noisy", "noisy_text", "text", "sentence",
  "source", "informal", "unnormalized", "original", "tweet",
  "comment", "content", "data", "post", "utterance"
];

const REF_KEYWORDS = [
  "target", "reference", "clean", "normalized", "gold", "ground_truth", "output"
];

export function extractSentencesFromCSV(rows: string[][]): {
  sentences: ParsedSentence[];
  detectedColumn: string;
  hasHeader: boolean;
} {
  if (rows.length === 0) {
    return { sentences: [], detectedColumn: "text", hasHeader: false };
  }

  const firstRow = rows[0];
  let inputColIndex = -1;
  let refColIndex = -1;
  let hasHeader = false;

  // 1. Exact match with input keyword list (excluding ignored columns)
  firstRow.forEach((col, idx) => {
    const clean = col.toLowerCase().replace(/[^a-z0-9_]/g, "");
    if (IGNORE_COLUMNS.has(clean)) return;

    if (inputColIndex === -1 && INPUT_KEYWORDS.includes(clean)) {
      inputColIndex = idx;
      hasHeader = true;
    }
    if (refColIndex === -1 && REF_KEYWORDS.includes(clean)) {
      refColIndex = idx;
      hasHeader = true;
    }
  });

  // 2. Partial match in header names (excluding ignored columns)
  if (inputColIndex === -1) {
    firstRow.forEach((col, idx) => {
      const clean = col.toLowerCase();
      if (Array.from(IGNORE_COLUMNS).some((ign) => clean === ign || clean.startsWith(ign + "_"))) return;

      if (inputColIndex === -1 && INPUT_KEYWORDS.some((kw) => clean.includes(kw))) {
        inputColIndex = idx;
        hasHeader = true;
      }
    });
  }

  // 3. Fallback: column with greatest average character count (ignoring metadata columns)
  if (inputColIndex === -1) {
    if (firstRow.length === 1) {
      inputColIndex = 0;
      hasHeader = false;
    } else {
      const colLengths = new Array(firstRow.length).fill(0);
      const sampleRows = rows.slice(0, 20);
      sampleRows.forEach((row) => {
        row.forEach((cell, idx) => {
          if (idx < colLengths.length) {
            // Penalize columns with IDs or short codes
            const cellClean = cell.toLowerCase().trim();
            if (cellClean.startsWith("tah-") || cellClean.startsWith("id-") || /^[0-9]+$/.test(cellClean)) {
              colLengths[idx] -= 100;
            } else {
              colLengths[idx] += cell.length;
            }
          }
        });
      });
      inputColIndex = colLengths.indexOf(Math.max(...colLengths));
      hasHeader = false;
    }
  }

  const dataRows = hasHeader ? rows.slice(1) : rows;
  const colName = hasHeader ? firstRow[inputColIndex] : `Column ${inputColIndex + 1}`;

  const sentences: ParsedSentence[] = dataRows
    .map((row) => {
      const rawInput = (row[inputColIndex] ?? "").trim();
      const rawRef = refColIndex !== -1 ? (row[refColIndex] ?? "").trim() : undefined;
      const input = sanitizeSentence(rawInput);
      const reference = rawRef ? sanitizeSentence(rawRef) : undefined;
      return { input, reference };
    })
    .filter((item) => item.input.length > 0);

  return { sentences, detectedColumn: colName, hasHeader };
}

// ── Multi-Format Dataset Parser (JSONL, JSON, CSV, TSV, Plain Text) ──────────

export function parseDatasetFile(content: string): ParseResult {
  const stripped = content.replace(/^\uFEFF/, "").trim();

  // 1. JSON array format [{ input: "...", target: "..." }]
  if (stripped.startsWith("[") && stripped.endsWith("]")) {
    try {
      const arr = JSON.parse(stripped);
      if (Array.isArray(arr)) {
        const sentences: ParsedSentence[] = arr
          .map((obj: Record<string, unknown>) => {
            const rawIn = String(obj.input || obj.text || obj.sentence || obj.source || obj.raw || obj.noisy || "").trim();
            const rawRef = obj.target || obj.reference || obj.clean || obj.normalized || obj.gold
              ? String(obj.target || obj.reference || obj.clean || obj.normalized || obj.gold).trim()
              : undefined;
            return { input: sanitizeSentence(rawIn), reference: rawRef ? sanitizeSentence(rawRef) : undefined };
          })
          .filter((s) => s.input.length > 0);

        if (sentences.length > 0) {
          return { sentences, detectedColumn: "input", format: "json" };
        }
      }
    } catch {
      // Fall through
    }
  }

  // 2. JSONL format (one JSON object per line, e.g. manual_pairs.jsonl)
  const lines = stripped.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  const potentialJsonl = lines.filter((l) => l.startsWith("{") || l.includes('"sentence_id"') || l.includes('"input"') || l.includes("'input'"));

  if (potentialJsonl.length >= Math.max(1, Math.floor(lines.length * 0.4))) {
    const jsonlSentences: ParsedSentence[] = [];
    for (const line of lines) {
      const parsed = parseJsonLine(line);
      if (parsed && parsed.input) {
        jsonlSentences.push({
          input: sanitizeSentence(parsed.input),
          reference: parsed.reference ? sanitizeSentence(parsed.reference) : undefined,
        });
      }
    }
    if (jsonlSentences.length > 0) {
      return { sentences: jsonlSentences, detectedColumn: "input", format: "jsonl" };
    }
  }

  // 3. Tabular CSV / TSV / Semicolon-delimited
  const rawRows = parseCSV(stripped);
  const { sentences: csvSentences, detectedColumn } = extractSentencesFromCSV(rawRows);

  const cleaned = csvSentences
    .map((s) => ({
      input: sanitizeSentence(s.input),
      reference: s.reference ? sanitizeSentence(s.reference) : undefined,
    }))
    .filter((s) => s.input.length > 0);

  if (cleaned.length > 0) {
    return { sentences: cleaned, detectedColumn, format: "csv" };
  }

  // 4. Fallback: plain text line-by-line
  const plainSentences = lines
    .map((l) => sanitizeSentence(l))
    .filter((l) => l.length > 0)
    .map((l) => ({ input: l }));

  return { sentences: plainSentences, detectedColumn: "plain text", format: "text" };
}

// ── Estimate Noise Gating & Token Visualization ──────────────────────────────

export function estimateSentenceGating(rawInput: string): {
  noise: number;
  pruning: number;
  tokens: Token[];
} {
  const words = rawInput.trim().split(/\s+/).filter(Boolean);
  if (words.length === 0) {
    return { noise: 0.1, pruning: 10, tokens: [] };
  }

  const informalRegex =
    /(nlng|tlga|lodi|sanaol|nyo|dyan|besh|di|grabe|nkklk|ko|bro|lam|resto|daw|aq|ikw|d2|werpa|petmalu|bat|bakit|pano|san|sanba|koya|ateh|hahaha|hehehe|[0-9]+|[^\w\s]{2,}|([a-zA-Z])\1{2,})/i;

  let noisyCount = 0;
  const tokens: Token[] = words.map((w) => {
    const isInformal = informalRegex.test(w) || w.length <= 2 || /[0-9!@#$%^&*]/.test(w);
    if (isInformal) noisyCount++;
    return { text: w, kept: !isInformal };
  });

  const noise = Math.min(0.95, Math.max(0.08, +(noisyCount / words.length).toFixed(2)));
  const pruning = Math.round(noise * 75);

  return { noise, pruning, tokens };
}
