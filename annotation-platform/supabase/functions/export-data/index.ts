// ============================================================================
// TAHIMIK — export-data Edge Function
// Exports normalization or labeling data as TSV downloads.
// ============================================================================

import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

/** The 14 label columns in the EXACT order for the 05_Reliability-Subset export */
const LABEL_ORDER = [
  'ABBR',
  'ORTHO',
  'ELONG',
  'CS',
  'EMOJI',
  'SLANG',
  'MORPH',
  'LAUGH_MARKER',
  'REACTION_MARKER',
  'CAPS',
  'PUNC',
  'HASHTAG',
  'MENTION',
  'URL',
] as const

/** Verify the caller is an admin */
async function verifyAdmin(req: Request): Promise<{ isAdmin: boolean; error?: string }> {
  const authHeader = req.headers.get('Authorization')
  if (!authHeader) {
    return { isAdmin: false, error: 'Missing Authorization header' }
  }

  const token = authHeader.replace('Bearer ', '')

  const supabaseUser = createClient(
    Deno.env.get('SUPABASE_URL') ?? '',
    Deno.env.get('SUPABASE_ANON_KEY') ?? '',
    { global: { headers: { Authorization: `Bearer ${token}` } } }
  )

  const {
    data: { user },
    error: userError,
  } = await supabaseUser.auth.getUser()

  if (userError || !user) {
    return { isAdmin: false, error: `Authentication failed: ${userError?.message ?? 'No user found'}` }
  }

  const supabaseAdmin = createClient(
    Deno.env.get('SUPABASE_URL') ?? '',
    Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
  )

  const { data: profile, error: profileError } = await supabaseAdmin
    .from('users')
    .select('role')
    .eq('id', user.id)
    .single()

  if (profileError || !profile) {
    return { isAdmin: false, error: `User profile not found: ${profileError?.message ?? 'Unknown'}` }
  }

  if (profile.role !== 'admin') {
    return { isAdmin: false, error: 'Forbidden: admin role required' }
  }

  return { isAdmin: true }
}

/** Escape a TSV cell value (handle tabs and newlines) */
function escapeTsv(value: string | number | null | undefined): string {
  if (value === null || value === undefined) return ''
  const str = String(value)
  // If the value contains a tab or newline, wrap in quotes and escape internal quotes
  if (str.includes('\t') || str.includes('\n') || str.includes('\r') || str.includes('"')) {
    return `"${str.replace(/"/g, '""')}"`
  }
  return str
}

serve(async (req: Request) => {
  // Handle CORS preflight
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    // Only accept GET
    if (req.method !== 'GET') {
      return new Response(
        JSON.stringify({ error: 'Method not allowed. Use GET.' }),
        { status: 405, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }

    // Verify admin role
    const { isAdmin, error: authError } = await verifyAdmin(req)
    if (!isAdmin) {
      return new Response(
        JSON.stringify({ error: authError }),
        { status: 403, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }

    const url = new URL(req.url)
    const task = url.searchParams.get('task')

    if (!task || !['normalization', 'labeling'].includes(task)) {
      return new Response(
        JSON.stringify({ error: 'Query param "task" must be "normalization" or "labeling".' }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }

    const supabaseAdmin = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
    )

    // -----------------------------------------------------------------------
    // Export: Normalization
    // -----------------------------------------------------------------------
    if (task === 'normalization') {
      // Fetch all normalizations joined with sentences
      const { data: normalizations, error: fetchError } = await supabaseAdmin
        .from('normalizations')
        .select(`
          sentence_id,
          normalized_text,
          needs_review,
          notes,
          sentences!inner (
            id,
            noisy_text
          )
        `)
        .order('sentence_id')

      if (fetchError) {
        return new Response(
          JSON.stringify({ error: `Failed to fetch normalizations: ${fetchError.message}` }),
          { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
        )
      }

      // Build TSV
      const header = ['Sentence_ID', 'Raw_Noisy_Sentence', 'Normalized_Sentence', 'NEEDS_REVIEW', 'Notes']
      const rows: string[] = [header.join('\t')]

      for (const norm of normalizations ?? []) {
        const sentence = norm.sentences as unknown as { id: string; noisy_text: string }
        rows.push(
          [
            escapeTsv(norm.sentence_id),
            escapeTsv(sentence?.noisy_text ?? ''),
            escapeTsv(norm.normalized_text),
            norm.needs_review ? '1' : '0',
            escapeTsv(norm.notes),
          ].join('\t')
        )
      }

      const tsv = rows.join('\n')
      const filename = `tahimik_normalizations_${new Date().toISOString().slice(0, 10)}.tsv`

      return new Response(tsv, {
        status: 200,
        headers: {
          ...corsHeaders,
          'Content-Type': 'text/tab-separated-values; charset=utf-8',
          'Content-Disposition': `attachment; filename="${filename}"`,
        },
      })
    }

    // -----------------------------------------------------------------------
    // Export: Labeling (05_Reliability-Subset wide format)
    // -----------------------------------------------------------------------

    // Fetch all labelings for reliability sentences
    const { data: labelings, error: labelError } = await supabaseAdmin
      .from('labelings')
      .select(`
        sentence_id,
        user_id,
        labels,
        needs_review,
        sentences!inner (
          id,
          noisy_text,
          in_reliability
        )
      `)
      .eq('sentences.in_reliability', true)
      .order('sentence_id')

    if (labelError) {
      return new Response(
        JSON.stringify({ error: `Failed to fetch labelings: ${labelError.message}` }),
        { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }

    // Fetch annotator names for display
    const { data: annotators, error: annotatorError } = await supabaseAdmin
      .from('users')
      .select('id, name')
      .eq('role', 'annotator')
      .eq('status', 'active')
      .order('name')

    if (annotatorError) {
      return new Response(
        JSON.stringify({ error: `Failed to fetch annotators: ${annotatorError.message}` }),
        { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }

    const annotatorMap = new Map<string, string>()
    for (const a of annotators ?? []) {
      annotatorMap.set(a.id, a.name)
    }

    // Group labelings by sentence_id
    const sentenceGroups = new Map<
      string,
      {
        noisy_text: string
        entries: Array<{
          user_id: string
          labels: Record<string, number>
          needs_review: boolean
        }>
      }
    >()

    for (const labeling of labelings ?? []) {
      const sentence = labeling.sentences as unknown as {
        id: string
        noisy_text: string
        in_reliability: boolean
      }
      const sentenceId = labeling.sentence_id

      if (!sentenceGroups.has(sentenceId)) {
        sentenceGroups.set(sentenceId, {
          noisy_text: sentence?.noisy_text ?? '',
          entries: [],
        })
      }

      sentenceGroups.get(sentenceId)!.entries.push({
        user_id: labeling.user_id,
        labels: (labeling.labels as Record<string, number>) ?? {},
        needs_review: labeling.needs_review,
      })
    }

    // Build the wide-format TSV header
    // Reliability_ID | Sentence_ID | Raw_Noisy_Sentence
    // then 3 annotator blocks each: Annotator | Output | 14 labels | NEEDS_REVIEW
    // then: Resolution_Status | Notes

    const headerParts: string[] = ['Reliability_ID', 'Sentence_ID', 'Raw_Noisy_Sentence']
    for (let i = 1; i <= 3; i++) {
      headerParts.push(`Annotator_${i}`)
      headerParts.push(`Output_${i}`)
      for (const label of LABEL_ORDER) {
        headerParts.push(`${label}_${i}`)
      }
      headerParts.push(`NEEDS_REVIEW_${i}`)
    }
    headerParts.push('Resolution_Status', 'Notes')

    const tsvRows: string[] = [headerParts.join('\t')]

    // Sort sentence groups by ID and build rows
    const sortedSentenceIds = [...sentenceGroups.keys()].sort()
    let reliabilityId = 1

    for (const sentenceId of sortedSentenceIds) {
      const group = sentenceGroups.get(sentenceId)!
      const rowParts: string[] = [
        String(reliabilityId),
        escapeTsv(sentenceId),
        escapeTsv(group.noisy_text),
      ]

      // Pad entries to exactly 3 annotators (fill missing with empties)
      const entries = group.entries.slice(0, 3)
      while (entries.length < 3) {
        entries.push({ user_id: '', labels: {}, needs_review: false })
      }

      for (const entry of entries) {
        // Annotator name
        rowParts.push(escapeTsv(annotatorMap.get(entry.user_id) ?? ''))
        // Output (always empty)
        rowParts.push('')

        // CRITICAL: When NEEDS_REVIEW=1, emit EMPTY STRINGS for all 14 label columns
        if (entry.needs_review) {
          for (let j = 0; j < LABEL_ORDER.length; j++) {
            rowParts.push('')
          }
          rowParts.push('1')
        } else {
          for (const label of LABEL_ORDER) {
            rowParts.push(String(entry.labels[label] ?? 0))
          }
          rowParts.push('0')
        }
      }

      // Resolution_Status and Notes (always empty in export)
      rowParts.push('', '')

      tsvRows.push(rowParts.join('\t'))
      reliabilityId++
    }

    const tsv = tsvRows.join('\n')
    const filename = `tahimik_reliability_labeling_${new Date().toISOString().slice(0, 10)}.tsv`

    return new Response(tsv, {
      status: 200,
      headers: {
        ...corsHeaders,
        'Content-Type': 'text/tab-separated-values; charset=utf-8',
        'Content-Disposition': `attachment; filename="${filename}"`,
      },
    })
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown error'
    return new Response(
      JSON.stringify({ error: `Internal server error: ${message}` }),
      { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    )
  }
})
