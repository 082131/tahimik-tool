// ============================================================================
// TAHIMIK — import-sentences Edge Function
// Accepts POST with CSV/TSV body, parses and upserts to sentences table.
// ============================================================================

import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

/** Sentinel values that indicate corrupt spreadsheet data */
const INVALID_MARKERS = ['#NAME?', '#REF!', '#N/A', '#VALUE!', '#NULL!', '#DIV/0!']

/** Check if a value is empty or contains a spreadsheet error marker */
function isInvalidValue(value: string): boolean {
  if (!value || value.trim() === '') return true
  const upper = value.trim().toUpperCase()
  return INVALID_MARKERS.some((marker) => upper.includes(marker))
}

/** Verify the caller is an admin by checking JWT + users table */
async function verifyAdmin(req: Request): Promise<{ isAdmin: boolean; error?: string }> {
  const authHeader = req.headers.get('Authorization')
  if (!authHeader) {
    return { isAdmin: false, error: 'Missing Authorization header' }
  }

  const token = authHeader.replace('Bearer ', '')

  // Create a client scoped to the user's JWT to get their identity
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

  // Use service-role client to check the users table (bypasses RLS)
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

serve(async (req: Request) => {
  // Handle CORS preflight
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    // Only accept POST
    if (req.method !== 'POST') {
      return new Response(
        JSON.stringify({ error: 'Method not allowed. Use POST.' }),
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

    // Parse query params
    const url = new URL(req.url)
    const delimiter = url.searchParams.get('delimiter') === 'tab' ? '\t' : ','
    const markReliability = url.searchParams.get('mark_reliability') === 'true'

    // Read body as text
    const body = await req.text()
    if (!body.trim()) {
      return new Response(
        JSON.stringify({ error: 'Empty request body' }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }

    // Parse CSV/TSV
    const lines = body.split(/\r?\n/).filter((line) => line.trim() !== '')
    if (lines.length < 2) {
      return new Response(
        JSON.stringify({ error: 'CSV/TSV must have a header row and at least one data row' }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }

    // Parse header to find column indices
    const headerCols = lines[0].split(delimiter).map((col) => col.trim())
    const idIdx = headerCols.findIndex(
      (col) => col.toLowerCase() === 'sentence_id' || col.toLowerCase() === 'sentenceid'
    )
    const textIdx = headerCols.findIndex(
      (col) =>
        col.toLowerCase() === 'raw_noisy_sentence' ||
        col.toLowerCase() === 'rawnoisysentence' ||
        col.toLowerCase() === 'noisy_text'
    )
    const platformIdx = headerCols.findIndex(
      (col) =>
        col.toLowerCase() === 'source_platform' ||
        col.toLowerCase() === 'sourceplatform' ||
        col.toLowerCase() === 'platform'
    )

    if (idIdx === -1 || textIdx === -1) {
      return new Response(
        JSON.stringify({
          error: 'CSV/TSV must contain "Sentence_ID" and "Raw_Noisy_Sentence" columns',
          found_columns: headerCols,
        }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }

    // Parse data rows
    const rows: Array<{
      id: string
      noisy_text: string
      source_platform: string | null
      in_reliability: boolean
    }> = []
    const errors: string[] = []
    let skipped = 0

    for (let i = 1; i < lines.length; i++) {
      const cols = lines[i].split(delimiter)
      const sentenceId = (cols[idIdx] ?? '').trim()
      const noisyText = (cols[textIdx] ?? '').trim()
      const sourcePlatform = platformIdx !== -1 ? (cols[platformIdx] ?? '').trim() || null : null

      // Skip rows with invalid/corrupt data
      if (isInvalidValue(sentenceId) || isInvalidValue(noisyText)) {
        skipped++
        continue
      }

      rows.push({
        id: sentenceId,
        noisy_text: noisyText,
        source_platform: sourcePlatform,
        in_reliability: markReliability,
      })
    }

    // Bulk upsert in batches of 500
    const supabaseAdmin = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
    )

    let imported = 0
    const BATCH_SIZE = 500

    for (let i = 0; i < rows.length; i += BATCH_SIZE) {
      const batch = rows.slice(i, i + BATCH_SIZE)

      const { data, error: upsertError } = await supabaseAdmin
        .from('sentences')
        .upsert(batch, { onConflict: 'id', ignoreDuplicates: false })
        .select('id')

      if (upsertError) {
        errors.push(`Batch ${Math.floor(i / BATCH_SIZE) + 1}: ${upsertError.message}`)
      } else {
        imported += data?.length ?? batch.length
      }
    }

    return new Response(
      JSON.stringify({
        imported,
        skipped,
        errors,
        total_rows_parsed: rows.length + skipped,
      }),
      {
        status: 200,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    )
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown error'
    return new Response(
      JSON.stringify({ error: `Internal server error: ${message}` }),
      { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    )
  }
})
