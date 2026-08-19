// ============================================================================
// TAHIMIK — assign-sentences Edge Function
// Generates normalization + labeling assignments or marks reliability subset.
// ============================================================================

import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

/** Fisher-Yates (Knuth) in-place shuffle */
function shuffleArray<T>(array: T[]): T[] {
  const arr = [...array]
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[arr[i], arr[j]] = [arr[j], arr[i]]
  }
  return arr
}

/** Verify the caller is an admin */
async function verifyAdmin(req: Request): Promise<{ isAdmin: boolean; userId?: string; error?: string }> {
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

  return { isAdmin: true, userId: user.id }
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

    // Parse request body
    const body = await req.json()
    const { action, sentence_ids } = body as {
      action: 'generate' | 'mark_reliability'
      sentence_ids?: string[]
    }

    if (!action || !['generate', 'mark_reliability'].includes(action)) {
      return new Response(
        JSON.stringify({ error: 'Invalid action. Must be "generate" or "mark_reliability".' }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }

    const supabaseAdmin = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
    )

    // -----------------------------------------------------------------------
    // Action: mark_reliability
    // -----------------------------------------------------------------------
    if (action === 'mark_reliability') {
      if (!sentence_ids || !Array.isArray(sentence_ids) || sentence_ids.length === 0) {
        return new Response(
          JSON.stringify({ error: 'sentence_ids array is required for mark_reliability' }),
          { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
        )
      }

      const { data, error: updateError } = await supabaseAdmin
        .from('sentences')
        .update({ in_reliability: true })
        .in('id', sentence_ids)
        .select('id')

      if (updateError) {
        return new Response(
          JSON.stringify({ error: `Failed to mark reliability: ${updateError.message}` }),
          { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
        )
      }

      return new Response(
        JSON.stringify({
          action: 'mark_reliability',
          marked_count: data?.length ?? 0,
          sentence_ids: data?.map((s: { id: string }) => s.id) ?? [],
        }),
        { status: 200, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }

    // -----------------------------------------------------------------------
    // Action: generate
    // -----------------------------------------------------------------------

    // Check for confirmation
    const url = new URL(req.url)
    const confirm = url.searchParams.get('confirm')

    // Delete existing assignments if confirmed
    if (confirm === 'true') {
      const { error: deleteError } = await supabaseAdmin
        .from('assignments')
        .delete()
        .neq('id', '00000000-0000-0000-0000-000000000000') // delete all (neq a non-existent id)

      if (deleteError) {
        return new Response(
          JSON.stringify({ error: `Failed to delete existing assignments: ${deleteError.message}` }),
          { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
        )
      }
    } else {
      // Check if assignments already exist
      const { count } = await supabaseAdmin
        .from('assignments')
        .select('id', { count: 'exact', head: true })

      if (count && count > 0) {
        return new Response(
          JSON.stringify({
            error: 'Assignments already exist. Add ?confirm=true to delete and regenerate.',
            existing_count: count,
          }),
          { status: 409, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
        )
      }
    }

    // Step 1: Fetch all active annotators
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

    if (!annotators || annotators.length !== 3) {
      return new Response(
        JSON.stringify({
          error: `Expected exactly 3 active annotators, found ${annotators?.length ?? 0}`,
          annotators: annotators?.map((a: { id: string; name: string }) => a.name) ?? [],
        }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }

    // Step 2: Fetch all sentences
    const { data: allSentences, error: sentenceError } = await supabaseAdmin
      .from('sentences')
      .select('id, in_reliability')
      .order('id')

    if (sentenceError) {
      return new Response(
        JSON.stringify({ error: `Failed to fetch sentences: ${sentenceError.message}` }),
        { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }

    if (!allSentences || allSentences.length === 0) {
      return new Response(
        JSON.stringify({ error: 'No sentences found in the database' }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }

    // Step 3: Shuffle sentences (Fisher-Yates)
    const shuffled = shuffleArray(allSentences)

    // Step 4: Split into 3 disjoint slices of equal size
    const totalSentences = shuffled.length
    const sliceSize = Math.floor(totalSentences / 3)
    const remainder = totalSentences % 3

    // Distribute remainder across first slices
    const sliceSizes = [sliceSize, sliceSize, sliceSize]
    for (let i = 0; i < remainder; i++) {
      sliceSizes[i]++
    }

    const slices: Array<typeof shuffled> = []
    let offset = 0
    for (let i = 0; i < 3; i++) {
      slices.push(shuffled.slice(offset, offset + sliceSizes[i]))
      offset += sliceSizes[i]
    }

    // Step 5: Create normalization assignments
    const normAssignments: Array<{
      sentence_id: string
      user_id: string
      task: string
      status: string
    }> = []

    for (let i = 0; i < 3; i++) {
      for (const sentence of slices[i]) {
        normAssignments.push({
          sentence_id: sentence.id,
          user_id: annotators[i].id,
          task: 'normalization',
          status: 'pending',
        })
      }
    }

    // Step 6: Fetch reliability sentences
    const reliabilitySentences = allSentences.filter(
      (s: { id: string; in_reliability: boolean }) => s.in_reliability
    )

    // Step 7: Create labeling assignments for reliability sentences -> all 3 annotators
    const labelAssignments: Array<{
      sentence_id: string
      user_id: string
      task: string
      status: string
    }> = []

    for (const sentence of reliabilitySentences) {
      for (const annotator of annotators) {
        labelAssignments.push({
          sentence_id: sentence.id,
          user_id: annotator.id,
          task: 'labeling',
          status: 'pending',
        })
      }
    }

    // Insert all assignments in batches
    const allAssignments = [...normAssignments, ...labelAssignments]
    const BATCH_SIZE = 500
    let insertedCount = 0
    const insertErrors: string[] = []

    for (let i = 0; i < allAssignments.length; i += BATCH_SIZE) {
      const batch = allAssignments.slice(i, i + BATCH_SIZE)
      const { data, error: insertError } = await supabaseAdmin
        .from('assignments')
        .insert(batch)
        .select('id')

      if (insertError) {
        insertErrors.push(`Batch ${Math.floor(i / BATCH_SIZE) + 1}: ${insertError.message}`)
      } else {
        insertedCount += data?.length ?? batch.length
      }
    }

    // Step 8: Return summary
    return new Response(
      JSON.stringify({
        action: 'generate',
        total_sentences: totalSentences,
        slices: sliceSizes,
        annotators: annotators.map((a: { id: string; name: string }) => ({
          name: a.name,
          normalization_count: slices[annotators.indexOf(a)]?.length ?? 0,
        })),
        reliability_count: reliabilitySentences.length,
        normalization_assignments: normAssignments.length,
        labeling_assignments: labelAssignments.length,
        total_inserted: insertedCount,
        errors: insertErrors,
      }),
      { status: 200, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    )
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown error'
    return new Response(
      JSON.stringify({ error: `Internal server error: ${message}` }),
      { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    )
  }
})
