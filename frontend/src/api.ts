import type { ChatLine, FilmPanel } from './types'

// One place that talks to the server. Every component below stays ignorant of URLs.

interface AdvanceResponse {
  thread_id: string
  state: 'review' | 'done'
  draft?: string
  answer?: string
  trace?: { kind: string; tool?: string; text?: string; args?: Record<string, unknown> }[]
}

async function post<T>(url: string, body: unknown): Promise<T> {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!response.ok) {
    throw new Error(`${url} returned ${response.status}`)
  }
  return response.json() as Promise<T>
}

export function ask(question: string, threadId: string) {
  return post<AdvanceResponse>('/api/ask', { question, thread_id: threadId })
}

export function approve(threadId: string) {
  return post<AdvanceResponse>('/api/resume', {
    thread_id: threadId,
    action: 'approve',
  })
}

export async function panelFor(answer: string, exclude: string[] = []): Promise<FilmPanel[]> {
  const data = await post<{ films: FilmPanel[] }>('/api/panel', { answer, exclude })
  return data.films
}

// Titles the agent itself asked to keep out, read straight from its tool calls.
// Exact, not guessed at: if it sent exclude_title="Jurassic Park", the panel is
// not going to put Jurassic Park at the top of the list.
export function excludedTitles(trace: AdvanceResponse['trace']): string[] {
  if (!trace) return []
  return trace
    .filter((step) => step.kind === 'tool_call')
    .map((step) => step.args?.exclude_title)
    .filter((value): value is string => typeof value === 'string' && value.length > 0)
}

// What the agent asked for, ARGUMENTS INCLUDED.
//
// "called search_films" tells you nothing you can act on. The query the agent
// actually built is the thing that decides which films come back, and it is
// usually where a surprising result comes from. Same lesson as the critic: a
// count says something happened, only the content says what.
export function toolLines(trace: AdvanceResponse['trace']): ChatLine[] {
  if (!trace) return []
  return trace
    .filter((step) => step.kind === 'tool_call')
    .map((step) => {
      const args = Object.entries(step.args ?? {})
        .filter(([, value]) => value !== null && value !== '')
        .map(([key, value]) => `${key}=${JSON.stringify(value)}`)
        .join(', ')
      return { speaker: 'tool' as const, text: `${step.tool}(${args})` }
    })
}
