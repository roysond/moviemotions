import type { ChatLine, FilmPanel } from './types'

// One place that talks to the server. Every component below stays ignorant of URLs.

interface AdvanceResponse {
  thread_id: string
  state: 'review' | 'done'
  draft?: string
  answer?: string
  trace?: { kind: string; tool?: string; text?: string }[]
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

export async function panelFor(answer: string): Promise<FilmPanel[]> {
  const data = await post<{ films: FilmPanel[] }>('/api/panel', { answer })
  return data.films
}

// Which tools the agent reached for, so the chat can show its working.
export function toolLines(trace: AdvanceResponse['trace']): ChatLine[] {
  if (!trace) return []
  return trace
    .filter((step) => step.kind === 'tool_call')
    .map((step) => ({ speaker: 'tool' as const, text: `called ${step.tool}` }))
}
