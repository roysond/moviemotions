import { useState } from 'react'
import ChatPanel from './ChatPanel'
import ResultsPanel from './ResultsPanel'
import { approve, ask, excludedTitles, panelFor, toolLines } from './api'
import type { ChatLine, FilmPanel } from './types'

// Chat on the left at 1/3, results on the right at 2/3.
// All the state lives here; the two panels just draw what they are handed.

export default function App() {
  const [threadId] = useState(() => crypto.randomUUID())
  const [lines, setLines] = useState<ChatLine[]>([
    { speaker: 'bot', text: 'What are you in the mood for tonight?' },
  ])
  const [draft, setDraft] = useState<string | null>(null)   // waiting on you
  const [films, setFilms] = useState<FilmPanel[]>([])
  const [busy, setBusy] = useState(false)
  const [excluded, setExcluded] = useState<string[]>([])
  const [error, setError] = useState<string | null>(null)

  function say(line: ChatLine) {
    setLines((previous) => [...previous, line])
  }

  async function send(question: string) {
    setBusy(true)
    setError(null)
    say({ speaker: 'you', text: question })
    try {
      const result = await ask(question, threadId)
      toolLines(result.trace).forEach(say)
      const keepOut = excludedTitles(result.trace)
      setExcluded(keepOut)
      if (result.state === 'review') {
        // The films are already known — the pause is about SENDING the message,
        // not about discovering them. Making you approve before you can see the
        // posters gets the human-in-the-loop step backwards: you are reviewing
        // the wording, so you should be looking at the evidence while you do it.
        setDraft(result.draft ?? '')
        setFilms(await panelFor(result.draft ?? '', keepOut))
      } else {
        await finish(result.answer ?? '')
      }
    } catch (problem) {
      setError(String(problem))
    } finally {
      setBusy(false)
    }
  }

  // THE HUMAN IN THE LOOP. The graph paused and wrote itself to disk; nothing is
  // held open on the server. This resumes it with the same thread id.
  async function accept() {
    setBusy(true)
    setError(null)
    try {
      const result = await approve(threadId)
      setDraft(null)
      await finish(result.answer ?? draft ?? '')   // refresh: the text may have changed
    } catch (problem) {
      setError(String(problem))
    } finally {
      setBusy(false)
    }
  }

  async function finish(answer: string) {
    say({ speaker: 'bot', text: answer })
    setFilms(await panelFor(answer, excluded))
  }

  return (
    <div className="app">
      <ChatPanel
        lines={lines}
        draft={draft}
        busy={busy}
        error={error}
        onSend={send}
        onApprove={accept}
      />
      <ResultsPanel films={films} busy={busy} />
    </div>
  )
}
