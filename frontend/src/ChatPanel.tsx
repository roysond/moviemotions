import { useState } from 'react'
import type { FormEvent } from 'react'
import type { ChatLine } from './types'

interface Props {
  lines: ChatLine[]
  draft: string | null
  busy: boolean
  error: string | null
  onSend: (question: string) => void
  onApprove: () => void
}

export default function ChatPanel({ lines, draft, busy, error, onSend, onApprove }: Props) {
  const [text, setText] = useState('')

  function submit(event: FormEvent) {
    event.preventDefault()
    const question = text.trim()
    if (!question || busy) return
    setText('')
    onSend(question)
  }

  return (
    <section className="chat">
      <h2>Chat</h2>

      <div className="stream">
        {lines.map((line, index) => (
          <div key={index} className={`msg ${line.speaker}`}>
            {line.text}
          </div>
        ))}

        {busy && <div className="msg tool">thinking…</div>}
        {error && <div className="msg error">{error}</div>}

        {draft !== null && (
          <div className="review">
            <div className="review-head">Ready to send — your call</div>
            <div className="review-body">{draft}</div>
            <button onClick={onApprove} disabled={busy}>Approve</button>
          </div>
        )}
      </div>

      <form className="composer" onSubmit={submit}>
        <input
          value={text}
          onChange={(event) => setText(event.target.value)}
          placeholder="Describe your mood…"
          disabled={busy}
        />
        <button type="submit" disabled={busy || !text.trim()}>Send</button>
      </form>
    </section>
  )
}
