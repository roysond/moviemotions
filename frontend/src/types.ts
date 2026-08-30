// THE CONTRACT between api.py and this app, written down once.
//
// This is the whole reason for choosing TypeScript. If the API stops sending
// `price_text`, plain JavaScript renders an empty cell — and an empty cell in a
// price column reads as "free", which is a lie. TypeScript refuses to build.

export type Band =
  | 'free'
  | 'subscription'
  | 'rent'
  | 'buy'
  | 'needs_tv_provider'

export interface Offer {
  display: string
  band: Band
  band_label: string
  price_text: string        // always human-readable, including "price unknown"
  verified: boolean         // false = we could not confirm this from an official page
  note: string
  resold_from: string | null
  logo_url: string | null
}

export interface FilmPanel {
  title: string
  year: string
  runtime_minutes: number | null
  poster_url: string | null
  reasons: string[]         // the agent's OWN sentences, never our paraphrase
  has_listing: boolean
  region: string
  checked_on: string
  stale_days: number
  link: string | null
  offers: Offer[]
}

export type Speaker = 'you' | 'bot' | 'tool'

export interface ChatLine {
  speaker: Speaker
  text: string
}
