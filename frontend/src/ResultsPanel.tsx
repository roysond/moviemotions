import type { FilmPanel, Offer } from './types'

// Bands are drawn in this order and never mixed. A one-off $3.99 rental and a
// $12.99 monthly subscription are not the same kind of cost, so they are not
// ranked against each other. The server sorts; this file only groups.
const BAND_ORDER = ['free', 'subscription', 'rent', 'buy', 'needs_tv_provider'] as const

function groupByBand(offers: Offer[]) {
  return BAND_ORDER
    .map((band) => ({
      band,
      label: offers.find((offer) => offer.band === band)?.band_label ?? band,
      items: offers.filter((offer) => offer.band === band),
    }))
    .filter((group) => group.items.length > 0)
}

function OfferRow({ offer }: { offer: Offer }) {
  return (
    <div className="offer">
      {offer.logo_url
        ? <img src={offer.logo_url} alt="" />
        : <span className="logo-gap" />}
      <span className="oname">
        {offer.display}
        {offer.resold_from && <span className="via"> via {offer.resold_from}</span>}
      </span>
      <span className={`price p-${offer.band}`}>
        {offer.price_text}
        {/* an unverified price is marked, never silently shown as fact */}
        {!offer.verified && <span className="q" title={offer.note}>?</span>}
      </span>
    </div>
  )
}

function FilmRow({ film }: { film: FilmPanel }) {
  return (
    <article className="film">
      <div className="left">
        {film.poster_url
          ? <img className="poster" src={film.poster_url} alt={`${film.title} poster`} />
          : <div className="poster empty" />}
        <div>
          <h3>{film.title}</h3>
          <div className="meta">
            {film.year}
            {film.runtime_minutes ? ` · ${film.runtime_minutes} min` : ''}
          </div>
          <ul className="why">
            {film.reasons.map((reason, index) => <li key={index}>{reason}</li>)}
          </ul>
        </div>
      </div>

      <div className="offers">
        {/* "no listing held" and "not available" are different sentences */}
        {!film.has_listing && (
          <div className="nolisting">
            No {film.region} availability data held for this film — that is not the
            same as it being unavailable.
          </div>
        )}

        {groupByBand(film.offers).map((group) => (
          <div key={group.band}>
            <div className="band">{group.label}</div>
            {group.items.map((offer, index) => <OfferRow key={index} offer={offer} />)}
          </div>
        ))}

        {film.link && (
          <div className="verify">
            checked {film.checked_on} ({film.stale_days}d ago) · rights change weekly ·{' '}
            <a href={film.link} target="_blank" rel="noreferrer">verify ↗</a>
          </div>
        )}
      </div>
    </article>
  )
}

export default function ResultsPanel({ films, busy }: { films: FilmPanel[]; busy: boolean }) {
  return (
    <section className="results">
      <div className="rhead">
        <h2>Top picks · where to watch</h2>
        {films.length > 0 && <span className="stamp">{films[0].region}</span>}
      </div>

      {films.length === 0 && (
        <div className="blank">
          {busy ? 'Working…' : 'Ask for something and the films will appear here.'}
        </div>
      )}

      {films.map((film) => <FilmRow key={film.title} film={film} />)}
    </section>
  )
}
