import { Link } from 'react-router-dom'

export default function NotFoundPage() {
  return <section className="not-found site-shell"><p className="eyebrow">404 · PAGE NOT FOUND</p><h1>We couldn’t find<br /><em>that page.</em></h1><Link to="/" className="text-link">Return to home</Link></section>
}

