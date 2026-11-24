import { Activity, ArrowUpRight } from 'lucide-react'
import { Link } from 'react-router-dom'

export default function Footer() {
  return (
    <footer className="site-footer">
      <div className="site-shell footer-main">
        <div>
          <Link to="/" className="brand footer-brand" aria-label="Parkinson’s AI Screening System home">
            <span className="brand-mark" aria-hidden="true"><Activity size={18} strokeWidth={2} /></span>
            <span className="brand-name">Parkinson’s AI</span>
          </Link>
          <p className="footer-description">Screening research prototype · Voice & movement</p>
        </div>
        <div className="footer-links">
          <div className="footer-link-group">
            <span className="footer-label">Explore</span>
            <Link to="/screening">Screening overview</Link>
            <Link to="/methodology">Methodology</Link>
          </div>
          <div className="footer-link-group">
            <span className="footer-label">Project</span>
            <Link to="/about">About this research</Link>
            <a href="https://archive.ics.uci.edu/dataset/174/parkinsons" target="_blank" rel="noreferrer">
              UCI dataset <ArrowUpRight size={13} aria-hidden="true" />
            </a>
          </div>
        </div>
      </div>
      <div className="site-shell footer-bottom">
        <span>© 2026 Parkinson’s AI Research Prototype</span>
        <span>Experimental research software · Not a medical device</span>
      </div>
    </footer>
  )
}

