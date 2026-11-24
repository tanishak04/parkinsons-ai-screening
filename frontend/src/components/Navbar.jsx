import { useState } from 'react'
import { Activity, Menu, X } from 'lucide-react'
import { Link, NavLink } from 'react-router-dom'

const links = [
  { to: '/screening', label: 'Screening' },
  { to: '/methodology', label: 'Methodology' },
  { to: '/about', label: 'About' },
]

export default function Navbar() {
  const [open, setOpen] = useState(false)

  return (
    <header className="site-header">
      <div className="site-shell nav-inner">
        <Link to="/" className="brand" aria-label="Parkinson’s AI Screening System home" onClick={() => setOpen(false)}>
          <span className="brand-mark" aria-hidden="true"><Activity size={19} strokeWidth={2.1} /></span>
          <span className="brand-name">Parkinson’s AI</span>
          <span className="brand-divider" />
          <span className="brand-context">VOICE RESEARCH</span>
        </Link>

        <button
          className="menu-toggle"
          type="button"
          aria-label={open ? 'Close navigation menu' : 'Open navigation menu'}
          aria-expanded={open}
          aria-controls="primary-navigation"
          onClick={() => setOpen((value) => !value)}
        >
          {open ? <X size={21} /> : <Menu size={21} />}
        </button>

        <nav id="primary-navigation" className={`nav-links ${open ? 'nav-links-open' : ''}`} aria-label="Primary navigation">
          {links.map((link) => (
            <NavLink key={link.to} to={link.to} onClick={() => setOpen(false)} className={({ isActive }) => `nav-link${isActive ? ' nav-link-active' : ''}`}>
              {link.label}
            </NavLink>
          ))}
          <Link to="/screening" className="button button-small button-ink nav-cta" onClick={() => setOpen(false)}>
            Explore the project
          </Link>
        </nav>
      </div>
    </header>
  )
}

