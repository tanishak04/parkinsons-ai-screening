import { ArrowRight, AudioLines, BookOpenText } from 'lucide-react'
import { Link } from 'react-router-dom'
import SectionEyebrow from './SectionEyebrow.jsx'

const topics = [
  { number: '01', title: 'Voice measures', text: 'Acoustic features studied in sustained-vowel recordings.' },
  { number: '02', title: 'Model exploration', text: 'An early machine-learning experiment, not a clinical assessment.' },
  { number: '03', title: 'Research context', text: 'Methods, limitations, and the questions still to be answered.' },
]

export default function HeroSection() {
  return (
    <section className="hero-section">
      <div className="site-shell hero-grid">
        <div className="hero-copy">
          <SectionEyebrow>VOICE · MOVEMENT · RESEARCH</SectionEyebrow>
          <h1>Listening for<br /><em>what’s changing.</em></h1>
          <p className="hero-lede">A research prototype exploring how acoustic voice features could contribute to the study of Parkinson’s disease.</p>
          <div className="hero-actions">
            <Link className="button button-ink" to="/methodology">Explore the research <ArrowRight size={16} /></Link>
            <Link className="text-link" to="/about"><BookOpenText size={16} /> About the project</Link>
          </div>
          <p className="hero-footnote"><span className="status-dot" />Experimental software · Not for diagnosis</p>
        </div>

        <aside className="research-panel" aria-label="Research overview">
          <div className="panel-topline">
            <span className="panel-index">FIELDNOTE 001</span>
            <span className="panel-mark"><AudioLines size={18} /></span>
          </div>
          <div className="panel-heading">
            <span className="panel-kicker">A question in progress</span>
            <h2>Can the voice<br />help us notice?</h2>
            <p>We’re examining established acoustic measures in published voice datasets and exploring their use in machine-learning research.</p>
          </div>
          <div className="topic-list">
            {topics.map((topic) => (
              <div className="topic-row" key={topic.number}>
                <span className="topic-number">{topic.number}</span>
                <div><h3>{topic.title}</h3><p>{topic.text}</p></div>
              </div>
            ))}
          </div>
          <div className="panel-note"><span className="note-rule" />No diagnostic claims. No clinical validation.</div>
        </aside>
      </div>
      <div className="hero-bottom-rule" />
    </section>
  )
}

