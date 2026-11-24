import { ArrowRight } from 'lucide-react'
import { Link } from 'react-router-dom'

export default function ResearchNote() {
  return (
    <section className="research-note-section">
      <div className="site-shell research-note">
        <div className="research-note-label"><span className="note-square" />A note on the model</div>
        <div className="research-note-copy">
          <h2>Today, the model is a placeholder.</h2>
          <p>The connected service currently returns simulated responses for development. It has not been clinically validated. We make that visible because research tools should be clear about what they can—and cannot—say.</p>
          <Link to="/methodology" className="text-link">Read the methodology <ArrowRight size={15} /></Link>
        </div>
      </div>
    </section>
  )
}

