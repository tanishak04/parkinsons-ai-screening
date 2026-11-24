import { ArrowRight } from 'lucide-react'
import { Link } from 'react-router-dom'
import SectionEyebrow from '../components/SectionEyebrow.jsx'

export default function AboutPage() {
  return (
    <article className="editorial-page site-shell">
      <SectionEyebrow>ABOUT THE PROJECT</SectionEyebrow>
      <h1>Research before<br /><em>answers.</em></h1>
      <div className="editorial-columns">
        <p className="editorial-lede">Parkinson’s AI Screening System is an early-stage research prototype exploring whether patterns in voice recordings can support the study of Parkinson’s disease.</p>
        <div className="editorial-body">
          <p>Parkinson’s can affect speech and voice. Researchers study measurable acoustic characteristics to better understand those changes. This project brings together an existing tabular-feature model and a service architecture for future research.</p>
          <p>The current connected model is a mock service used to develop and evaluate the software integration. It does not make real predictions. Any future trained model would require careful independent validation before its outputs could be interpreted in a research or clinical setting.</p>
          <Link to="/methodology" className="text-link">Understand the approach <ArrowRight size={15} /></Link>
        </div>
      </div>
      <div className="editorial-callout"><strong>Scope</strong><span>Research and software exploration only. This application is not a medical device, diagnostic tool, or substitute for care from a qualified professional.</span></div>
    </article>
  )
}

