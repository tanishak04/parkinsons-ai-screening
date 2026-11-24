import { Activity, AudioLines, Microscope } from 'lucide-react'
import SectionEyebrow from './SectionEyebrow.jsx'

const disciplines = [
  { icon: AudioLines, title: 'Acoustic features', text: 'Research measures such as pitch, jitter, shimmer, and harmonic-to-noise ratio.' },
  { icon: Activity, title: 'Careful interpretation', text: 'A model output is a research signal to investigate, not a conclusion about a person.' },
  { icon: Microscope, title: 'Transparent methods', text: 'The project documents its data, approach, and limitations as the work develops.' },
]

export default function FeatureSection() {
  return (
    <section className="section-block features-section">
      <div className="site-shell">
        <div className="section-intro">
          <div><SectionEyebrow>THE RESEARCH LENS</SectionEyebrow><h2>Voice is complex.<br /><em>So is the question.</em></h2></div>
          <p>Speech can reflect many things: a person’s health, environment, language, and recording conditions. This project explores a narrow research question while keeping those limits in view.</p>
        </div>
        <div className="discipline-list">
          {disciplines.map(({ icon: Icon, title, text }, index) => (
            <article className="discipline-item" key={title}>
              <span className="discipline-count">0{index + 1}</span>
              <span className="discipline-icon"><Icon size={19} strokeWidth={1.7} aria-hidden="true" /></span>
              <div><h3>{title}</h3><p>{text}</p></div>
            </article>
          ))}
        </div>
      </div>
    </section>
  )
}

