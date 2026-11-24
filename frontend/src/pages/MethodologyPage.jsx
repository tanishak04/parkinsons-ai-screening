import { ArrowUpRight } from 'lucide-react'
import SectionEyebrow from '../components/SectionEyebrow.jsx'

const steps = [
  { number: '01', title: 'Acoustic measurements', body: 'The existing tabular workflow uses 22 precomputed features, including measures related to fundamental frequency, jitter, shimmer, noise, and nonlinear dynamics.' },
  { number: '02', title: 'Exploratory modelling', body: 'The repository includes a baseline workflow for precomputed UCI voice features and a separate, in-progress workflow for extracting features from recordings.' },
  { number: '03', title: 'Evaluation and limits', body: 'A small participant-grouped baseline evaluation is recorded in the repository. It is preliminary and does not establish clinical validity or suitability for individual screening.' },
]

export default function MethodologyPage() {
  return (
    <article className="editorial-page methodology-page site-shell">
      <SectionEyebrow>RESEARCH / METHODOLOGY</SectionEyebrow>
      <h1>How we’re<br /><em>approaching it.</em></h1>
      <p className="method-intro">An overview of the current software and research workflows. The application accepts precomputed features in its current integration; it does not yet process uploaded audio.</p>
      <div className="method-steps">
        {steps.map((step) => <section className="method-step" key={step.number}>
          <span className="method-number">{step.number}</span>
          <h2>{step.title}</h2>
          <p>{step.body}</p>
        </section>)}
      </div>
      <div className="source-line"><span>DATASET REFERENCE</span><a href="https://archive.ics.uci.edu/dataset/174/parkinsons" target="_blank" rel="noreferrer">UCI Parkinson’s Disease dataset <ArrowUpRight size={14} /></a></div>
      <p className="method-caution">The current mock model produces simulated responses. Baseline results in the repository are research-only and should not be used to assess an individual’s health.</p>
    </article>
  )
}

