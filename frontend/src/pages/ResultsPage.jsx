import { AlertCircle, AlertTriangle, ArrowLeft, ArrowRight, CheckCircle2, FlaskConical, Info, ShieldAlert } from 'lucide-react'
import { Link, useLocation } from 'react-router-dom'
import SectionEyebrow from '../components/SectionEyebrow.jsx'

function isProbability(value) {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1
}

function isPredictionResponse(value) {
  return value && typeof value === 'object'
    && (value.prediction === 0 || value.prediction === 1)
    && typeof value.label === 'string'
    && value.label.trim().length > 0
}

function isAudioDisabledResponse(value) {
  return value && typeof value === 'object'
    && (value.screening_available === false || value.status === 'screening_disabled' || value.status === 'validation_failed')
}

function percentage(value) {
  return new Intl.NumberFormat(undefined, { style: 'percent', maximumFractionDigits: 1 }).format(value)
}

function ResultStatus({ state, message }) {
  if (state === 'loading') {
    return <section className="result-state-panel" aria-live="polite" aria-busy="true">
      <span className="result-state-mark result-mark-loading" />
      <div><h2>Processing the research request</h2><p>The backend is handling the request. This may take a moment.</p></div>
    </section>
  }
  if (state === 'error') {
    return <section className="result-state-panel result-state-error" role="alert">
      <span className="result-state-icon"><AlertCircle size={19} /></span>
      <div><h2>We couldn’t complete the request</h2><p>{message || 'The backend did not return a usable result. Please try again later.'}</p></div>
    </section>
  }
  return <section className="result-state-panel">
    <span className="result-state-icon"><Info size={19} /></span>
    <div><h2>No result to display</h2><p>Results appear here only after a compatible backend returns a prediction or research response. Audio capture and preview alone do not generate a result.</p></div>
  </section>
}

function ProbabilityRow({ label, value, tone }) {
  if (!isProbability(value)) return null
  return (
    <div className="probability-row">
      <div className="probability-caption"><span>{label}</span><strong>{percentage(value)}</strong></div>
      <div className="probability-track" role="img" aria-label={`${label}: ${percentage(value)}`}>
        <span className={`probability-fill probability-fill-${tone}`} style={{ width: `${value * 100}%` }} />
      </div>
    </div>
  )
}

function ResearchStatusResult({ result }) {
  const features = result.extracted_features || {}
  const featureEntries = Object.entries(features)
  const disclaimer = result.disclaimer || 'Research status notification only; this output is not a medical diagnosis.'

  return (
    <>
      <div className="mock-result-banner research-disabled-banner" role="status">
        <ShieldAlert size={18} aria-hidden="true" />
        <div>
          <strong>AUTOMATED SCREENING DISABLED</strong>
          <span>Live audio classification is deactivated following validation review.</span>
        </div>
      </div>

      <div className="result-heading-row">
        <div>
          <SectionEyebrow>RESEARCH AUDIT · MODEL STATUS</SectionEyebrow>
          <h1>Screening Disabled</h1>
        </div>
        <span className="result-class-badge result-class-neutral">
          Validation Criteria Not Met
        </span>
      </div>

      <div className="result-overview">
        <section className="result-reading" aria-labelledby="status-reading-title">
          <p className="result-reading-label" id="status-reading-title">Model status</p>
          <p className="result-reading-value" style={{ fontSize: '1.4rem', color: '#854d0e' }}>
            Screening Deactivated
          </p>
          <div className="confidence-reading">
            <span>Validation outcome</span>
            <strong>Target: &ge;0.70 &middot; Achieved: 0.6667</strong>
            <p>Participant-level Leave-One-Group-Out (LOGO) cross-validation demonstrated insufficient generalization.</p>
          </div>
        </section>

        <dl className="model-details">
          <div><dt>Status code</dt><dd>{result.status || 'screening_disabled'}</dd></div>
          <div><dt>Features extracted</dt><dd>{result.extracted_feature_count || featureEntries.length || 0}</dd></div>
          <div><dt>Audio duration</dt><dd>{result.duration_seconds ? `${result.duration_seconds}s` : 'N/A'}</dd></div>
          <div><dt>Automated prediction</dt><dd>Disabled</dd></div>
        </dl>
      </div>

      <section className="research-guidance-panel" aria-labelledby="guidance-title">
        <h2 id="guidance-title" style={{ fontSize: '1.1rem', marginBottom: '0.75rem', fontWeight: 600 }}>
          Why is live screening unavailable?
        </h2>
        <ul style={{ margin: 0, paddingLeft: '1.25rem', lineHeight: 1.6, color: '#374151' }}>
          <li>
            <strong>Audio analysis is experimental:</strong> Acoustic vocal biomarkers extracted via consumer microphones in domestic environments are subject to acoustic reverberation and background noise.
          </li>
          <li>
            <strong>Validation criteria not met:</strong> In rigorous participant-level cross-validation, the experimental 16-feature model achieved a balanced accuracy of 0.6667 (below our project acceptance threshold of &ge;0.70) and exhibited high participant-level instability.
          </li>
          <li>
            <strong>Automated screening temporarily disabled:</strong> To prevent misleading false-positive or false-negative screening outcomes, no disease classification label (PARKINSONS or HEALTHY), risk probability, or confidence score is generated.
          </li>
          <li>
            <strong>Not a medical assessment:</strong> This software is a research engineering prototype and does not represent or replace clinical evaluation by a medical professional.
          </li>
        </ul>
      </section>

      {featureEntries.length > 0 && (
        <section className="probability-section" aria-labelledby="features-title" style={{ marginTop: '1.5rem' }}>
          <div className="probability-heading">
            <div>
              <SectionEyebrow>RESEARCH EXTRACTION</SectionEyebrow>
              <h2 id="features-title">Extracted Acoustic Biomarkers</h2>
            </div>
            <span>Extracted via Praat / Parselmouth (Research Inspection Only)</span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: '0.75rem', marginTop: '1rem' }}>
            {featureEntries.map(([name, val]) => (
              <div key={name} style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: '6px', padding: '0.5rem 0.75rem' }}>
                <span style={{ display: 'block', fontSize: '0.75rem', color: '#64748b', fontFamily: 'monospace' }}>{name}</span>
                <strong style={{ fontSize: '0.95rem', color: '#0f172a' }}>
                  {typeof val === 'number' ? (Math.abs(val) < 0.001 ? val.toExponential(4) : val.toFixed(4)) : String(val)}
                </strong>
              </div>
            ))}
          </div>
          <p className="probability-note" style={{ marginTop: '0.75rem' }}>
            <Info size={14} /> These measurements are raw acoustic parameters and carry no clinical or diagnostic interpretation.
          </p>
        </section>
      )}

      <div className="result-disclaimer" role="note">
        <strong>Research use only</strong>
        <span>{disclaimer}</span>
      </div>
    </>
  )
}

function PredictionResult({ result }) {
  const mock = result.mock === true || result._mock === true
  const confidenceProvided = isProbability(result.confidence)
  const label = (result.label || '').toUpperCase()
  const disclaimer = typeof result.disclaimer === 'string' && result.disclaimer.trim()
    ? result.disclaimer
    : 'Research prototype output only. This is not a medical diagnosis.'

  return (
    <>
      {mock && <div className="mock-result-banner" role="status">
        <FlaskConical size={17} aria-hidden="true" />
        <div><strong>MOCK / SIMULATED RESULT</strong><span>The current model is temporary. This output is not an assessment of your health.</span></div>
      </div>}

      <div className="result-heading-row">
        <div>
          <SectionEyebrow>{mock ? 'SIMULATED MODEL OUTPUT' : 'MODEL OUTPUT'}</SectionEyebrow>
          <h1>{label}</h1>
        </div>
        <span className={`result-class-badge${result.prediction === 1 ? ' result-class-positive' : ''}`}>
          Model label · {result.prediction}
        </span>
      </div>

      <div className="result-overview">
        <section className="result-reading" aria-labelledby="result-reading-title">
          <p className="result-reading-label" id="result-reading-title">Prediction label</p>
          <p className="result-reading-value">{label}</p>
          {confidenceProvided && <div className="confidence-reading">
            <span>Model confidence</span>
            <strong>{percentage(result.confidence)}</strong>
            <p>Confidence describes the model output; it is not medical certainty.</p>
          </div>}
        </section>

        <dl className="model-details">
          <div><dt>Model version</dt><dd>{typeof result.model_version === 'string' && result.model_version.trim() ? result.model_version : 'Not provided'}</dd></div>
          {Number.isInteger(result.feature_count) && <div><dt>Features received</dt><dd>{result.feature_count}</dd></div>}
          <div><dt>Output type</dt><dd>{mock ? 'Simulated' : 'Model response'}</dd></div>
        </dl>
      </div>

      {(isProbability(result.pd_probability) || isProbability(result.healthy_probability)) && <section className="probability-section" aria-labelledby="probability-title">
        <div className="probability-heading"><div><SectionEyebrow>RESPONSE VALUES</SectionEyebrow><h2 id="probability-title">Class probabilities</h2></div><span>As returned by the model</span></div>
        <div className="probability-list">
          <ProbabilityRow label="Parkinson’s class" value={result.pd_probability} tone="teal" />
          <ProbabilityRow label="Healthy class" value={result.healthy_probability} tone="green" />
        </div>
        <p className="probability-note"><Info size={14} />These are model probabilities, not an estimate of a person’s medical condition.</p>
      </section>}

      <div className="result-disclaimer" role="note"><strong>Research use only</strong><span>{disclaimer}</span></div>
    </>
  )
}

export default function ResultsPage() {
  const location = useLocation()
  const state = location.state || {}
  const result = state.result
  const isAudioDisabled = isAudioDisabledResponse(result)
  const isStandardPrediction = isPredictionResponse(result)
  const isValidResult = isAudioDisabled || isStandardPrediction

  const viewState = state.status === 'loading'
    ? 'loading'
    : state.status === 'error'
      ? 'error'
      : state.status === 'success'
        ? isValidResult ? 'success' : 'error'
        : 'empty'
  const stateMessage = state.status === 'success' && !isValidResult
    ? 'The backend response did not include a valid prediction or research payload.'
    : state.message

  return (
    <article className="results-page site-shell">
      <div className="results-topline">
        <SectionEyebrow>VOICE RESEARCH / RESULTS</SectionEyebrow>
        {viewState === 'success' && <span className="result-return-label"><CheckCircle2 size={14} /> Response received</span>}
      </div>

      {viewState === 'success'
        ? (isAudioDisabled ? <ResearchStatusResult result={result} /> : <PredictionResult result={result} />)
        : <>
          <h1>{viewState === 'loading' ? <>Request <em>in progress.</em></> : viewState === 'error' ? <>Request <em>interrupted.</em></> : <>Waiting for a<br /><em>backend response.</em></>}</h1>
          <ResultStatus state={viewState} message={stateMessage} />
        </>}

      <div className="results-actions">
        <Link className="button button-outline" to="/screening"><ArrowLeft size={15} /> Return to screening</Link>
        <Link className="button button-ink" to="/screening" state={{ startNewSession: true }}>Start another session <ArrowRight size={15} /></Link>
      </div>
    </article>
  )
}
