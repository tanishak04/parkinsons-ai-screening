import { useEffect, useState } from 'react'
import { AlertCircle, ArrowRight, CheckCircle2, Info, LockKeyhole, Send } from 'lucide-react'
import { Link, useNavigate } from 'react-router-dom'
import AudioRecorder from '../components/AudioRecorder.jsx'
import AudioUploader from '../components/AudioUploader.jsx'
import SectionEyebrow from '../components/SectionEyebrow.jsx'
import { getAudioPredictionStatus, submitAudioPrediction } from '../api/client.js'

function getMediaType(audio) {
  return audio?.file?.type || audio?.blob?.type || audio?.mimeType || ''
}

export default function ScreeningPage() {
  const navigate = useNavigate()
  const [audio, setAudio] = useState(null)
  const [source, setSource] = useState(null)
  const [capability, setCapability] = useState({ loading: true, available: false, acceptedFormats: [], reason: '' })
  const [submitting, setSubmitting] = useState(false)

  useEffect(() => {
    let current = true
    getAudioPredictionStatus().then((result) => {
      if (current) setCapability({ loading: false, ...result })
    })
    return () => { current = false }
  }, [])

  const setSelectedAudio = (kind, asset) => {
    setAudio(asset)
    setSource(kind)
  }

  const clearAudio = () => {
    setAudio(null)
    setSource(null)
  }

  const formatCompatible = capability.acceptedFormats?.includes(getMediaType(audio))
  const canSubmit = Boolean(audio && capability.available && formatCompatible && !submitting)

  const handleSubmit = async () => {
    setSubmitting(true)
    navigate('/results', { state: { status: 'loading' } })
    try {
      const result = await submitAudioPrediction(audio)
      navigate('/results', { replace: true, state: { status: 'success', result } })
    } catch (error) {
      navigate('/results', { replace: true, state: { status: 'error', message: error.message || 'The audio request could not be sent.' } })
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <article className="screening-page audio-screening-page site-shell">
      <SectionEyebrow>VOICE RESEARCH / AUDIO CAPTURE</SectionEyebrow>
      <h1>A voice sample,<br /><em>for research.</em></h1>
      <p className="screening-intro">Explore audio capture for an early research prototype. Nothing recorded here is a diagnosis or a substitute for clinical care.</p>

      <section className="recording-instructions" aria-labelledby="instruction-title">
        <span className="instruction-icon"><Info size={18} /></span>
        <div>
          <h2 id="instruction-title">Before you record</h2>
          <p>Choose a quiet environment and record a sustained vowel sound, such as “ahhh”. Speak at a comfortable volume. This prototype does not provide a medically validated recording protocol.</p>
        </div>
        <span className="instruction-tag">AUDIO CAPTURE ONLY</span>
      </section>

      <div className="capture-heading">
        <div><span className="capture-kicker">YOUR RECORDING</span><h2>Choose one way to add audio</h2></div>
        <span className="capture-private"><LockKeyhole size={13} /> No recording is sent until you submit</span>
      </div>

      <div className="capture-workspace">
        <div className="capture-options">
          {source !== 'upload' && <AudioRecorder
            onRecordingStart={() => {
              setAudio(null)
            }}
            onAudioReady={(asset) => setSelectedAudio('recording', asset)}
            onRemove={clearAudio}
          />}
          {(source !== 'recording' || !audio) && <AudioUploader
            selectedAudio={source === 'upload' ? audio : null}
            onAudioReady={(asset) => setSelectedAudio('upload', asset)}
            onRemove={clearAudio}
          />}
        </div>

        <aside className="processing-panel" aria-labelledby="processing-title">
          <div className="processing-panel-top"><span className="processing-icon"><Info size={17} /></span><span className="processing-label">PROCESSING STATUS</span></div>
          <h2 id="processing-title">
            {capability.loading
              ? 'Checking backend support'
              : capability.available
                ? (capability.screeningCompatible ? 'Audio screening available' : 'Research extraction active · Screening disabled')
                : 'Audio processing not connected'}
          </h2>
          <p>
            {capability.loading
              ? 'Checking whether the backend confirms audio processing support.'
              : capability.available
                ? (capability.screeningCompatible
                    ? 'The backend reports live screening model compatibility. You can submit a supported recording.'
                    : 'Acoustic feature extraction is active for research inspection. Automated disease screening is disabled following validation review (model did not meet project targets).')
                : 'The current backend accepts numerical features only. It has not confirmed audio extraction compatible with the model.'}
          </p>
          <div className={`processing-status-line${capability.available ? ' processing-ready' : ''}`}>
            {capability.available ? <CheckCircle2 size={15} /> : <AlertCircle size={15} />}
            <span>
              {capability.loading
                ? 'Checking capability'
                : capability.available
                  ? (capability.screeningCompatible ? 'Backend capability confirmed' : 'Research extraction ready · Automated screening disabled')
                  : 'Waiting for compatible backend support'}
            </span>
          </div>
          {audio && capability.available && !formatCompatible && <p className="format-warning" role="status">The backend has not declared support for this recording format ({getMediaType(audio) || 'unknown'}).</p>}
          <button className="button button-ink submit-audio-button" type="button" disabled={!canSubmit} onClick={handleSubmit}>
            <Send size={15} />{submitting ? 'Analyzing audio…' : (capability.screeningCompatible ? 'Submit audio for processing' : 'Submit sample for research analysis')}
          </button>
          {!capability.available && <p className="submit-disabled-note">Submission stays disabled until the backend explicitly confirms audio extraction and accepted formats.</p>}
        </aside>
      </div>

      <div className="audio-prototype-note">
        <span>RESEARCH PROTOTYPE</span>
        <p>Capturing or previewing audio does not produce a prediction. The connected mock model accepts 22 numerical acoustic features and cannot analyze audio files.</p>
        <Link to="/methodology" className="text-link">Read the methodology <ArrowRight size={14} /></Link>
      </div>
      <p className="screening-foot">If you have health concerns, please speak with a qualified healthcare professional.</p>
    </article>
  )
}
