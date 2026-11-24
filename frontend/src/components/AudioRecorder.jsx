import { useEffect, useRef, useState } from 'react'
import { AlertCircle, Mic, Square, Trash2 } from 'lucide-react'

function formatDuration(milliseconds) {
  const totalSeconds = Math.floor(milliseconds / 1000)
  return `${String(Math.floor(totalSeconds / 60)).padStart(2, '0')}:${String(totalSeconds % 60).padStart(2, '0')}`
}

function getRecorderError(error) {
  if (error?.name === 'NotAllowedError' || error?.name === 'SecurityError') {
    return 'Microphone permission was denied. Allow microphone access in your browser settings and try again.'
  }
  if (error?.name === 'NotFoundError' || error?.name === 'DevicesNotFoundError') {
    return 'No microphone was found. Connect a microphone or upload a WAV file instead.'
  }
  if (error?.name === 'NotReadableError' || error?.name === 'TrackStartError') {
    return 'The microphone is unavailable. Close other apps using it and try again.'
  }
  return 'Microphone access could not be started. Check your device settings and try again.'
}

function stopTracks(stream) {
  stream?.getTracks().forEach((track) => track.stop())
}

export default function AudioRecorder({ onAudioReady, onRecordingStart, onRemove }) {
  const [state, setState] = useState('idle')
  const [error, setError] = useState('')
  const [elapsed, setElapsed] = useState(0)
  const [recording, setRecording] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const streamRef = useRef(null)
  const recorderRef = useRef(null)
  const chunksRef = useRef([])
  const startedAtRef = useRef(0)
  const timerRef = useRef(null)
  const activeRef = useRef(true)
  const failedRef = useRef(false)

  useEffect(() => {
    activeRef.current = true
    return () => {
      activeRef.current = false
      window.clearInterval(timerRef.current)
      const recorder = recorderRef.current
      if (recorder && recorder.state !== 'inactive') recorder.stop()
      stopTracks(streamRef.current)
    }
  }, [])

  useEffect(() => {
    if (!previewUrl) return undefined
    return () => URL.revokeObjectURL(previewUrl)
  }, [previewUrl])

  const clearPreview = () => {
    setPreviewUrl('')
  }

  const startRecording = async () => {
    setError('')
    clearPreview()
    setRecording(null)
    setElapsed(0)
    onRecordingStart?.()

    if (!navigator.mediaDevices?.getUserMedia || typeof window.MediaRecorder === 'undefined') {
      setState('error')
      setError('Audio recording is not supported in this browser. Try a recent browser or upload a WAV file.')
      return
    }

    setState('requesting')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      if (!activeRef.current) {
        stopTracks(stream)
        return
      }
      streamRef.current = stream
      const supportedTypes = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4', 'audio/ogg;codecs=opus']
      const mimeType = supportedTypes.find((type) => window.MediaRecorder.isTypeSupported?.(type))
      const recorder = mimeType ? new window.MediaRecorder(stream, { mimeType }) : new window.MediaRecorder(stream)
      recorderRef.current = recorder
      chunksRef.current = []
      failedRef.current = false
      recorder.ondataavailable = (event) => {
        if (event.data?.size) chunksRef.current.push(event.data)
      }
      recorder.onerror = () => {
        failedRef.current = true
        setError('Recording stopped because the browser encountered an audio error. Please try again.')
        setState('error')
        window.clearInterval(timerRef.current)
        stopTracks(streamRef.current)
      }
      recorder.onstop = () => {
        window.clearInterval(timerRef.current)
        stopTracks(streamRef.current)
        streamRef.current = null
        if (failedRef.current) return
        const blobType = recorder.mimeType || chunksRef.current[0]?.type || 'audio/webm'
        const blob = new Blob(chunksRef.current, { type: blobType })
        if (!activeRef.current) return
        if (!blob.size) {
          setError('No audio was captured. Check your microphone and record again.')
          setState('error')
          return
        }
        const url = URL.createObjectURL(blob)
        const finalDuration = Math.max(0, Date.now() - startedAtRef.current)
        setElapsed(finalDuration)
        setRecording({ blob, name: 'Voice recording', size: blob.size, duration: finalDuration / 1000, format: blobType })
        setPreviewUrl(url)
        setState('ready')
        onAudioReady?.({ blob, name: 'voice-recording', size: blob.size, duration: finalDuration / 1000, format: blobType, mimeType: blobType })
      }
      recorder.start()
      startedAtRef.current = Date.now()
      timerRef.current = window.setInterval(() => setElapsed(Date.now() - startedAtRef.current), 250)
      setState('recording')
    } catch (cause) {
      stopTracks(streamRef.current)
      streamRef.current = null
      if (!activeRef.current) return
      setState('error')
      setError(getRecorderError(cause))
    }
  }

  const stopRecording = () => {
    window.clearInterval(timerRef.current)
    if (recorderRef.current?.state === 'recording') {
      setElapsed(Date.now() - startedAtRef.current)
      setState('stopping')
      recorderRef.current.stop()
    }
  }

  const removeRecording = () => {
    clearPreview()
    setRecording(null)
    setElapsed(0)
    setError('')
    setState('idle')
    onRemove?.()
  }

  return (
    <section className="audio-choice" aria-labelledby="record-title">
      <div className="audio-choice-heading"><span className="audio-choice-number">01</span><div><h3 id="record-title">Record a voice sample</h3><p>Use your device microphone in a quiet place.</p></div></div>
      <div className={`recorder-console${state === 'recording' ? ' recorder-active' : ''}`}>
        <div className="recorder-status">
          <span className={`recorder-indicator recorder-indicator-${state}`} aria-hidden="true" />
          <span>{state === 'recording' ? 'Recording' : state === 'requesting' ? 'Waiting for microphone permission' : state === 'stopping' ? 'Preparing preview' : state === 'ready' ? 'Recording ready' : 'Microphone ready when you are'}</span>
          <time className="recording-clock" aria-label={`Recording duration ${formatDuration(elapsed)}`}>{formatDuration(elapsed)}</time>
        </div>
        {state === 'recording' ? (
          <button className="button button-stop" type="button" onClick={stopRecording}><Square size={15} fill="currentColor" /> Stop recording</button>
        ) : state === 'ready' ? (
          <div className="recorder-actions">
            <button className="button button-ink" type="button" onClick={startRecording}><Mic size={16} /> Record again</button>
            <button className="icon-button" type="button" onClick={removeRecording} aria-label="Delete recording"><Trash2 size={17} /></button>
          </div>
        ) : (
          <button className="button button-ink" type="button" onClick={startRecording} disabled={state === 'requesting' || state === 'stopping'}><Mic size={16} />{state === 'requesting' ? 'Requesting microphone…' : 'Start recording'}</button>
        )}
        {state === 'ready' && previewUrl && <audio className="audio-player recorder-player" controls src={previewUrl} aria-label="Preview recorded voice" />}
        {recording && <p className="recording-format-note">Captured as {recording.format || 'browser audio'} for preview. Processing is not enabled.</p>}
      </div>
      {error && <p className="audio-error" role="alert"><AlertCircle size={15} />{error}</p>}
    </section>
  )
}
