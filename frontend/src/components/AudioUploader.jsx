import { useEffect, useRef, useState } from 'react'
import { AlertCircle, Check, FileAudio2, Trash2, Upload } from 'lucide-react'

export const AUDIO_VALIDATION = Object.freeze({
  maxFileSizeMB: 20,
  maxDurationSeconds: 60,
})

const WAVE_MIME_TYPES = new Set(['audio/wav', 'audio/wave', 'audio/x-wav', 'audio/vnd.wave'])

export async function readAudioDuration(file) {
  const objectUrl = URL.createObjectURL(file)
  try {
    return await new Promise((resolve, reject) => {
      const audio = new Audio()
      audio.preload = 'metadata'
      audio.onloadedmetadata = () => resolve(audio.duration)
      audio.onerror = () => reject(new Error('The audio duration could not be read.'))
      audio.src = objectUrl
    })
  } finally {
    URL.revokeObjectURL(objectUrl)
  }
}

async function hasWaveHeader(file) {
  const header = new Uint8Array(await file.slice(0, 12).arrayBuffer())
  const read = (start, end) => String.fromCharCode(...header.slice(start, end))
  return header.length >= 12 && read(0, 4) === 'RIFF' && read(8, 12) === 'WAVE'
}

export async function validateAudioFile(file, {
  maxFileSizeMB = AUDIO_VALIDATION.maxFileSizeMB,
  maxDurationSeconds = AUDIO_VALIDATION.maxDurationSeconds,
  getDuration = readAudioDuration,
} = {}) {
  if (!file) return { valid: false, error: 'Choose a WAV audio file to continue.' }
  if (!file.name.toLowerCase().endsWith('.wav') || (file.type && !WAVE_MIME_TYPES.has(file.type.toLowerCase()))) {
    return { valid: false, error: 'This file is not a supported WAV audio file.' }
  }
  if (file.size > maxFileSizeMB * 1024 * 1024) {
    return { valid: false, error: `The file is larger than the ${maxFileSizeMB} MB limit.` }
  }

  try {
    if (!(await hasWaveHeader(file))) {
      return { valid: false, error: 'The file does not appear to contain WAV audio.' }
    }
    const duration = await getDuration(file)
    if (!Number.isFinite(duration) || duration <= 0) {
      return { valid: false, error: 'The audio duration could not be verified.' }
    }
    if (duration > maxDurationSeconds) {
      return { valid: false, error: `The recording exceeds the ${maxDurationSeconds} second technical limit.` }
    }
    return { valid: true, duration }
  } catch {
    return { valid: false, error: 'This audio file could not be read. Try another WAV file.' }
  }
}

function formatDuration(seconds) {
  const wholeSeconds = Math.floor(seconds)
  return `${Math.floor(wholeSeconds / 60)}:${String(wholeSeconds % 60).padStart(2, '0')}`
}

function formatFileSize(bytes) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export default function AudioUploader({
  onAudioReady,
  onRemove,
  maxFileSizeMB = AUDIO_VALIDATION.maxFileSizeMB,
  maxDurationSeconds = AUDIO_VALIDATION.maxDurationSeconds,
  getDuration = readAudioDuration,
  selectedAudio,
}) {
  const inputRef = useRef(null)
  const activeRef = useRef(true)
  const [dragging, setDragging] = useState(false)
  const [error, setError] = useState('')
  const [checking, setChecking] = useState(false)
  const [previewUrl, setPreviewUrl] = useState('')
  const [localSelection, setLocalSelection] = useState(null)
  const displayedAudio = selectedAudio || localSelection

  useEffect(() => {
    activeRef.current = true
    return () => { activeRef.current = false }
  }, [])

  useEffect(() => {
    if (!previewUrl) return undefined
    return () => URL.revokeObjectURL(previewUrl)
  }, [previewUrl])

  const removeFile = () => {
    setError('')
    setPreviewUrl('')
    setLocalSelection(null)
    if (inputRef.current) inputRef.current.value = ''
    onRemove?.()
  }

  const processFile = async (file) => {
    setError('')
    setPreviewUrl('')
    if (!file) return
    setChecking(true)
    const result = await validateAudioFile(file, { maxFileSizeMB, maxDurationSeconds, getDuration })
    if (!activeRef.current) return
    setChecking(false)
    if (!result.valid) {
      setError(result.error)
      onRemove?.()
      if (inputRef.current) inputRef.current.value = ''
      return
    }
    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
    const asset = { file, name: file.name, size: file.size, duration: result.duration, format: 'WAV', mimeType: file.type || 'audio/wav' }
    setLocalSelection(asset)
    onAudioReady?.(asset)
  }

  return (
    <section className="audio-choice" aria-labelledby="upload-title">
      <div className="audio-choice-heading"><span className="audio-choice-number">02</span><div><h3 id="upload-title">Upload a WAV file</h3><p>Choose an existing recording from your device.</p></div></div>
      {displayedAudio ? (
        <div className="audio-file-preview">
          <div className="audio-file-meta">
            <span className="audio-file-icon"><FileAudio2 size={18} /></span>
            <div className="audio-file-name"><strong>{displayedAudio.name}</strong><span>{formatFileSize(displayedAudio.size)} <i>·</i> {formatDuration(displayedAudio.duration)} <i>·</i> WAV</span></div>
            <button className="icon-button" type="button" onClick={removeFile} aria-label="Remove uploaded audio"><Trash2 size={17} /></button>
          </div>
          {previewUrl && <audio className="audio-player" controls src={previewUrl} aria-label="Preview uploaded audio" />}
          <p className="audio-valid-note"><Check size={14} /> WAV file checked for format, size, and duration.</p>
        </div>
      ) : (
        <>
          <input
            ref={inputRef}
            className="visually-hidden"
            type="file"
            accept=".wav,audio/wav,audio/x-wav"
            aria-label="Choose WAV audio file"
            onChange={(event) => processFile(event.target.files?.[0])}
          />
          <div
            className={`audio-dropzone${dragging ? ' audio-dropzone-active' : ''}${error ? ' audio-dropzone-error' : ''}`}
            role="button"
            tabIndex={0}
            aria-label="Choose a WAV file or drop it here"
            aria-describedby={error ? 'upload-error' : 'upload-limits'}
            onClick={() => inputRef.current?.click()}
            onKeyDown={(event) => {
              if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault()
                inputRef.current?.click()
              }
            }}
            onDragOver={(event) => { event.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={(event) => {
              event.preventDefault()
              setDragging(false)
              processFile(event.dataTransfer.files?.[0])
            }}
          >
            <span className="dropzone-icon"><Upload size={19} /></span>
            <strong>{checking ? 'Checking audio…' : 'Choose a file or drop it here'}</strong>
            <span>WAV audio · Up to {maxFileSizeMB} MB · {maxDurationSeconds} seconds maximum</span>
            <span id="upload-limits" className="dropzone-limits">These are technical file limits, not a validated recording protocol.</span>
          </div>
          {error && <p className="audio-error" id="upload-error" role="alert"><AlertCircle size={15} />{error}</p>}
        </>
      )}
    </section>
  )
}
