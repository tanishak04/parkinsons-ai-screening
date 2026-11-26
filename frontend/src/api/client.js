import axios from 'axios'

// The browser talks only to FastAPI. The ML server URL remains server-side.
const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
  timeout: 15000,
  headers: { 'Content-Type': 'application/json' },
})

export default apiClient

/**
 * Audio feature extraction is available for research inspection, while live
 * automated screening is explicitly reported as disabled when audio_input_compatible is false.
 */
export async function getAudioPredictionStatus() {
  try {
    const { data } = await apiClient.get('/health')
    const capabilities = data?.capabilities
    const extractionAvailable = capabilities?.audio_feature_extraction === true
      && Array.isArray(capabilities?.accepted_audio_formats)
      && capabilities.accepted_audio_formats.length > 0
    const screeningCompatible = capabilities?.audio_input_compatible === true
    return {
      available: extractionAvailable,
      screeningCompatible,
      acceptedFormats: extractionAvailable ? capabilities.accepted_audio_formats : [],
      reason: capabilities?.incompatibility_reason || (extractionAvailable ? null : 'The backend has not confirmed audio processing support.'),
    }
  } catch {
    return { available: false, screeningCompatible: false, acceptedFormats: [], reason: 'The backend could not be reached.' }
  }
}

/** Convert browser AudioBuffer to 16-bit PCM mono WAV Blob. */
function audioBufferToWav(buffer) {
  const numChannels = 1
  const sampleRate = buffer.sampleRate
  const channelData = buffer.getChannelData(0)
  const dataLength = channelData.length * 2
  const arrayBuffer = new ArrayBuffer(44 + dataLength)
  const view = new DataView(arrayBuffer)

  const writeString = (offset, string) => {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i))
    }
  }

  writeString(0, 'RIFF')
  view.setUint32(4, 36 + dataLength, true)
  writeString(8, 'WAVE')
  writeString(12, 'fmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true) // PCM format
  view.setUint16(22, numChannels, true)
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * numChannels * 2, true)
  view.setUint16(32, numChannels * 2, true)
  view.setUint16(34, 16, true) // 16-bit
  writeString(36, 'data')
  view.setUint32(40, dataLength, true)

  let offset = 44
  for (let i = 0; i < channelData.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, channelData[i]))
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true)
  }

  return new Blob([view], { type: 'audio/wav' })
}

/** Decode browser-captured audio (WebM, OGG, etc.) into uncompressed WAV if possible. */
async function ensureWavBlob(blob) {
  if (blob.type === 'audio/wav' || blob.type === 'audio/x-wav' || blob.type === 'audio/wave') {
    return blob
  }
  try {
    const AudioContextClass = typeof window !== 'undefined' ? (window.AudioContext || window.webkitAudioContext) : null
    if (!AudioContextClass) return blob
    const arrayBuffer = await blob.arrayBuffer()
    const audioCtx = new AudioContextClass()
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer)
    return audioBufferToWav(audioBuffer)
  } catch {
    return blob
  }
}

/** Submit multipart audio only after the backend explicitly advertises support. */
export async function submitAudioPrediction(audio) {
  const capability = await getAudioPredictionStatus()
  if (!capability.available) {
    throw new Error(capability.reason || 'Audio prediction is not available.')
  }
  const blob = audio.file || audio.blob
  const mimeType = blob?.type || audio.mimeType || ''
  const allowedFormats = capability.acceptedFormats
  if (!blob || !allowedFormats.includes(mimeType)) {
    throw new Error('This recording format is not supported by the audio processing service.')
  }
  const finalBlob = await ensureWavBlob(blob)
  const baseName = (audio.name || 'voice-recording').replace(/\.[^/.]+$/, '')
  const formData = new FormData()
  formData.append('file', finalBlob, `${baseName}.wav`)
  const { data } = await apiClient.post('/predict/audio', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}
