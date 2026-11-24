import axios from 'axios'

// The browser talks only to FastAPI. The ML server URL remains server-side.
const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
  timeout: 15000,
  headers: { 'Content-Type': 'application/json' },
})

export default apiClient

/**
 * Audio support is opt-in only when FastAPI explicitly confirms both extraction
 * support and compatibility with the configured model. The current backend
 * health response does not advertise these capabilities, so this stays false.
 */
export async function getAudioPredictionStatus() {
  try {
    const { data } = await apiClient.get('/health')
    const capabilities = data?.capabilities
    const available = capabilities?.audio_feature_extraction === true
      && capabilities?.audio_input_compatible === true
      && Array.isArray(capabilities?.accepted_audio_formats)
      && capabilities.accepted_audio_formats.length > 0
    return {
      available,
      acceptedFormats: available ? capabilities.accepted_audio_formats : [],
      reason: available
        ? null
        : 'The backend has not confirmed audio extraction, model compatibility, and accepted audio formats.',
    }
  } catch {
    return { available: false, acceptedFormats: [], reason: 'The backend could not be reached.' }
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
  const formData = new FormData()
  formData.append('file', blob, audio.name || 'voice-recording')
  const { data } = await apiClient.post('/predict/audio', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}
