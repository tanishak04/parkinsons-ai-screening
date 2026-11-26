import { describe, expect, it, vi } from 'vitest'
import apiClient, { getAudioPredictionStatus, submitAudioPrediction } from './client.js'

describe('audio API readiness', () => {
  it('keeps audio prediction unavailable unless backend health explicitly confirms compatibility', async () => {
    vi.spyOn(apiClient, 'get').mockResolvedValue({ data: { status: 'ok', model_server: 'reachable' } })
    const post = vi.spyOn(apiClient, 'post')
    const status = await getAudioPredictionStatus()
    expect(status.available).toBe(false)

    await expect(submitAudioPrediction({ blob: new Blob(['sample'], { type: 'audio/wav' }) })).rejects.toThrow(/has not confirmed/i)
    expect(post).not.toHaveBeenCalled()
    vi.restoreAllMocks()
  })

  it('allows audio extraction for research while screening remains disabled', async () => {
    vi.spyOn(apiClient, 'get').mockResolvedValue({
      data: {
        status: 'ok',
        capabilities: {
          audio_feature_extraction: true,
          audio_input_compatible: false,
          accepted_audio_formats: ['audio/wav'],
          incompatibility_reason: 'Screening model validation not met',
        },
      },
    })
    const post = vi.spyOn(apiClient, 'post').mockResolvedValue({
      data: {
        status: 'screening_disabled',
        screening_available: false,
        extracted_feature_count: 16,
      },
    })

    const status = await getAudioPredictionStatus()
    expect(status.available).toBe(true)
    expect(status.screeningCompatible).toBe(false)
    expect(status.reason).toBe('Screening model validation not met')

    const res = await submitAudioPrediction({ blob: new Blob(['sample'], { type: 'audio/wav' }) })
    expect(res.status).toBe('screening_disabled')
    expect(post).toHaveBeenCalledTimes(1)
    vi.restoreAllMocks()
  })
})
