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
})
