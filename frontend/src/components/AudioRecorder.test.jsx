// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import '@testing-library/jest-dom/vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import AudioRecorder from './AudioRecorder.jsx'

let recorderInstances
let tracks

class MockMediaRecorder {
  static isTypeSupported = () => true

  constructor(stream, options = {}) {
    this.stream = stream
    this.mimeType = options.mimeType || 'audio/webm'
    this.state = 'inactive'
    recorderInstances.push(this)
  }

  start() { this.state = 'recording' }

  stop() {
    this.state = 'inactive'
    this.ondataavailable?.({ data: new Blob(['sample voice'], { type: this.mimeType }) })
    this.onstop?.()
  }
}

function setMicrophone(getUserMedia = vi.fn().mockResolvedValue({ getTracks: () => tracks })) {
  Object.defineProperty(navigator, 'mediaDevices', {
    configurable: true,
    value: { getUserMedia },
  })
  return getUserMedia
}

describe('AudioRecorder', () => {
  beforeEach(() => {
    recorderInstances = []
    tracks = [{ stop: vi.fn() }]
    vi.stubGlobal('MediaRecorder', MockMediaRecorder)
    vi.stubGlobal('URL', {
      ...URL,
      createObjectURL: vi.fn(() => 'blob:voice-preview'),
      revokeObjectURL: vi.fn(),
    })
    setMicrophone()
  })

  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
  })

  it('requests permission, records, stops tracks, and exposes playback and re-record controls', async () => {
    const user = userEvent.setup()
    const onAudioReady = vi.fn()
    const getUserMedia = setMicrophone()
    render(<AudioRecorder onAudioReady={onAudioReady} />)

    await user.click(screen.getByRole('button', { name: /start recording/i }))
    expect(getUserMedia).toHaveBeenCalledWith({ audio: true })
    expect(await screen.findByText('Recording', { exact: true })).toBeTruthy()

    await user.click(screen.getByRole('button', { name: /stop recording/i }))
    expect(await screen.findByText('Recording ready')).toBeTruthy()
    expect(tracks[0].stop).toHaveBeenCalledOnce()
    expect(screen.getByLabelText('Preview recorded voice')).toBeTruthy()
    expect(onAudioReady).toHaveBeenCalledWith(expect.objectContaining({ name: 'voice-recording', mimeType: 'audio/webm;codecs=opus' }))
    expect(screen.getByRole('button', { name: /record again/i })).toBeTruthy()
  })

  it('shows a useful message when microphone permission is denied', async () => {
    const denied = Object.assign(new Error('denied'), { name: 'NotAllowedError' })
    setMicrophone(vi.fn().mockRejectedValue(denied))
    render(<AudioRecorder />)

    fireEvent.click(screen.getByRole('button', { name: /start recording/i }))
    expect(await screen.findByRole('alert')).toHaveTextContent(/permission was denied/i)
  })

  it('explains when browser recording APIs are unavailable', async () => {
    vi.stubGlobal('MediaRecorder', undefined)
    render(<AudioRecorder />)

    fireEvent.click(screen.getByRole('button', { name: /start recording/i }))
    expect(await screen.findByRole('alert')).toHaveTextContent(/not supported in this browser/i)
  })

  it('stops active microphone tracks when the component unmounts', async () => {
    const user = userEvent.setup()
    const { unmount } = render(<AudioRecorder />)
    await user.click(screen.getByRole('button', { name: /start recording/i }))
    await waitFor(() => expect(screen.getByText('Recording', { exact: true })).toBeTruthy())

    unmount()
    expect(tracks[0].stop).toHaveBeenCalled()
  })
})
