// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from 'vitest'
import '@testing-library/jest-dom/vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import AudioUploader, { validateAudioFile } from './AudioUploader.jsx'

function makeWav(name = 'sample.wav', size = 12) {
  const bytes = new Uint8Array(size)
  if (size >= 12) {
    bytes.set([...new TextEncoder().encode('RIFF')], 0)
    bytes.set([...new TextEncoder().encode('WAVE')], 8)
  }
  return new File([bytes], name, { type: 'audio/wav' })
}

describe('AudioUploader', () => {
  afterEach(() => cleanup())
  it('accepts WAV files and offers an audio preview with file details', async () => {
    const onAudioReady = vi.fn()
    const onRemove = vi.fn()
    vi.stubGlobal('URL', {
      ...URL,
      createObjectURL: vi.fn(() => 'blob:upload-preview'),
      revokeObjectURL: vi.fn(),
    })
    render(<AudioUploader onAudioReady={onAudioReady} onRemove={onRemove} getDuration={async () => 7} />)
    fireEvent.change(screen.getByLabelText('Choose WAV audio file'), {
      target: { files: [makeWav()] },
    })

    expect(await screen.findByText('sample.wav')).toBeTruthy()
    expect(screen.getByLabelText('Preview uploaded audio')).toBeTruthy()
    expect(onAudioReady).toHaveBeenCalledWith(expect.objectContaining({ name: 'sample.wav', duration: 7, format: 'WAV' }))
    fireEvent.click(screen.getByRole('button', { name: /remove uploaded audio/i }))
    expect(onRemove).toHaveBeenCalledOnce()
    expect(screen.queryByLabelText('Preview uploaded audio')).toBeNull()
  })

  it('warns when the selected file is not WAV', async () => {
    render(<AudioUploader />)
    fireEvent.change(screen.getByLabelText('Choose WAV audio file'), {
      target: { files: [new File(['voice'], 'voice.mp3', { type: 'audio/mpeg' })] },
    })
    expect(await screen.findByRole('alert')).toHaveTextContent(/not a supported WAV/i)
  })

  it('enforces configured file-size and recording-duration limits', async () => {
    const oversized = await validateAudioFile(makeWav('large.wav', 100), { maxFileSizeMB: 0.00001 })
    const tooLong = await validateAudioFile(makeWav(), { maxDurationSeconds: 5, getDuration: async () => 6 })

    expect(oversized.valid).toBe(false)
    expect(oversized.error).toMatch(/larger than/i)
    expect(tooLong.valid).toBe(false)
    expect(tooLong.error).toMatch(/technical limit/i)
  })

  it('rejects files with a WAV extension but no WAV header', async () => {
    const result = await validateAudioFile(new File(['not wave'], 'fake.wav', { type: 'audio/wav' }), {
      getDuration: async () => 2,
    })
    expect(result.valid).toBe(false)
    expect(result.error).toMatch(/does not appear to contain WAV/i)
  })

  it('checks a dropped file as well as a selected file', async () => {
    render(<AudioUploader getDuration={async () => 3} />)
    const dropzone = screen.getByText(/choose a file or drop it here/i).closest('[role="button"]')
    fireEvent.drop(dropzone, { dataTransfer: { files: [new File(['bad'], 'bad.mp3', { type: 'audio/mpeg' })] } })
    expect(await screen.findByRole('alert')).toHaveTextContent(/not a supported WAV/i)
  })
})
