// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest'
import { afterEach, describe, expect, it } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import ResultsPage from './ResultsPage.jsx'

function renderResults(state) {
  return render(
    <MemoryRouter initialEntries={[{ pathname: '/results', state }]}>
      <Routes><Route path="/results" element={<ResultsPage />} /></Routes>
    </MemoryRouter>,
  )
}

const backendResponse = {
  prediction: 1,
  label: 'PARKINSONS',
  confidence: 0.8145,
  pd_probability: 0.8145,
  healthy_probability: 0.1855,
  model_version: '0.0.0-mock',
  mock: true,
  disclaimer: 'Research prototype only; this output is not a diagnosis.',
}

describe('ResultsPage', () => {
  afterEach(() => cleanup())

  it('renders actual backend values and prominently identifies a mock response', () => {
    renderResults({ status: 'success', result: backendResponse })

    expect(screen.getAllByText('PARKINSONS')).toHaveLength(2)
    expect(screen.getByText('MOCK / SIMULATED RESULT')).toBeInTheDocument()
    expect(screen.getByText('0.0.0-mock')).toBeInTheDocument()
    expect(screen.getAllByText('81.5%')).toHaveLength(2)
    expect(screen.getByText('18.6%')).toBeInTheDocument()
    expect(screen.getByText(/not a diagnosis/i)).toBeInTheDocument()
    expect(screen.getByText(/not medical certainty/i)).toBeInTheDocument()
    expect(screen.getByRole('link', { name: /start another session/i })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: /return to screening/i })).toBeInTheDocument()
  })

  it('recognizes the upstream _mock alias and hides absent confidence values', () => {
    renderResults({ status: 'success', result: { prediction: 0, label: 'HEALTHY', _mock: true } })
    expect(screen.getByText('MOCK / SIMULATED RESULT')).toBeInTheDocument()
    expect(screen.queryByText('Model confidence')).not.toBeInTheDocument()
    expect(screen.queryByText('Class probabilities')).not.toBeInTheDocument()
  })

  it('shows loading state without inventing a result', () => {
    renderResults({ status: 'loading' })
    expect(screen.getByText(/processing the research request/i)).toBeInTheDocument()
  })

  it('shows request errors without inventing a result', () => {
    renderResults({ status: 'error', message: 'Backend unavailable' })
    expect(screen.getByText('Backend unavailable')).toBeInTheDocument()
  })

  it('shows an empty state when there is no response', () => {
    renderResults(null)
    expect(screen.getByText(/no result to display/i)).toBeInTheDocument()
  })

  it('treats a success state with an invalid response as an error', () => {
    renderResults({ status: 'success', result: { label: 'PARKINSONS' } })
    expect(screen.getByText(/did not include a valid prediction or research payload/i)).toBeInTheDocument()
  })

  it('renders research-status UI when live audio screening is disabled', () => {
    const disabledResponse = {
      status: 'screening_disabled',
      screening_available: false,
      message: 'Automated audio screening is temporarily disabled.',
      validation_summary: 'Evaluated under LOGO; achieved balanced accuracy 0.6667.',
      duration_seconds: 1.5,
      sampling_frequency: 44100,
      extracted_feature_count: 16,
      extracted_features: {
        'MDVP:Fo(Hz)': 154.23,
        'HNR': 21.45,
      },
      disclaimer: 'Research status notification only; this output is not a diagnosis.',
    }

    renderResults({ status: 'success', result: disabledResponse })

    // Verify clear, professional research notice
    expect(screen.getByText('AUTOMATED SCREENING DISABLED')).toBeInTheDocument()
    expect(screen.getByText('Screening Disabled')).toBeInTheDocument()
    expect(screen.getAllByText(/Validation criteria not met/i).length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText(/Audio analysis is experimental/i)).toBeInTheDocument()
    expect(screen.getByText(/Automated screening temporarily disabled/i)).toBeInTheDocument()
    expect(screen.getByText(/Not a medical assessment/i)).toBeInTheDocument()

    // Verify raw feature inspection is displayed
    expect(screen.getByText('Extracted Acoustic Biomarkers')).toBeInTheDocument()
    expect(screen.getByText('MDVP:Fo(Hz)')).toBeInTheDocument()
    expect(screen.getByText('HNR')).toBeInTheDocument()

    // CRITICAL: Ensure disease classifications, confidence, and probabilities are NOT rendered
    expect(screen.queryByText('PARKINSONS')).not.toBeInTheDocument()
    expect(screen.queryByText('HEALTHY')).not.toBeInTheDocument()
    expect(screen.queryByText('Model confidence')).not.toBeInTheDocument()
    expect(screen.queryByText('Class probabilities')).not.toBeInTheDocument()
    expect(screen.queryByText('Parkinson’s class')).not.toBeInTheDocument()
    expect(screen.queryByText('Healthy class')).not.toBeInTheDocument()
  })
})
