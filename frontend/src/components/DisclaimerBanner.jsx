import { FlaskConical } from 'lucide-react'

export default function DisclaimerBanner() {
  return (
    <div className="prototype-banner" role="note">
      <div className="site-shell prototype-inner">
        <FlaskConical size={14} strokeWidth={1.8} aria-hidden="true" />
        <span><strong>Research prototype</strong><span className="banner-divider">/</span>Current model responses are simulated and are not medical advice.</span>
      </div>
    </div>
  )
}

