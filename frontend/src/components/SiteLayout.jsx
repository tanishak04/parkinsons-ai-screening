import { Outlet } from 'react-router-dom'
import Navbar from './Navbar.jsx'
import Footer from './Footer.jsx'
import DisclaimerBanner from './DisclaimerBanner.jsx'

export default function SiteLayout() {
  return (
    <div className="min-h-screen bg-paper text-charcoal">
      <DisclaimerBanner />
      <Navbar />
      <main id="main-content">
        <Outlet />
      </main>
      <Footer />
    </div>
  )
}

