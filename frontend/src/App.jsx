import { BrowserRouter as Router, Routes, Route } from "react-router-dom"
import LandingPage from "./components/LandingPage.jsx"
import PhishingDetector from "./components/PhishingDetector.jsx"
import About from "./components/About.jsx"

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/detect" element={<PhishingDetector />} />
        <Route path="/about" element={<About />} />
      </Routes>
    </Router>
  )
}
