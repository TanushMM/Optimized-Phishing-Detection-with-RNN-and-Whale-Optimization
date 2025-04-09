import React from "react"
import { Link } from "react-router-dom"
import "./LandingPage.css"

export default function LandingPage() {
  return (
    <div className="landing-container">
      <header className="landing-header">
        <h1>Phishing Detector</h1>
        <p>
          Leveraging cutting-edge machine learning to ensure your safety online.
          Detect phishing URLs effortlessly and accurately.
        </p>
      </header>
      <main className="landing-content">
        <div className="animated-box">
          <h2>Stay Safe Online</h2>
          <p>
            Input a URL and let our model classify it as legitimate or phishing
            in seconds.
          </p>
        </div>
        <div className="landing-buttons">
          <Link to="/detect" className="primary-button">
            Get Started
          </Link>
          <Link to="/about" className="secondary-button">
            About Us
          </Link>
        </div>
      </main>
    </div>
  )
}
