import React, { useState } from "react"
import axios from "axios"
import "./PhishingDetector.css"

export default function PhishingDetector() {
  const [url, setUrl] = useState("")
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setResult(null)

    try {
      const response = await axios.post("http://localhost:8000/predict", {
        url,
      })
      console.log(response.data)
      setResult(response.data)
    } catch (error) {
      setResult({ error: error.response?.data?.detail || "Error occurred." })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="container">
      <div className="card fade-in">
        <h1 className="title">Phishing URL Detection</h1>
        <form onSubmit={handleSubmit}>
          <div className="form-grid">
            <div className="form-group">
              <label htmlFor="url">Enter URL:</label>
              <input
                type="text"
                id="url"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="https://example.com"
              />
            </div>
          </div>
          <div className="submit-wrapper">
            <button type="submit" className="submit-btn" disabled={loading}>
              {loading ? "Loading..." : "Check URL"}
            </button>
          </div>
        </form>
        {result && (
          <div
            className={`result ${
              result.error
                ? "danger"
                : result.prediction === "phishing"
                ? "danger"
                : "safe"
            } fade-in-slow`}
          >
            {result.error ? (
              <p>{result.error}</p>
            ) : (
              <p>
                The URL <strong>{result.url}</strong> is classified as{" "}
                <span>
                  {result.prediction === "Phishing" ? "Phishing" : "Legitimate"}
                </span>
                .
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
