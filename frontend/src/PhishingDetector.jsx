import React, { useState } from "react"
import "./PhishingDetector.css"

const featureList = [
  "length_url",
  "length_hostname",
  "nb_www",
  "ratio_digits_url",
  "length_words_raw",
  "char_repeat",
  "shortest_word_host",
  "longest_words_raw",
  "longest_word_path",
  "avg_word_path",
  "phish_hints",
  "nb_hyperlinks",
  "ratio_intHyperlinks",
  "ratio_extHyperlinks",
  "ratio_extRedirection",
  "links_in_tags",
  "safe_anchor",
  "domain_in_title",
  "domain_age",
  "web_traffic",
  "google_index",
  "page_rank",
]

const legitimateValues = [
  37, 19, 1, 0.0, 11, 4, 3, 11, 4.5, 7.0, 0, 17, 0.529411765, 0.470588235, 0.5,
  0, 1, 1, 0, 1, 1, 4,
]

const phishingValues = [
  77, 23, 0, 0.22077922100000005, 32, 0, 19, 32, 15.75, 14.66666667, 0, 30,
  0.966666667, 0.033333333, 0.0, 0, 0, 0, 5767, 0, 0, 2,
]

const verySafeLegitimateValues = [
  25, 10, 1, 0.05, 10, 1, 2, 12, 5, 6.5, 0, 15, 0.8, 0.2, 0.0, 0, 1, 1, 1200,
  5000, 1, 6,
]

const presets = {
  "Very Safe Legitimate": verySafeLegitimateValues,
  "Typical Legitimate": legitimateValues,
  "Typical Phishing": phishingValues,
}

const buildObject = (arr) =>
  featureList.reduce((obj, key, i) => {
    obj[key] = arr[i]
    return obj
  }, {})

export default function PhishingDetector() {
  const [formData, setFormData] = useState(
    buildObject(verySafeLegitimateValues)
  )
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [selectedPreset, setSelectedPreset] = useState("Very Safe Legitimate")

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value })
  }

  const handlePresetChange = (e) => {
    const preset = e.target.value
    setSelectedPreset(preset)
    setFormData(buildObject(presets[preset]))
    setResult(null)
  }

  const handleSubmit = async () => {
    setLoading(true)
    setResult(null)
    try {
      const orderedData = featureList.map(
        (key) => parseFloat(formData[key]) || 0
      )

      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features: orderedData }),
      })

      const data = await response.json()
      setResult(data.prediction)
    } catch (error) {
      setResult("Error fetching prediction.")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="container fade-in">
      <h1 className="title">🎯 Phishing URL Detector</h1>

      <div className="preset-select">
        <label htmlFor="preset">Choose a sample input:</label>
        <select
          id="preset"
          value={selectedPreset}
          onChange={handlePresetChange}
        >
          {Object.keys(presets).map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </select>
      </div>

      <div className="card shadow fade-in-slow">
        <div className="card-content">
          <div className="form-grid">
            {featureList.map((key, index) => (
              <div key={index} className="form-group">
                <label htmlFor={key}>{key}</label>
                <input
                  id={key}
                  name={key}
                  value={formData[key]}
                  onChange={handleChange}
                  placeholder={`Enter ${key}`}
                />
              </div>
            ))}
          </div>

          <div className="submit-wrapper">
            <button
              onClick={handleSubmit}
              disabled={loading}
              className="submit-btn"
            >
              {loading ? "🔍 Checking..." : "🚨 Detect Phishing"}
            </button>
          </div>

          {result !== null && (
            <div
              className={`result ${result === "phishing" ? "danger" : "safe"}`}
            >
              {result === "phishing" ? "⚠️ Phishing Detected" : "✅ Safe"}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
