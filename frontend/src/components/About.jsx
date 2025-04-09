import React from "react"
import { Link } from "react-router-dom"
import "./About.css"

export default function About() {
  return (
    <div className="about-container">
      <h1>About Phishing Detector</h1>
      <p>
        Phishing Detector is built to protect users from malicious websites by
        analyzing URLs using advanced machine learning techniques.
      </p>
      <section className="project-details">
        <h2>How It Works</h2>
        <ol>
          <li>
            <strong>Data Processing:</strong> The model is trained on a dataset
            of URLs labeled as phishing or legitimate. Preprocessing ensures
            clean and consistent data.
          </li>
          <li>
            <strong>Feature Selection:</strong> Features are selected based on
            their importance, determined using a Random Forest model. This step
            ensures only relevant data contributes to the predictions.
          </li>
          <li>
            <strong>Model Training:</strong> An RNN model is employed, tuned
            with the Whale Optimization Algorithm (WOA) for hyperparameter
            selection. The best hyperparameters lead to an optimal model.
          </li>
          <li>
            <strong>Evaluation:</strong> The trained model undergoes rigorous
            testing using metrics like accuracy, precision, recall, and F1 score
            to validate its effectiveness.
          </li>
        </ol>
        <p>
          This workflow guarantees a robust and reliable phishing detection
          system.
        </p>
      </section>
      <div className="back-button-wrapper">
        <Link to="/" className="back-button">
          Back to Home
        </Link>
      </div>
    </div>
  )
}
