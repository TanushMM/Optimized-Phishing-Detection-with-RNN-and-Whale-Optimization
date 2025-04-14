import React from "react"
import { Link } from "react-router-dom"
import "./About.css"

import dataFlowDiagram from "../images/Data_Flow_Diagram.png"
import systemArchitectureDiagram from "../images/System Architecture.png"

export default function About() {
  return (
    <div className="about-container">
      <div className="back-button-wrapper">
        <Link to="/" className="back-button">
          Back to Home
        </Link>
      </div>
      <h1>About Phishing Detector</h1>
      <p>
        The <strong>Phishing Detector</strong> project aims to safeguard users
        from malicious websites by utilizing advanced machine learning
        techniques to analyze URLs effectively. With the rise of online threats,
        especially phishing attacks which manipulate user trust to gain access
        to sensitive information, this project is developed to provide an
        intelligent detection system leveraging Deep Learning and optimization
        algorithms for real-time threat identification.
      </p>

      <section className="project-details">
        <h2>Project Overview</h2>
        <p>
          Our system is built upon the principles of deep learning, particularly
          utilizing <strong>Recurrent Neural Networks (RNNs)</strong> for their
          ability to capture complex sequential dependencies inherent in URL
          structures. To optimize model performance, we employ the{" "}
          <strong>Whale Optimization Algorithm (WOA)</strong>, a bio-inspired
          algorithm that tunes hyperparameters by mimicking the foraging
          strategies of humpback whales.
        </p>

        <h3>System Architecture</h3>
        <div className="image-section">
          <img src={systemArchitectureDiagram} alt="System Architecture" />
          <p>
            Image Description: Flow diagram of the phishing detection process.
          </p>
        </div>

        <h3>How It Works</h3>
        <ol>
          <li>
            <strong>Dataset Preparation:</strong> Utilizes a balanced dataset
            comprising equal entries of phishing and legitimate URLs (100,945
            each) ensuring unbiased learning.
          </li>
          <li>
            <strong>Feature Extraction:</strong> Extracts crucial features from
            URLs, including length, domain attributes, and structural
            characteristics. Key features are calculated using custom scripts.
          </li>
          <li>
            <strong>Feature Selection:</strong> Implements a{" "}
            <strong>Random Forest</strong> model to identify the most
            informative features, enhancing the efficiency of our detection
            mechanism.
          </li>
          <li>
            <strong>Model Training:</strong> The main classification model
            integrates LSTM layers to learn from extracted features, capturing
            sequential patterns and improving detection accuracy.
          </li>
          <li>
            <strong>Hyperparameter Optimization:</strong> The WOA enhances the
            RNN model by efficiently tuning essential hyperparameters such as
            learning rate and batch size.
          </li>
          <li>
            <strong>Evaluation:</strong> The model's performance is assessed
            using metrics like accuracy, precision, recall, and F1 score,
            ensuring robust and reliable predictions.
          </li>
        </ol>

        <h3>Mathematical Formulation</h3>
        <p>
          The Whale Optimization Algorithm operates through two strategies: the{" "}
          <strong>encircling prey</strong> and the{" "}
          <strong>spiral updating position</strong>. The mathematical
          expressions governing these strategies include:
        </p>
        <ul>
          <li>
            <strong>Encircling Prey:</strong> <br />
            <em>New Position = Current Best Position - A × D</em> <br />
            Where: <br />
            <code>A = 2 × a × r - a</code>,{" "}
            <code>D = |C × Current Best Position - Current Position|</code>, and{" "}
            <code>C = 2 × r</code> <br />(<code>a</code> and <code>r</code> are
            constants defining the essence of exploration and exploitation.)
          </li>
          <li>
            <strong>Spiral Updating Position:</strong> <br />
            <code>
              New Position = Current Best Position + D × exp(b × l) × cos(2π ×
              l)
            </code>{" "}
            <br />
            Where <code>b</code> is a constant defining the logarithmic spiral,{" "}
            <code>l ∈ [-1, 1]</code>, and <code>D</code> is the distance between
            the current position and the best position.
          </li>
        </ul>
        <p>
          The WOA balances exploration and exploitation by randomly selecting
          search agents to enhance the accuracy of the detection model.
        </p>

        <h3>System Deployment</h3>
        <p>
          The system is designed for modular integration, allowing deployment in
          various environments—from personal machines to enterprise networks or
          as part of a browser-based plugin for end-user protection. The modular
          approach ensures adaptability for a broad spectrum of applications in
          cybersecurity.
        </p>
      </section>

      <section className="team-section">
        <h2>Project Team</h2>
        <p>
          This project was developed under the guidance of{" "}
          <strong>Dr. S. Sharon Priya</strong>. The team includes:
        </p>
        <ul>
          <li>Syed Fayaadh S (210071601178)</li>
          <li>Tanush M M (210071601181)</li>
        </ul>
      </section>

      <section className="acknowledgements">
        <h2>Acknowledgements</h2>
        <p>
          We express our deepest gratitude to all faculty members and our
          project supervisor for their invaluable support and insights
          throughout this project journey.
        </p>
      </section>

      {/* Placeholder for images */}
      <div className="image-section">
        <img src={dataFlowDiagram} alt="Phishing Detection Data Flow" />
        <p>
          Image Description: Flow diagram of the phishing detection process.
        </p>
      </div>
    </div>
  )
}
