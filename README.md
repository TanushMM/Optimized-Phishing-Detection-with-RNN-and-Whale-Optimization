# Optimized Phishing Detection with Recurrent Neural Network and Whale Optimization Algorithm

This project presents an intelligent phishing URL detection system using a **Recurrent Neural Network (RNN)** optimized with the **Whale Optimization Algorithm (WOA)**. It consists of a web-based front end for real-time prediction, powered by a FastAPI backend and a trained deep learning model.

---

## 🚀 Overview

Phishing remains one of the most dangerous cybersecurity threats. Traditional blacklists are reactive and often miss new attacks. This system addresses that gap using a **data-driven, intelligent approach**.

We trained an **RNN classifier** that captures temporal dependencies across multiple features extracted from URLs and page structures. The model's hyperparameters were fine-tuned using the **Whale Optimization Algorithm**, enabling better convergence and improved generalization.

---

## 🧠 Model Training Highlights

- 📊 **Dataset**: A preprocessed dataset containing 22 numerical features extracted from URLs (e.g., length, special characters, word count, domain age, etc.)
- 🔍 **Normalization**: Features scaled using `MinMaxScaler`
- 🧬 **Model Architecture**:
  - Input Layer: Shape (22, 1)
  - 2 LSTM Layers with `relu` activations and Dropout
  - Dense Output Layer with Sigmoid activation
- 🐋 **Whale Optimization Algorithm**: Applied to tune key hyperparameters like:
  - Number of LSTM units
  - Dropout rate
  - Batch size and learning rate
- ✅ **Final Accuracy**: Achieved high precision and recall on validation and test sets

---

## 🧪 Real-Time Inference (FastAPI Backend)

The backend accepts an array of 22 feature values, scales it using the saved `MinMaxScaler`, reshapes it for the RNN input format, and outputs whether the site is `safe` or `phishing`.

### Endpoint

```
POST /predict
```

#### Payload

```json
{
  "features": [feature_1, feature_2, ..., feature_22]
}
```

#### Response

```json
{
  "prediction": "safe" // or "phishing"
}
```

---

## 🌐 Frontend

- Built using **React.js**
- Interactive form to input features or choose from **sample presets**
- Animated, responsive UI
- Displays prediction results clearly with color-coded messages

---

## 🛠️ How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/TanushMM/Optimized-Phishing-Detection-with-RNN-and-Whale-Optimization.git
cd phishing-rnn-woa
```

### 2. Install Backend Dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the FastAPI Backend

```bash
uvicorn main:app --reload --port 8000
```

### 4. Start the React Frontend

```bash
cd frontend
npm install
npm start
```

---

## 📚 References

- Whale Optimization Algorithm (Mirjalili & Lewis, 2016)
- TensorFlow Keras Documentation
- FastAPI Official Docs

---

## ✨ Future Work

- Integrate URL parsing and feature extraction pipeline
- Add model confidence score in predictions
- Convert into a mobile-friendly Progressive Web App (PWA)

---

## 🛡️ Disclaimer

This system is for **educational and research purposes only**. Do not rely solely on this tool for security decisions in production environments.

---

## 📬 Contact

Created by **Tanush Modem Mahesh** & **Syed Fayaadh**
If you find this project helpful, feel free to ⭐️ the repo and share your thoughts!
