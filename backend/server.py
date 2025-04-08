import os
import re
import socket
import joblib
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow.keras.models import load_model

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the scaler and model
try:
    scaler = joblib.load("transform.pkl")
    model = load_model("model.h5")
except Exception as e:
    raise RuntimeError(f"Failed to load model or scaler: {e}")

# Define input structure
class URLInput(BaseModel):
    url: str

# Utility function to check if domain is an IP address.
def is_ip(domain):
    try:
        socket.inet_aton(domain)
        return True
    except socket.error:
        return False

# Feature extraction: extract 55 features and convert to a DataFrame (preserving feature names)
def extract_features(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
    except Exception:
        raise HTTPException(status_code=400, detail="Unable to fetch URL.")
    
    parsed = urlparse(url)
    domain = parsed.netloc
    path = parsed.path
    html = response.text
    lines = html.splitlines()
    
    features = {
        "URLLength": len(url),
        "DomainLength": len(domain),
        "IsDomainIP": int(is_ip(domain)),
        "URLSimilarityIndex": len(set(domain).intersection(path)) / len(domain) if domain else 0,
        "CharContinuationRate": max([len(x) for x in re.findall(r"([a-zA-Z])\1*", url)] + [1]) / len(url),
        "TLDLegitimateProb": int(parsed.netloc.split(".")[-1] in ["com", "org", "net", "gov", "edu", "in"]),
        "TLDLength": len(parsed.netloc.split(".")[-1]),
        "NoOfSubDomain": len(domain.split(".")) - 2,
        "NoOfEqualsInURL": url.count("="),
        "NoOfQMarkInURL": url.count("?"),
        "NoOfAmpersandInURL": url.count("&"),
        "NoOfOtherSpecialCharsInURL": len(re.findall(r"[^\w\d\-.:/@]", url)),
        "IsHTTPS": int(parsed.scheme.lower() == "https"),
        "SpacialCharRatioInURL": len(re.findall(r"[^\w]", url)) / len(url),
        "URLCharProb": len(re.findall(r"[a-zA-Z0-9]", url)) / len(url),
        "NoOfLettersInURL": len(re.findall(r"[a-zA-Z]", url)),
        "LetterRatioInURL": len(re.findall(r"[a-zA-Z]", url)) / len(url),
        "NoOfDegitsInURL": len(re.findall(r"[0-9]", url)),
        "DegitRatioInURL": len(re.findall(r"[0-9]", url)) / len(url),
        "HasObfuscation": int(bool(re.search(r"%[0-9a-fA-F]{2}", url))),
        "NoOfObfuscatedChar": len(re.findall(r"%[0-9a-fA-F]{2}", url)),
        "ObfuscationRatio": len(re.findall(r"%[0-9a-fA-F]{2}", url)) / len(url),
        "LineOfCode": len(lines),
        "LargestLineLength": max([len(line) for line in lines], default=0),
        "HasTitle": int(bool(soup.title)),
        "DomainTitleMatchScore": len(set(domain.lower()).intersection(soup.title.string.lower())) / len(domain) if soup.title else 0,
        "URLTitleMatchScore": len(set(url.lower()).intersection(soup.title.string.lower())) / len(url) if soup.title else 0,
        "HasFavicon": int(bool(soup.find("link", rel=lambda x: x and "icon" in x.lower()))),
        "Robots": int(bool(soup.find("meta", attrs={"name": "robots"}))),
        "IsResponsive": int("viewport" in str(soup.find_all("meta"))),
        "NoOfURLRedirect": sum(1 for h in response.history if h.status_code in [301, 302]),
        "NoOfSelfRedirect": sum(1 for h in response.history if urlparse(h.url).netloc == domain),
        "HasDescription": int(bool(soup.find("meta", attrs={"name": "description"}))),
        "NoOfPopup": len(re.findall(r"window\\.open", html)),
        "NoOfiFrame": len(soup.find_all("iframe")),
        "HasExternalFormSubmit": int(
            any(urlparse(form.get("action")).netloc != domain for form in soup.find_all("form", action=True))
        ),
        "HasSocialNet": int(
            any(net in a.get("href", "") for net in ["facebook", "twitter", "instagram", "linkedin", "youtube"] for a in soup.find_all("a"))
        ),
        "HasSubmitButton": int(bool(soup.find("input", {"type": "submit"}))),
        "HasHiddenFields": int(bool(soup.find("input", {"type": "hidden"}))),
        "HasPasswordField": int(bool(soup.find("input", {"type": "password"}))),
        "Bank": int(any(word in html.lower() for word in ["bank", "account", "transfer"])),
        "Pay": int(any(word in html.lower() for word in ["pay", "payment", "checkout"])),
        "Crypto": int(any(word in html.lower() for word in ["crypto", "bitcoin", "btc", "eth"])),
        "HasCopyrightInfo": int("©" in html or "copyright" in html.lower()),
        "NoOfImage": len(soup.find_all("img")),
        "NoOfCSS": len([l for l in soup.find_all("link", href=True) if l.get("rel") and "stylesheet" in l["rel"]]),
        "NoOfJS": len(soup.find_all("script", src=True)),
        "NoOfSelfRef": sum(1 for a in soup.find_all("a", href=True) if urlparse(urljoin(url, a["href"])).netloc == domain),
        "NoOfEmptyRef": sum(1 for a in soup.find_all("a", href=True) if a["href"].strip() == "#"),
        "NoOfExternalRef": sum(1 for a in soup.find_all("a", href=True) if urlparse(urljoin(url, a["href"])).netloc != domain),
    }
    
    # Convert to DataFrame to preserve the feature names and order.
    df_features = pd.DataFrame([features])
    return df_features

# Routes
@app.post("/predict")
def predict(input: URLInput):
    try:
        # Extract all 55 features as a DataFrame with column names
        df_features = extract_features(input.url)
        
        # If the scaler was fitted with a DataFrame, its feature order is in scaler.feature_names_in_
        if hasattr(scaler, "feature_names_in_"):
            expected_order = list(scaler.feature_names_in_)
            df_features = df_features[expected_order]
        
        # Transform with scaler
        scaled_array = scaler.transform(df_features)
        scaled_df = pd.DataFrame(scaled_array, columns=df_features.columns)
        
        # Select only the 13 features using their actual names.
        selected_columns = [
            "URLSimilarityIndex",
            "NoOfOtherSpecialCharsInURL",
            "IsHTTPS",
            "LineOfCode",
            "HasDescription",
            "NoOfiFrame",
            "HasSocialNet",
            "HasCopyrightInfo",
            "NoOfImage",
            "NoOfCSS",
            "NoOfJS",
            "NoOfSelfRef",
            "NoOfExternalRef"
        ]
        
        # Check that all selected columns exist in our DataFrame
        for col in selected_columns:
            if col not in scaled_df.columns:
                raise HTTPException(status_code=500, detail=f"Expected feature {col} not found.")
        
        selected_scaled = scaled_df[selected_columns].values
        
        # Reshape to the shape expected by the RNN: (batch, time_steps, features)
        reshaped = selected_scaled.reshape((selected_scaled.shape[0], selected_scaled.shape[1], 1))
        
        prediction = model.predict(reshaped)
        print(prediction)
        
        # Using a threshold: if prediction[0][0] <= 0.5, it's legitimate; otherwise, phishing.
        label = "legitimate" if prediction[0][0] <= 0.5 else "phishing"
        return {"url": input.url, "prediction": label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Phishing Detection API is live!"}

import nest_asyncio
import uvicorn
nest_asyncio.apply()
uvicorn.run(app, host="0.0.0.0", port=8000)







# import os
# import re
# import socket
# import joblib
# import numpy as np
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from bs4 import BeautifulSoup
# from urllib.parse import urlparse, urljoin
# from tensorflow.keras.models import load_model
# import requests

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load the scaler and model
# try:
#     scaler = joblib.load("transform.pkl")
#     model = load_model("model.h5")
# except Exception as e:
#     raise RuntimeError(f"Failed to load model or scaler: {e}")

# # Define input structure
# class URLInput(BaseModel):
#     url: str

# # Utility functions
# def is_ip(domain):
#     try:
#         socket.inet_aton(domain)
#         return True
#     except socket.error:
#         return False

# def extract_features(url):
#     try:
#         response = requests.get(url, timeout=10)
#         soup = BeautifulSoup(response.text, "html.parser")
#     except Exception:
#         raise HTTPException(status_code=400, detail="Unable to fetch URL.")

#     parsed = urlparse(url)
#     domain = parsed.netloc
#     path = parsed.path
#     html = response.text
#     lines = html.splitlines()

#     features = {
#         "URLLength": len(url),
#         "DomainLength": len(domain),
#         "IsDomainIP": int(is_ip(domain)),
#         "URLSimilarityIndex": len(set(domain).intersection(path)) / len(domain) if domain else 0,
#         "CharContinuationRate": max([len(x) for x in re.findall(r"([a-zA-Z])\1*", url)] + [1]) / len(url),
#         "TLDLegitimateProb": int(parsed.netloc.split(".")[-1] in ["com", "org", "net", "gov", "edu", "in"]),
#         "TLDLength": len(parsed.netloc.split(".")[-1]),
#         "NoOfSubDomain": len(domain.split(".")) - 2,
#         "NoOfEqualsInURL": url.count("="),
#         "NoOfQMarkInURL": url.count("?"),
#         "NoOfAmpersandInURL": url.count("&"),
#         "NoOfOtherSpecialCharsInURL": len(re.findall(r"[^\w\d\-.:/@]", url)),
#         "IsHTTPS": int(parsed.scheme.lower() == "https"),
#         "SpacialCharRatioInURL": len(re.findall(r"[^\w]", url)) / len(url),
#         "URLCharProb": len(re.findall(r"[a-zA-Z0-9]", url)) / len(url),
#         "NoOfLettersInURL": len(re.findall(r"[a-zA-Z]", url)),
#         "LetterRatioInURL": len(re.findall(r"[a-zA-Z]", url)) / len(url),
#         "NoOfDegitsInURL": len(re.findall(r"[0-9]", url)),
#         "DegitRatioInURL": len(re.findall(r"[0-9]", url)) / len(url),
#         "HasObfuscation": int(bool(re.search(r"%[0-9a-fA-F]{2}", url))),
#         "NoOfObfuscatedChar": len(re.findall(r"%[0-9a-fA-F]{2}", url)),
#         "ObfuscationRatio": len(re.findall(r"%[0-9a-fA-F]{2}", url)) / len(url),
#         "LineOfCode": len(lines),
#         "LargestLineLength": max([len(line) for line in lines], default=0),
#         "HasTitle": int(bool(soup.title)),
#         "DomainTitleMatchScore": len(set(domain.lower()).intersection(soup.title.string.lower())) / len(domain) if soup.title else 0,
#         "URLTitleMatchScore": len(set(url.lower()).intersection(soup.title.string.lower())) / len(url) if soup.title else 0,
#         "HasFavicon": int(bool(soup.find("link", rel=lambda x: x and "icon" in x.lower()))),
#         "Robots": int(bool(soup.find("meta", attrs={"name": "robots"}))),
#         "IsResponsive": int("viewport" in str(soup.find_all("meta"))),
#         "NoOfURLRedirect": sum(1 for h in response.history if h.status_code in [301, 302]),
#         "NoOfSelfRedirect": sum(1 for h in response.history if urlparse(h.url).netloc == domain),
#         "HasDescription": int(bool(soup.find("meta", attrs={"name": "description"}))),
#         "NoOfPopup": len(re.findall(r"window\\.open", html)),
#         "NoOfiFrame": len(soup.find_all("iframe")),
#         "HasExternalFormSubmit": int(
#             any(urlparse(form.get("action")).netloc != domain for form in soup.find_all("form", action=True))
#         ),
#         "HasSocialNet": int(
#             any(net in a.get("href", "") for net in ["facebook", "twitter", "instagram", "linkedin", "youtube"] for a in soup.find_all("a"))
#         ),
#         "HasSubmitButton": int(bool(soup.find("input", {"type": "submit"}))),
#         "HasHiddenFields": int(bool(soup.find("input", {"type": "hidden"}))),
#         "HasPasswordField": int(bool(soup.find("input", {"type": "password"}))),
#         "Bank": int(any(word in html.lower() for word in ["bank", "account", "transfer"])),
#         "Pay": int(any(word in html.lower() for word in ["pay", "payment", "checkout"])),
#         "Crypto": int(any(word in html.lower() for word in ["crypto", "bitcoin", "btc", "eth"])),
#         "HasCopyrightInfo": int("©" in html or "copyright" in html.lower()),
#         "NoOfImage": len(soup.find_all("img")),
#         "NoOfCSS": len([l for l in soup.find_all("link", href=True) if l.get("rel") and "stylesheet" in l["rel"]]),
#         "NoOfJS": len(soup.find_all("script", src=True)),
#         "NoOfSelfRef": sum(1 for a in soup.find_all("a", href=True) if urlparse(urljoin(url, a["href"])).netloc == domain),
#         "NoOfEmptyRef": sum(1 for a in soup.find_all("a", href=True) if a["href"].strip() == "#"),
#         "NoOfExternalRef": sum(1 for a in soup.find_all("a", href=True) if urlparse(urljoin(url, a["href"])).netloc != domain),
#     }

#     return np.array([list(features.values())], dtype=np.float32)

# # Routes
# @app.post("/predict")
# def predict(input: URLInput):
#     try:
#         # Extract and scale all features
#         features = extract_features(input.url)
#         scaled_features = scaler.transform(features)

#         # Select only the 13 features for the model
#         selected_indices = [3, 11, 12, 20, 28, 34, 36, 39, 40, 41, 42, 43, 47]
#         selected_features = scaled_features[:, selected_indices]

#         # Reshape and predict
#         reshaped = selected_features.reshape((selected_features.shape[0], selected_features.shape[1], 1))
#         prediction = model.predict(reshaped)

#         print(prediction)

#         label = "legitimate" if prediction[0][0] <= 0.5 else "phishing"
#         return {"url": input.url, "prediction": label}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/")
# def read_root():
#     return {"message": "Phishing Detection API is live!"}

# import nest_asyncio
# import uvicorn
# nest_asyncio.apply()
# uvicorn.run(app, host="0.0.0.0", port=8000)