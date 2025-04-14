import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

# Step 1: Load original dataset
df = pd.read_csv("PhiUSIIL_Phishing_URL_Dataset.csv")

# Step 2: Separate phishing and legitimate samples
df_legit = df[df['label'] == 0]
df_phish = df[df['label'] == 1]

# Step 3: Balance the classes (downsample phishing to 100,945 to match legitimate)
desired_count = 100_945
df_legit = df_legit.sample(n=desired_count, random_state=42)
df_phish = df_phish.sample(n=desired_count, random_state=42)

# Step 4: Combine and shuffle the balanced dataset
balanced_df = pd.concat([df_legit, df_phish]).sample(frac=1.0, random_state=42).reset_index(drop=True)

# Step 5: Separate label
label = balanced_df['label']

# Step 6: Drop non-numeric columns (like URL, filenames, etc.)
features = balanced_df.drop(columns=['label'])
features = features.select_dtypes(include=[np.number])

print(f"🧮 Feature shape before scaling: {features.shape}")

# Step 7: Normalize features using MinMaxScaler
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# Save the scaler for use during prediction
joblib.dump(scaler, "transform.pkl")
print("✅ Saved scaler as transform.pkl")

# Step 8: Save the balanced dataset
scaled_df = pd.DataFrame(scaled_features, columns=features.columns)
scaled_df['label'] = label.values
scaled_df.to_csv("balanced_dataset.csv", index=False)

# Final Info
print(f"\n📦 transform.pkl shape: ({scaled_features.shape[1]},) ✅")
print("📦 balanced_dataset.csv shape:", scaled_df.shape)
