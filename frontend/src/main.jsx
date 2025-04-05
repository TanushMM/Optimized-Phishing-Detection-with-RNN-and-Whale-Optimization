import { StrictMode } from "react"
import { createRoot } from "react-dom/client"
import "./index.css"
import PhishingDetector from "./PhishingDetector.jsx"

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <PhishingDetector />
  </StrictMode>
)
