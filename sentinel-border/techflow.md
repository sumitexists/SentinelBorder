# SentinelBorder: Technology Stack and Architecture Flow

SentinelBorder is an autonomous edge-native border document screening and biometric triage system designed for SSB checkpoints. This document outlines the technologies used and the architectural flow of the system.

## 1. Technologies Used

### Backend (Core Processing & APIs)
- **Framework:** **FastAPI** (Python) served by **Uvicorn** for high-performance, asynchronous REST API endpoints.
- **Computer Vision & Image Processing:** 
  - **OpenCV** (`opencv-python-headless`) and **Pillow** for image manipulation, cropping, and preprocessing.
- **OCR & Document Reading:** 
  - **PyTesseract** and **EasyOCR** for extracting raw text from Visual Inspection Zones (VIZ).
  - **PassportEye** for detecting and decoding Machine Readable Zones (MRZ) on passports and visas.
- **Biometrics & Facial Recognition:** 
  - **DeepFace** for deep learning-based face detection and verification (matching document photo vs. live capture).
  - **TensorFlow/Keras** (`tf-keras`) underlying the biometric and custom machine learning models.
- **Forensic Analysis & Anomaly Detection:**
  - **Scikit-learn** and custom mathematical models via **NumPy** for Error Level Analysis (ELA), edge discontinuity, and metadata anomaly detection.
- **Generative AI (Shadow Mode):** 
  - **Google GenAI** used for advanced context parsing and cross-checking unstructured document vision fields against classical OCR.
- **Utilities:** 
  - **Pikepdf** for secure PDF ingestion.
  - **python-dateutil** and **python-dotenv** for configuration and temporal data parsing.

### Frontend (Tactical User Interface)
- **Architecture:** Single Page Application (SPA).
- **Core Languages:** **HTML5**, **CSS3** (Vanilla, Grid/Flexbox layouts), and **Vanilla JavaScript**.
- **Media Handling:** Uses native HTML5 `<video>` and `<canvas>` APIs for live webcam integration and snapshot capturing.
- **Communication:** Native `fetch` API used in `js/api.js` to communicate asynchronously with the FastAPI backend.

---

## 2. Architecture & Request Flow

The SentinelBorder pipeline operates in a sequential, multi-modular architecture. Below is the step-by-step flow when a user processes an identity document:

### Step 1: Document & Live Capture Ingestion (Frontend)
1. The operator uploads a document (JPG, PNG, PDF) into the **Document Ingestion panel**.
2. The operator optionally captures a live snapshot using the integrated webcam.
3. The frontend packages these as a `multipart/form-data` payload and sends a `POST /api/v1/screen` request to the backend.

### Step 2: Module 1 - OCR & MRZ Extraction (Backend)
1. **Routing:** The request is received by the FastAPI entrypoint (`app.py`).
2. **Extraction:** The system routes the document bytes to the `ocr_engine`.
3. **Parsing:** It extracts the MRZ (if present) and performs standard OCR on the document to retrieve structured fields (Name, Document Number, Expiry, Nationality, etc.).

### Step 3: Module 2 - Credential Validation
1. **ICAO 9303 Checks:** The `validator` module receives the OCR output.
2. **Checksums:** Validates check digits for the document number, date of birth, and expiry date.
3. **Consistency:** Compares data between the VIZ (Visual Inspection Zone) and MRZ to flag any parity mismatches. Checks if the document is expired.

### Step 4: Module 3 - Forensic Audit
1. **Tamper Detection:** The `forensics` module processes the raw image bytes to identify modifications.
2. **Techniques Used:** 
   - Generates an **Error Level Analysis (ELA) heatmap**.
   - Analyzes edge discontinuity and copy-move artifacts.
   - Inspects image metadata for tampering anomalies.
   - Checks for QR code data mismatch.
3. **Shadow-Mode AI:** (If enabled) A Gemini vision model corroborates the structural fields to detect sophisticated forgeries that classical OCR might misinterpret.

### Step 5: Module 4 - Biometric Verification
1. **Face Extraction:** The `biometrics` module detects and crops the face from the identity document.
2. **Live Comparison:** Using DeepFace, it calculates the cosine distance between the document face and the live webcam snapshot.
3. **Match Status:** Returns a confidence percentage and a match/mismatch status.

### Step 6: Composite Threat Scoring & Response
1. **Scoring Engine:** A weighted algorithm calculates a `Composite Risk Score` (0-100) based on factors like:
   - Checksum failures
   - VIZ/MRZ mismatch
   - ELA/Tamper evidence
   - Facial mismatch
   - Expiry or metadata anomalies
2. **Threat Level:** The score is categorized into an actionable threat level: **GREEN** (Clear), **YELLOW** (Warning/Manual Review), or **RED** (High Threat).
3. **JSON Response:** The backend aggregates all flags, extracted data, heatmaps (Base64), and scores into a JSON response.

### Step 7: Presentation & Action (Frontend)
1. The frontend parses the JSON response.
2. Populates the **Credential Verification panel** with extracted fields and checksum indicators.
3. Renders the **Forensic Audit panel**, displaying the Risk Score gauge, the threat level badge, ELA heatmaps, biometric comparison photos, and the threat flags log.
