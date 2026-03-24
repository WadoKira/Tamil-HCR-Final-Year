# Gemini API Integration Setup

## Steps to Setup:

1. **Install Gemini SDK**
   ```bash
   cd frontend
   npm install @google/generative-ai
   ```

2. **Get Gemini API Key**
   - Go to https://makersuite.google.com/app/apikey
   - Create a new API key
   - Copy the API key

3. **Configure API Key**
   - Open `frontend/.env` file
   - Replace `YOUR_GEMINI_API_KEY_HERE` with your actual API key
   ```
   REACT_APP_GEMINI_API_KEY=your_actual_api_key_here
   ```

4. **Restart the Frontend**
   ```bash
   npm start
   ```

## Features:

- **Toggle Switch**: Choose between Gemini AI (better accuracy) or Local Model (prototype)
- **Tamil OCR**: Gemini extracts Tamil handwritten text from images
- **Translation**: Automatically translates Tamil text to English
- **Local Model**: Still available for prototype demonstration

## Usage:

1. Upload a Tamil handwritten image
2. Check "Use Gemini AI" for better accuracy (default)
3. Click "Detect Text"
4. View Tamil text and English translation
