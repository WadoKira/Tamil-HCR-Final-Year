# Hugging Face AI OCR Setup

## Why Hugging Face?

- **TrOCR Model** - Microsoft's state-of-the-art handwriting recognition
- **Free API** - 30,000 requests/month on free tier
- **Better Accuracy** - Specifically trained for handwritten text
- **Translation** - Built-in multilingual translation

## Setup Steps:

### 1. Get Free API Token

1. Go to https://huggingface.co/join
2. Create a free account
3. Go to https://huggingface.co/settings/tokens
4. Click "New token" → Create a token with "read" access
5. Copy your token

### 2. Configure Token

Open `frontend/.env` and add:
```
REACT_APP_HF_API_KEY=hf_your_token_here
```

### 3. Restart App

```bash
cd frontend
npm start
```

## Features:

✅ **TrOCR Model** - Trained on handwritten text  
✅ **Free Tier** - 30K requests/month  
✅ **No Installation** - Works via API  
✅ **Auto Translation** - Tamil to English  
✅ **Better than Tesseract** - For handwriting  

## Models Used:

- **OCR**: microsoft/trocr-large-handwritten
- **Translation**: Helsinki-NLP/opus-mt-mul-en

## Note:

First request may take 20 seconds (model loading). Subsequent requests are instant!
