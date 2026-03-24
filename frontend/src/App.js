import React, { useState } from 'react';
import Tesseract from 'tesseract.js';
import './App.css';

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [tamilText, setTamilText] = useState('');
  const [englishText, setEnglishText] = useState('');
  const [loading, setLoading] = useState(false);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      setPreview(URL.createObjectURL(file));
      setTamilText('');
      setEnglishText('');
    }
  };

  const processImage = async () => {
    if (!image) return;
    
    setLoading(true);
    setTamilText('');
    setEnglishText('');

    try {
      // OCR to extract Tamil text
      const { data: { text } } = await Tesseract.recognize(image, 'tam');
      setTamilText(text);

      // Translate using MyMemory API
      const response = await fetch(
        `https://api.mymemory.translated.net/get?q=${encodeURIComponent(text)}&langpair=ta|en`
      );
      const data = await response.json();
      setEnglishText(data.responseData.translatedText);
    } catch (error) {
      console.error('Processing error:', error);
      setEnglishText('Processing failed. Please try again.');
    }
    
    setLoading(false);
  };

  return (
    <div className="app-container">
      <div className="card">
        <h1>Tamil Handwriting Translator</h1>
        <p className="subtitle">Upload handwritten Tamil text to translate to English</p>

        <div className="upload-section">
          <input 
            type="file" 
            accept="image/*" 
            onChange={handleImageChange}
            id="file-input"
          />
          <label htmlFor="file-input" className="upload-label">
            {preview ? 'Change Image' : 'Upload Image'}
          </label>
        </div>

        {preview && (
          <div className="preview">
            <img src={preview} alt="Preview" />
          </div>
        )}

        <button 
          onClick={processImage} 
          disabled={loading || !image}
          className="translate-btn"
        >
          {loading ? 'Processing...' : 'Convert to English'}
        </button>

        {tamilText && (
          <div className="output-section">
            <label>Tamil Text</label>
            <div className="output-text">{tamilText}</div>
          </div>
        )}

        {englishText && (
          <div className="output-section">
            <label>English Translation</label>
            <div className="output-text">{englishText}</div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;