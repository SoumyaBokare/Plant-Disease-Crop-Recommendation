# 🌱 Plant Disease Classification & Crop Recommendation System

A comprehensive web application that combines AI-powered plant disease detection with intelligent crop recommendation features to help farmers and agricultural professionals make informed decisions.

## 🚀 Features

### 🔍 Plant Disease Classification
- **AI-Powered Detection**: Upload images of plant leaves to identify diseases using a trained TensorFlow/Keras model
- **Instant Results**: Get immediate diagnosis with confidence scores
- **Visual Feedback**: Clear display of uploaded images alongside predictions
- **Support for Multiple Crops**: Trained on various plant species and common diseases

### 🌾 Crop Recommendation System
- **Smart Recommendations**: Get crop suggestions based on soil and environmental conditions
- **Multiple Input Parameters**:
  - Nitrogen (N), Phosphorus (P), Potassium (K) levels
  - Temperature and humidity
  - pH levels
  - Rainfall data
- **Machine Learning Powered**: Uses Random Forest and Decision Tree algorithms for accurate predictions

### 🤖 Interactive Chatbot
- **Agricultural Assistance**: Get answers to farming and crop-related questions
- **Powered by Ollama**: Local LLM integration for intelligent responses
- **Context-Aware**: Understands agricultural terminology and provides relevant advice

## 📋 Prerequisites

- Python 3.8 or higher
- TensorFlow 2.15.0+
- Streamlit 1.28.0+
- Required Python packages (see requirements.txt)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/SoumyaBokare/Plant-Disease-Crop-Recommendation.git
   cd Plant-Disease-Crop-Recommendation
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install TensorFlow** (if not included in requirements)
   ```bash
   pip install tensorflow
   ```

## 🚀 Running the Application

1. **Start the Streamlit app**
   ```bash
   streamlit run app_api.py
   ```

2. **Access the application**
   - Local URL: http://localhost:8501
   - Network URL: http://[your-ip]:8501

## 📁 Project Structure

```
├── app_api.py                 # Main Streamlit application
├── app_mac.py                 # macOS-specific version
├── plant_disease_model.h5     # Trained plant disease classification model
├── crop_recommendation_model.pkl # Trained crop recommendation model
├── requirements.txt           # Python dependencies
├── CHATBOT_SETUP.md          # Chatbot configuration guide
└── Plant_Disease_Classification_and_Crop_Recommendation.ipynb # Development notebook
```

## 🎯 How to Use

### Plant Disease Detection
1. Navigate to the "Plant Disease Classification" tab
2. Upload an image of a plant leaf (JPG, JPEG, PNG)
3. Click "Classify Disease" to get instant results
4. Review the prediction and confidence score

### Crop Recommendation
1. Go to the "Crop Recommendation" tab
2. Enter soil parameters:
   - NPK values (Nitrogen, Phosphorus, Potassium)
   - Temperature and humidity
   - pH level
   - Rainfall data
3. Click "Get Recommendation" for crop suggestions

### Chatbot Assistance
1. Use the "Agricultural Chatbot" tab
2. Ask questions about farming, crops, or agriculture
3. Get AI-powered responses and advice

## 🔧 Models Used

- **Plant Disease Classification**: Deep Learning CNN model trained on plant leaf images
- **Crop Recommendation**: Ensemble of Random Forest and Decision Tree algorithms
- **Chatbot**: Ollama-powered local language model integration

## 📊 Performance

- **Disease Classification**: High accuracy on common plant diseases
- **Crop Recommendation**: Optimized for various soil and climate conditions
- **Response Time**: Real-time predictions and recommendations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- TensorFlow and Keras teams for the deep learning framework
- Scikit-learn for machine learning algorithms
- Streamlit for the web application framework
- Agricultural datasets and research communities

## 📧 Contact

For questions, suggestions, or support:
- GitHub Issues: [Create an issue](https://github.com/SoumyaBokare/Plant-Disease-Crop-Recommendation/issues)
- Project Link: [Plant Disease & Crop Recommendation System](https://github.com/SoumyaBokare/Plant-Disease-Crop-Recommendation)

---

**Note**: This system is designed to assist agricultural decision-making but should not replace professional agricultural consultation for critical farming decisions.