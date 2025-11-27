# Agricultural AI Chatbot Setup Guide

## 🤖 About the Chatbot

The Agricultural AI Assistant is integrated into your Plant Disease & Crop Recommendation System. It specializes in:

- **Climate Knowledge**: Understanding how weather patterns affect crop growth
- **Crop Guidance**: Recommendations for crop selection based on environmental factors
- **Seasonal Planning**: Planting and harvesting calendars
- **Sustainable Farming**: Best practices for eco-friendly agriculture
- **Soil Management**: Soil requirements and crop compatibility

## 🚀 Quick Setup

### Step 1: Install Ollama
1. Download Ollama from: https://ollama.ai/
2. Install it on your system
3. Open terminal/command prompt and run:
   ```bash
   ollama pull llama3.2
   ```

### Step 2: Start Ollama Server
```bash
ollama serve
```
*Keep this terminal open while using the chatbot*

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Application
```bash
streamlit run app_api.py
```

## 📱 How to Use

1. **Navigate to the "🤖 Agricultural AI Chat" tab**
2. **Select your preferred AI model** (llama3.2 recommended)
3. **Test the connection** using the "🔍 Test Ollama Connection" button
4. **Start chatting!** 

### Quick Start Questions:
- "What crops grow best in hot, humid climates?"
- "How does rainfall affect crop selection?"
- "What are the soil requirements for rice cultivation?"
- "Which crops are drought resistant?"

## 🎯 Example Conversations

**Climate-focused:**
- User: *"I live in a region with high temperature and low rainfall. What crops should I consider?"*
- AI: *Provides drought-resistant crop suggestions with climate reasoning*

**Crop Knowledge:**
- User: *"Why is my area suitable for wheat cultivation?"*
- AI: *Explains temperature, rainfall, and soil requirements for wheat*

**Seasonal Planning:**
- User: *"When should I plant corn in tropical climate?"*
- AI: *Provides planting calendar considering monsoon patterns*

## 🔧 Troubleshooting

### Ollama Connection Issues:
- Ensure Ollama is running: `ollama serve`
- Check if models are downloaded: `ollama list`
- Verify port 11434 is available

### Model Performance:
- **llama3.2**: Best balance of speed and accuracy
- **mistral**: Good for quick responses
- **llama3.1**: More detailed but slower responses

### Chat History:
- Use "🗑️ Clear Chat" to start fresh conversations
- Chat history persists during the session but resets on page refresh

## 🌟 Features

### Smart Context Awareness
- Remembers recent conversation context
- Provides coherent multi-turn conversations
- Builds on previous questions and answers

### Quick Questions
- Pre-defined common agricultural questions
- One-click access to popular topics
- Instant responses without typing

### Model Selection
- Choose from multiple AI models
- Each model has different strengths
- Switch models mid-conversation if needed

## 💡 Tips for Best Results

1. **Be specific**: Instead of "crop advice", ask "best crops for sandy soil in dry climate"
2. **Provide context**: Mention your location, climate type, or current season
3. **Ask follow-ups**: Build on previous answers for more detailed guidance
4. **Use examples**: "Like rice but for drier conditions" helps AI understand better

## 🔄 Integration with Main App

The chatbot works alongside your existing features:

- **Complements Disease Detection**: Ask about treatment after disease identification
- **Enhances Crop Recommendations**: Get detailed explanations for recommended crops
- **Climate Context**: Understand why weather API suggests certain crops

---

**Need Help?** Test the Ollama connection first, then try asking simple questions to verify everything works correctly.