import streamlit as st
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import os
import requests
import json
from datetime import datetime
import json
from datetime import datetime

# ---- TensorFlow import with platform checks ----
TENSORFLOW_AVAILABLE = False
tf = None
try:
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
    import tensorflow as tf_module
    tf = tf_module
    TENSORFLOW_AVAILABLE = True
except Exception as e:
    st.sidebar.warning(f"⚠️ TensorFlow not available: {str(e)[:100]}...")
    TENSORFLOW_AVAILABLE = False

st.set_page_config(
    page_title="Plant Disease & Crop Recommendation System",
    page_icon="🌱",
    layout="wide"
)

@st.cache_resource
def load_models():
    plant_model = None
    crop_model = None
    if TENSORFLOW_AVAILABLE and tf is not None:
        try:
            plant_model = tf.keras.models.load_model('plant_disease_model.h5')
            st.sidebar.success("✅ Plant disease model loaded!")
        except Exception as e:
            st.sidebar.error(f"❌ Plant disease model failed: {str(e)[:100]}...")
            plant_model = None
    try:
        with open('crop_recommendation_model.pkl', 'rb') as file:
            crop_model = pickle.load(file)
        st.sidebar.success("✅ Crop recommendation model loaded!")
    except Exception as e:
        st.sidebar.error(f"❌ Crop model failed: {str(e)[:100]}...")
        crop_model = None
    return plant_model, crop_model

# Ollama chatbot functions
def query_ollama(prompt, model_name="llama3.2"):
    """Query Ollama API"""
    try:
        url = "http://localhost:11434/api/generate"
        data = {
            "model": model_name,
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(url, json=data, timeout=30)
        if response.status_code == 200:
            return response.json().get('response', 'No response received')
        else:
            return f"Error: {response.status_code} - {response.text}"
    except requests.exceptions.ConnectionError:
        return "❌ Cannot connect to Ollama. Please ensure Ollama is running on localhost:11434"
    except Exception as e:
        return f"❌ Error querying Ollama: {str(e)}"

def create_agricultural_prompt(user_question, context=""):
    """Create a specialized prompt for agricultural chatbot"""
    base_prompt = f"""You are an expert Agricultural AI Assistant specializing in crop knowledge and climate science. 
Your role is to provide accurate, practical advice to farmers and agricultural professionals.

Key areas of expertise:
1. Climate requirements for different crops
2. Soil conditions and crop suitability
3. Seasonal planting and harvesting advice
4. Weather patterns and agricultural planning
5. Crop rotation and sustainable farming practices
6. Regional agricultural recommendations

Guidelines:
- Provide specific, actionable advice
- Include scientific reasoning when relevant
- Consider regional variations in climate
- Focus on practical farming applications
- Suggest climate-resilient alternatives when appropriate

{context}

User Question: {user_question}

Response:"""
    return base_prompt

def initialize_chat_history():
    """Initialize chat history in session state"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        # Add welcome message
        welcome_msg = {
            "role": "assistant", 
            "content": "🌱 Hello! I'm your Agricultural AI Assistant. I can help you with:\n\n• Climate requirements for different crops\n• Soil conditions and crop suitability\n• Seasonal planting advice\n• Weather patterns and farming\n• Sustainable agriculture practices\n\nWhat would you like to know about crops and climate?",
            "timestamp": datetime.now()
        }
        st.session_state.chat_history.append(welcome_msg)

# Ollama chatbot functions
def query_ollama(prompt, model_name="llama3.2"):
    """Query Ollama API"""
    try:
        url = "http://localhost:11434/api/generate"
        data = {
            "model": model_name,
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(url, json=data, timeout=30)
        if response.status_code == 200:
            return response.json().get('response', 'No response received')
        else:
            return f"Error: {response.status_code} - {response.text}"
    except requests.exceptions.ConnectionError:
        return "❌ Cannot connect to Ollama. Please ensure Ollama is running on localhost:11434"
    except Exception as e:
        return f"❌ Error querying Ollama: {str(e)}"

def create_agricultural_prompt(user_question, context=""):
    """Create a specialized prompt for agricultural chatbot"""
    base_prompt = f"""You are an expert Agricultural AI Assistant specializing in crop knowledge and climate science. 
Your role is to provide accurate, practical advice to farmers and agricultural professionals.

Key areas of expertise:
1. Climate requirements for different crops
2. Soil conditions and crop suitability
3. Seasonal planting and harvesting advice
4. Weather patterns and agricultural planning
5. Crop rotation and sustainable farming practices
6. Regional agricultural recommendations

Guidelines:
- Provide specific, actionable advice
- Include scientific reasoning when relevant
- Consider regional variations in climate
- Focus on practical farming applications
- Suggest climate-resilient alternatives when appropriate

{context}

User Question: {user_question}

Response:"""
    return base_prompt

def initialize_chat_history():
    """Initialize chat history in session state"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        # Add welcome message
        welcome_msg = {
            "role": "assistant", 
            "content": "🌱 Hello! I'm your Agricultural AI Assistant. I can help you with:\n\n• Climate requirements for different crops\n• Soil conditions and crop suitability\n• Seasonal planting advice\n• Weather patterns and farming\n• Sustainable agriculture practices\n\nWhat would you like to know about crops and climate?",
            "timestamp": datetime.now()
        }
        st.session_state.chat_history.append(welcome_msg)

PLANT_DISEASE_CLASSES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy',
    'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch',
    'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

def preprocess_image(image):
    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_disease(model, image):
    if not TENSORFLOW_AVAILABLE or model is None:
        return "TensorFlow not available", 0.0
    try:
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image, verbose=0)
        predicted_class_idx = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        if predicted_class_idx < len(PLANT_DISEASE_CLASSES):
            predicted_class = PLANT_DISEASE_CLASSES[predicted_class_idx]
        else:
            predicted_class = f"Class_{predicted_class_idx}"
        return predicted_class, confidence
    except Exception as e:
        return f"Prediction error: {str(e)}", 0.0

def predict_crop(model, features):
    if model is None:
        return "Model not available"
    try:
        prediction = model.predict([features])
        return prediction[0]
    except Exception as e:
        return f"Prediction error: {str(e)}"

# ====== Weather-Fetch function using OpenWeatherMap ======
def get_weather_params(city, api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={api_key}&units=metric"
    try:
        r = requests.get(url)
        data = r.json()
        temp = data['main']['temp'] if 'main' in data else 25.0
        humidity = data['main']['humidity'] if 'main' in data else 60.0
        rainfall = data.get('rain', {}).get('1h', 50.0)
        return temp, humidity, rainfall, None
    except Exception as e:
        return 25.0, 60.0, 50.0, str(e)

def main():
    st.title("🌱 Plant Disease Classification & Crop Recommendation System")
    st.markdown("---")
    plant_model, crop_model = load_models()
    st.sidebar.header("🔧 System Status")
    st.sidebar.info(f"TensorFlow: {'✅ Available' if TENSORFLOW_AVAILABLE else '❌ Not Available'}")
    st.sidebar.info(f"Plant Disease Model: {'✅ Loaded' if plant_model else '❌ Not Loaded'}")
    st.sidebar.info(f"Crop Model: {'✅ Loaded' if crop_model else '❌ Not Loaded'}")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Plant Disease Classification",
        "🌾 Manual Crop Recommendation",
        "⚡ Auto-Fill Crop Recommendation (Weather API)",
        "🤖 Agricultural AI Chat"
    ])

    # ---- Plant Disease Tab ----
    with tab1:
        st.header("Plant Disease Classification")
        if not TENSORFLOW_AVAILABLE or plant_model is None:
            st.error("🚫 **Plant Disease Classification is currently unavailable**")
            if not TENSORFLOW_AVAILABLE:
                st.info("**Reason:** TensorFlow not available or failed to load")
            else:
                st.info("**Reason:** Could not load the disease classification model")
            st.markdown("**💡 Alternative Solutions:**")
            st.markdown("- Use Google Colab to run your original notebook")
            st.markdown("- Try running on a Linux or Windows machine")
            st.markdown("- Use the Crop Recommendation feature below (fully functional)")
        else:
            st.success("✅ **Plant Disease Classification is ready!**")
            st.write("Upload an image of a plant leaf to detect diseases")
            uploaded_file = st.file_uploader("Choose a plant image...", type=['jpg', 'jpeg', 'png'])
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Uploaded Image", use_container_width=True)
                with col2:
                    if st.button("🔍 Analyze Disease", type="primary"):
                        with st.spinner("Analyzing image..."):
                            predicted_class, confidence = predict_disease(plant_model, image)
                            if confidence > 0:
                                st.success("Analysis Complete!")
                                clean_class = predicted_class.replace('___', ' - ').replace('_', ' ')
                                st.markdown(f"### 📊 Results:")
                                st.markdown(f"**Predicted Class:** {clean_class}")
                                st.markdown(f"**Confidence:** {confidence:.2f}%")
                                st.progress(confidence / 100)
                                if 'healthy' in predicted_class.lower():
                                    st.success("✅ Plant appears to be healthy!")
                                else:
                                    st.warning("⚠️ Disease detected. Consider consulting an agricultural expert.")
                            else:
                                st.error(f"❌ {predicted_class}")

    # ---- Manual Crop Recommendation Tab ----
    with tab2:
        st.header("Crop Recommendation System (Manual Entry)")
        if crop_model is None:
            st.error("🚫 **Crop Recommendation is currently unavailable**")
            st.info("Could not load the crop recommendation model file.")
            st.markdown("**Please ensure:** `crop_recommendation_model.pkl` is in the same directory")
        else:
            st.success("✅ **Crop Recommendation System is ready!**")
            st.write("Enter soil and climate parameters to get crop recommendations")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📊 Soil Parameters")
                nitrogen = st.slider("Nitrogen (N)", 0, 140, 50)
                phosphorus = st.slider("Phosphorus (P)", 5, 145, 50)
                potassium = st.slider("Potassium (K)", 5, 205, 50)
                ph = st.slider("pH Level", 3.5, 10.0, 6.5, step=0.1)
            with col2:
                st.subheader("🌤️ Climate Parameters")
                temperature = st.slider("Temperature (°C)", 8.0, 44.0, 25.0, step=0.5)
                humidity = st.slider("Humidity (%)", 14.0, 100.0, 65.0, step=0.5)
                rainfall = st.slider("Rainfall (mm)", 20.0, 300.0, 100.0, step=5.0)
            features = [nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🌾 Get Crop Recommendation", type="primary", use_container_width=True):
                    with st.spinner("Analyzing soil and climate conditions..."):
                        recommended_crop = predict_crop(crop_model, features)
                        if "error" not in str(recommended_crop).lower():
                            st.success("Analysis Complete!")
                            st.markdown("### 🎯 Recommended Crop:")
                            st.markdown(f"## **{recommended_crop.title()}**")
                            st.markdown("### 📋 Input Summary:")
                            input_df = pd.DataFrame({
                                'Parameter': ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH', 'Rainfall'],
                                'Value': [f"{nitrogen}", f"{phosphorus}", f"{potassium}", f"{temperature}°C", f"{humidity}%", f"{ph}", f"{rainfall}mm"]
                            })
                            st.table(input_df)
                        else:
                            st.error(f"❌ {recommended_crop}")

    # ---- Auto-Fill (Weather API) Crop Recommendation Tab ----
    with tab3:
        st.header("⚡ Auto-Fill Crop Recommendation (Weather API)")
        st.markdown("""
        Enter your city name (India) below. Live temperature, humidity, and rainfall will be fetched automatically.<br>
        You can override soil parameters for more precise results.
        """, unsafe_allow_html=True)
        api_key = st.text_input(
            "🔑 Enter your OpenWeatherMap API Key (get free from https://openweathermap.org/api)",
            "",
            type="password"
        )
        city = st.text_input("🏙️ Enter the City Name (India, e.g., Delhi, Mumbai, Kolkata, Bengaluru)", "")
        # Editable (slider) defaults for soil parameters
        nitrogen_default = st.slider("Nitrogen (N, Weather Tab)", 0, 140, 50)
        phosphorus_default = st.slider("Phosphorus (P, Weather Tab)", 5, 145, 50)
        potassium_default = st.slider("Potassium (K, Weather Tab)", 5, 205, 50)
        ph_default = st.slider("pH Level (Weather Tab)", 3.5, 10.0, 6.5, step=0.1)

        if st.button("⚡ Auto-Fill and Get Crop Recommendation", type="primary", use_container_width=True):
            if not api_key or not city:
                st.error("Please provide both API key and city name.")
            else:
                with st.spinner(f"Fetching weather data for {city}..."):
                    temp, humidity, rainfall, err = get_weather_params(city, api_key)
                    if err:
                        st.warning(f"Issue fetching weather data: {err}")
                    else:
                        st.success(f"Fetched: {city} — Temperature: {temp}°C, Humidity: {humidity}%, Rainfall: {rainfall}mm")
                        features = [
                            nitrogen_default, phosphorus_default, potassium_default,
                            temp, humidity, ph_default, rainfall
                        ]
                        recommended_crop = predict_crop(crop_model, features)
                        if "error" not in str(recommended_crop).lower():
                            st.markdown("### 🎯 Recommended Crop (Weather-Based):")
                            st.markdown(f"## **{recommended_crop.title()}**")
                            st.markdown("### 📋 Input Summary:")
                            input_df = pd.DataFrame({
                                'Parameter': [
                                    'Nitrogen', 'Phosphorus', 'Potassium', 'Temperature',
                                    'Humidity', 'pH', 'Rainfall'
                                ],
                                'Value': [
                                    f"{nitrogen_default}", f"{phosphorus_default}", f"{potassium_default}",
                                    f"{temp}°C", f"{humidity}%", f"{ph_default}", f"{rainfall}mm"
                                ]
                            })
                            st.table(input_df)
                        else:
                            st.error(f"❌ {recommended_crop}")

    # ---- Agricultural AI Chatbot Tab ----
    with tab4:
        st.header("🤖 Agricultural AI Assistant")
        st.markdown("""
        Ask me anything about:
        • **Climate Requirements** for different crops
        • **Soil Conditions** and crop suitability  
        • **Seasonal Planning** and planting calendars
        • **Weather Patterns** and agricultural planning
        • **Sustainable Farming** practices
        """)
        
        # Initialize chat history
        initialize_chat_history()
        
        # Ollama model selection
        col1, col2 = st.columns([3, 1])
        with col2:
            model_options = ["llama3.2", "llama3.1", "mistral", "codellama", "gemma2"]
            selected_model = st.selectbox("🤖 Model", model_options, index=0)
            
            # Clear chat button
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                initialize_chat_history()
                st.rerun()
        
        # Chat container
        chat_container = st.container()
        
        # Display chat history
        with chat_container:
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    with st.chat_message("user"):
                        st.write(f"**You:** {message['content']}")
                        st.caption(f"_{message['timestamp'].strftime('%H:%M:%S')}_")
                else:
                    with st.chat_message("assistant"):
                        st.write(f"**AI Assistant:** {message['content']}")
                        st.caption(f"_{message['timestamp'].strftime('%H:%M:%S')}_")
        
        # Chat input
        user_input = st.chat_input("Ask about crops, climate, or farming...")
        
        if user_input:
            # Add user message to history
            user_msg = {
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now()
            }
            st.session_state.chat_history.append(user_msg)
            
            # Generate context from recent conversation
            recent_context = ""
            if len(st.session_state.chat_history) > 2:
                recent_messages = st.session_state.chat_history[-4:]  # Last 2 exchanges
                context_parts = []
                for msg in recent_messages:
                    context_parts.append(f"{msg['role'].title()}: {msg['content'][:200]}")
                recent_context = f"Recent conversation context:\n" + "\n".join(context_parts) + "\n\n"
            
            # Create specialized prompt
            agricultural_prompt = create_agricultural_prompt(user_input, recent_context)
            
            # Show thinking indicator
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    # Query Ollama
                    response = query_ollama(agricultural_prompt, selected_model)
                    
                    # Add assistant response to history
                    assistant_msg = {
                        "role": "assistant",
                        "content": response,
                        "timestamp": datetime.now()
                    }
                    st.session_state.chat_history.append(assistant_msg)
                    
                    # Display response
                    st.write(f"**AI Assistant:** {response}")
                    st.caption(f"_{assistant_msg['timestamp'].strftime('%H:%M:%S')}_")
            
            # Auto-scroll to bottom
            st.rerun()
        
        # Quick question buttons
        st.markdown("### 💡 Quick Questions:")
        col1, col2, col3 = st.columns(3)
        
        quick_questions = [
            "What crops grow best in hot, humid climates?",
            "How does rainfall affect crop selection?",
            "What are the soil requirements for rice cultivation?",
            "Which crops are drought resistant?",
            "What is the best planting season for wheat?",
            "How does temperature affect crop growth?"
        ]
        
        for i, question in enumerate(quick_questions):
            col = col1 if i % 3 == 0 else (col2 if i % 3 == 1 else col3)
            with col:
                if st.button(question, key=f"quick_q_{i}", use_container_width=True):
                    # Simulate user input
                    user_msg = {
                        "role": "user",
                        "content": question,
                        "timestamp": datetime.now()
                    }
                    st.session_state.chat_history.append(user_msg)
                    
                    # Generate response
                    agricultural_prompt = create_agricultural_prompt(question)
                    response = query_ollama(agricultural_prompt, selected_model)
                    
                    assistant_msg = {
                        "role": "assistant",
                        "content": response,
                        "timestamp": datetime.now()
                    }
                    st.session_state.chat_history.append(assistant_msg)
                    st.rerun()
        
        # Ollama status check
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("**💡 Tip:** Make sure Ollama is running locally for the chatbot to work.")
        with col2:
            if st.button("🔍 Test Ollama Connection"):
                test_response = query_ollama("Hello", selected_model)
                if "Cannot connect" in test_response or "Error" in test_response:
                    st.error("❌ Ollama not accessible")
                else:
                    st.success("✅ Ollama connected")

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>🌱 Built with Streamlit | Plant Disease Classification & Crop Recommendation System</p>
        <p><small>TensorFlow Status: {}</small></p>
    </div>
    """.format("Available" if TENSORFLOW_AVAILABLE else "Not Available"), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
