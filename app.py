import streamlit as st
from urllib.parse import quote, unquote
import google.generativeai as genai
from PIL import Image
import io
import requests  # Add this for DeepSeek API calls

# Configure page
st.set_page_config(page_title="Retinal Image Analyzer", layout="wide")

# Initialize session state
if 'api_key_configured' not in st.session_state:
    st.session_state.api_key_configured = False
if 'chat_session' not in st.session_state:
    st.session_state.chat_session = None
if 'model_type' not in st.session_state:
    st.session_state.model_type = "gemini"
if 'generation_config' not in st.session_state:
    st.session_state.generation_config = {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
    }

AVAILABLE_MODELS = {
    "Gemini": {
        "Gemini 1.5 Flash": "gemini-1.5-flash",
        "Gemini 2.0 Flash (NEW)": "gemini-2.0-flash-exp"
    },
    "DeepSeek": {
        "DeepSeek-Vision": "deepseek-vision-r1"
    },
    "Custom": {}
}

def initialize_model(api_key, model_name, model_type="gemini"):
    """Initialize the selected model with specific configurations"""
    try:
        if model_type == "gemini":
            genai.configure(api_key=api_key)
            
            model = genai.GenerativeModel(
                model_name=model_name,
                generation_config=st.session_state.generation_config
            )
            
            chat_session = model.start_chat(history=[])
            
            # Send system prompt
            chat_session.send_message("""Act as a retina specialist. When analyzing OCT and/or fundus photographs:
            1. Identify key landmarks and biomarkers
            2. Provide initial diagnosis
            3. List differential diagnoses
            4. Suggest pertinent investigations
            5. Outline management strategy
            Keep responses formal and brief. End with confidence level.""")
            
            return model, chat_session
        
        elif model_type == "deepseek":
            # Initialize DeepSeek (placeholder for actual implementation)
            # You'll need to implement the actual DeepSeek API integration
            return None, None
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
    except Exception as e:
        raise Exception(f"Model initialization failed: {str(e)}")

# Add model configuration sidebar
def show_model_config():
    with st.sidebar:
        st.markdown("### Model Configuration")
        
        # Model type selection
        model_type = st.radio(
            "Select Model Type",
            ["Gemini", "DeepSeek", "Custom"],
            key="model_type"
        )
        
        # Model selection based on type
        if model_type == "Custom":
            custom_model = st.text_input(
                "Enter Custom Model Name",
                placeholder="e.g., custom-model-name"
            )
            if custom_model:
                AVAILABLE_MODELS["Custom"][custom_model] = custom_model
        
        model_options = AVAILABLE_MODELS[model_type].keys()
        selected_model_name = st.selectbox(
            "Select Model",
            options=list(model_options),
            index=0 if model_options else None
        )
        
        # Model parameters configuration
        st.markdown("### Generation Parameters")
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.generation_config["temperature"],
            step=0.1
        )
        
        top_p = st.slider(
            "Top P",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.generation_config["top_p"],
            step=0.05
        )
        
        top_k = st.slider(
            "Top K",
            min_value=1,
            max_value=100,
            value=st.session_state.generation_config["top_k"]
        )
        
        max_tokens = st.slider(
            "Max Output Tokens",
            min_value=1000,
            max_value=16000,
            value=st.session_state.generation_config["max_output_tokens"],
            step=1000
        )
        
        # Update generation config
        st.session_state.generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_output_tokens": max_tokens,
        }
        
        return model_type, selected_model_name

# Update the sidebar configuration section
with st.sidebar:
    st.title("⚙️ Configuration")
    
    if not st.session_state.api_key_configured:
        model_type, selected_model_name = show_model_config()
        
        if model_type == "gemini":
            api_key = st.text_input("Enter your Google API Key", type="password")
        elif model_type == "deepseek":
            api_key = st.text_input("Enter your DeepSeek API Key", type="password")
        else:
            api_key = st.text_input("Enter your API Key", type="password")
        
        if st.button("Configure API"):
            if api_key:
                try:
                    selected_model = AVAILABLE_MODELS[model_type][selected_model_name]
                    model, chat_session = initialize_model(api_key, selected_model, model_type)
                    st.session_state.model = model
                    st.session_state.chat_session = chat_session
                    st.session_state.api_key_configured = True
                    st.session_state.api_key = api_key
                    st.session_state.selected_model = selected_model
                    st.session_state.model_type = model_type
                except Exception as e:
                    st.error(f"Error configuring API: {str(e)}")
            else:
                st.error("Please enter an API key")
    else:
        st.success("API Key configured!")
        model_name = next(name for name, model in AVAILABLE_MODELS[st.session_state.model_type].items() 
                         if model == st.session_state.selected_model)
        st.info(f"Using model: {model_name}")
        
        # Show current configuration
        st.markdown("### Current Configuration")
        st.json(st.session_state.generation_config)
        
        base_url = st.query_params.get('base_url', "ai-retina3advanced.streamlit.app")
        shareable_link = f"{base_url}?api_key={quote(st.session_state.api_key)}&model={st.session_state.selected_model}"
        st.markdown("### Shareable Link")
        st.code(shareable_link, language="text")
        st.markdown("⚠️ **Note**: This link contains your API key. Share securely!")
