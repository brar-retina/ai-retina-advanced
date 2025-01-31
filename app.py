import streamlit as st
from urllib.parse import quote, unquote
import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content
import os
from PIL import Image
import io

# Configure page
st.set_page_config(page_title="Retinal Image Analyzer", layout="wide")

# Initialize session state
if 'api_key_configured' not in st.session_state:
    st.session_state.api_key_configured = False
if 'chat_session' not in st.session_state:
    st.session_state.chat_session = None
if 'search_results' not in st.session_state:
    st.session_state.search_results = []

def initialize_model(api_key):
    """Initialize the Gemini model with specific configurations"""
    try:
        genai.configure(api_key=api_key)
        
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }

        safety_settings = {
            "HARASSMENT": "block_none",
            "HATE_SPEECH": "block_none",
            "SEXUALLY_EXPLICIT": "block_none",
            "DANGEROUS_CONTENT": "block_none",
        }

        model = genai.GenerativeModel(
            model_name="gemini-pro-vision",
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        # Initialize chat session with empty history
        chat_session = model.start_chat(history=[])
        
        return model, chat_session
    except Exception as e:
        raise Exception(f"Model initialization failed: {str(e)}")

def process_search_results(response):
    """Extract and format search results from the response"""
    search_results = []
    if hasattr(response, 'candidates') and response.candidates:
        for candidate in response.candidates:
            if hasattr(candidate, 'search_results') and candidate.search_results:
                for result in candidate.search_results:
                    search_results.append({
                        'title': result.title,
                        'url': result.url,
                        'snippet': result.snippet
                    })
    return search_results

# Get query parameters
query_params = st.experimental_get_query_params()
if 'api_key' in query_params and not st.session_state.api_key_configured:
    api_key = unquote(query_params['api_key'][0])
    try:
        model, chat_session = initialize_model(api_key)
        st.session_state.model = model
        st.session_state.chat_session = chat_session
        st.session_state.api_key_configured = True
        st.session_state.api_key = api_key
    except Exception as e:
        st.error(f"Error configuring API from URL: {str(e)}")

def configure_gemini(api_key):
    """Configure Gemini API with the provided key"""
    try:
        model, chat_session = initialize_model(api_key)
        st.session_state.model = model
        st.session_state.chat_session = chat_session
        st.session_state.api_key_configured = True
        st.session_state.api_key = api_key
        return True
    except Exception as e:
        st.error(f"Error configuring API: {str(e)}")
        return False

# Sidebar configuration
with st.sidebar:
    st.title("⚙️ Configuration")
    
    if not st.session_state.api_key_configured:
        api_key = st.text_input("Enter your Google API Key", type="password")
        if st.button("Configure API"):
            if api_key:
                configure_gemini(api_key)
            else:
                st.error("Please enter an API key")
    else:
        st.success("API Key configured!")
        
        base_url = st.experimental_get_query_params().get('base_url', [None])[0]
        if base_url is None:
            base_url = st.secrets.get("STREAMLIT_BASE_URL", "YOUR_DEPLOYED_URL")
        
        shareable_link = f"{base_url}?api_key={quote(st.session_state.api_key)}"
        st.markdown("### Shareable Link")
        st.code(shareable_link, language="text")
        st.markdown("⚠️ **Note**: This link contains your API key. Share securely!")

# Main app
st.title("🔬 Retinal Image Analyzer")
st.write("Upload retinal images (OCT/fundus) for AI-powered analysis")

# Main form
with st.form("analysis_form"):
    # Optional clinical notes
    case_notes = st.text_area(
        "Clinical Notes (optional)",
        height=100,
        placeholder="Enter any relevant clinical information..."
    )
    
    # Image upload
    uploaded_files = st.file_uploader(
        "Upload retinal images (OCT/fundus)",
        accept_multiple_files=True,
        type=['png', 'jpg', 'jpeg']
    )
    
    submit_button = st.form_submit_button("Analyze Images")

if submit_button:
    if not st.session_state.api_key_configured:
        st.error("Please configure your API key first")
    elif not uploaded_files:
        st.error("Please upload at least one image")
    else:
        try:
            with st.spinner("Analyzing images..."):
                # Prepare prompt
                prompt = "Please analyze these retinal images."
                if case_notes:
                    prompt += f"\n\nClinical Notes: {case_notes}"
                
                # Prepare images
                images = []
                for uploaded_file in uploaded_files:
                    image = Image.open(uploaded_file)
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    images.append(image)

                # Generate response
                response = st.session_state.model.generate_content([prompt, *images])
                response.resolve()
                
                # Display results
                st.success("Analysis complete!")
                st.markdown("### Analysis Results")
                st.write(response.text)
                
                # Display uploaded images
                st.markdown("### Uploaded Images")
                cols = st.columns(len(uploaded_files))
                for idx, uploaded_file in enumerate(uploaded_files):
                    with cols[idx]:
                        st.image(uploaded_file, caption=f"Image {idx + 1}")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Usage instructions
with st.sidebar:
    st.markdown("### How to use")
    st.markdown("""
    1. Enter your Google API key (or use shared link)
    2. Upload retinal images (OCT/fundus)
    3. Add any relevant clinical notes
    4. Click 'Analyze Images'
    """)
    
    st.markdown("### Privacy Notice")
    st.markdown("""
    - Ensure no PHI (Protected Health Information) is uploaded
    - Data is not stored and is only used for analysis
    - Use appropriate de-identification methods
    - API keys in URLs should be shared securely
    """)
