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

def initialize_model(api_key):
    """Initialize the Gemini model with specific configurations"""
    try:
        genai.configure(api_key=api_key)
        
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }

        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            generation_config=generation_config,
            system_instruction="""Act as a retina specialist, by looking at provided OCT and/or fundus photographs - 
            identify landmarks/biomarkers and then tell the initial diagnosis, the differentials, 
            the pertinent investigations and the management strategy. Keep it formal and brief. 
            Provide a level of certainty at the end.""",
            tools=[
                genai.protos.Tool(
                    google_search=genai.protos.Tool.GoogleSearch(),
                ),
            ],
        )
        
        # Initialize chat session
        chat_session = model.start_chat(history=[])
        
        return model, chat_session
    except Exception as e:
        raise Exception(f"Model initialization failed: {str(e)}")

# Get query parameters
query_params = st.query_params()
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
                # Prepare images
                image_parts = []
                for uploaded_file in uploaded_files:
                    image = Image.open(uploaded_file)
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    byte_stream = io.BytesIO()
                    image.save(byte_stream, format='JPEG')
                    image_bytes = byte_stream.getvalue()
                    image_parts.append({
                        "mime_type": "image/jpeg",
                        "data": image_bytes
                    })

                # Prepare message
                message = "Please analyze these retinal images."
                if case_notes:
                    message += f"\n\nClinical Notes: {case_notes}"

                # Get response using chat session
                response = st.session_state.chat_session.send_message(
                    [message] + image_parts
                )
                
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
