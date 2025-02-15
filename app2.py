import streamlit as st
from urllib.parse import quote, unquote
import google.generativeai as genai
from PIL import Image
import io

# Configure page
st.set_page_config(page_title="Retinal Image Analyzer", layout="wide")

# Initialize session state
if 'api_key_configured' not in st.session_state:
    st.session_state.api_key_configured = False
if 'chat_session' not in st.session_state:
    st.session_state.chat_session = None

AVAILABLE_MODELS = {
    "Gemini 1.5 Flash": "gemini-1.5-flash",
    "Gemini 2.0 Flash (NEW)": "gemini-2.0-flash-exp"
}

def initialize_model(api_key, model_name):
    """Initialize the Gemini model with specific configurations"""
    try:
        genai.configure(api_key=api_key)
        
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }

        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config
        )
        
        # Initialize chat session
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
    except Exception as e:
        raise Exception(f"Model initialization failed: {str(e)}")

def process_image(uploaded_file):
    """Process uploaded image file into format required by Gemini"""
    image = Image.open(uploaded_file)
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image if too large
    max_size = 1600
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = tuple(int(dim * ratio) for dim in image.size)
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    # Convert to bytes
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    image_bytes = buffer.getvalue()
    
    return {
        "mime_type": "image/jpeg",
        "data": image_bytes
    }

# Get query parameters
if 'api_key' in st.query_params and not st.session_state.api_key_configured:
    api_key = unquote(st.query_params['api_key'])
    selected_model = st.query_params.get('model', "gemini-1.5-flash")
    try:
        model, chat_session = initialize_model(api_key, selected_model)
        st.session_state.model = model
        st.session_state.chat_session = chat_session
        st.session_state.api_key_configured = True
        st.session_state.api_key = api_key
        st.session_state.selected_model = selected_model
    except Exception as e:
        st.error(f"Error configuring API from URL: {str(e)}")

# Sidebar configuration
with st.sidebar:
    st.title("⚙️ Configuration")
    
    if not st.session_state.api_key_configured:
        api_key = st.text_input("Enter your Google API Key", type="password")
        
        # Add model selection dropdown
        selected_model_name = st.selectbox(
            "Select Gemini Model",
            options=list(AVAILABLE_MODELS.keys()),
            index=0
        )
        selected_model = AVAILABLE_MODELS[selected_model_name]
        
        if st.button("Configure API"):
            if api_key:
                try:
                    model, chat_session = initialize_model(api_key, selected_model)
                    st.session_state.model = model
                    st.session_state.chat_session = chat_session
                    st.session_state.api_key_configured = True
                    st.session_state.api_key = api_key
                    st.session_state.selected_model = selected_model
                except Exception as e:
                    st.error(f"Error configuring API: {str(e)}")
            else:
                st.error("Please enter an API key")
    else:
        st.success("API Key configured!")
        st.info(f"Using model: {next(name for name, model in AVAILABLE_MODELS.items() if model == st.session_state.selected_model)}")
        
        base_url = st.query_params.get('base_url', "ai-retina3advanced.streamlit.app")
        shareable_link = f"{base_url}?api_key={quote(st.session_state.api_key)}&model={st.session_state.selected_model}"
        st.markdown("### Shareable Link")
        st.code(shareable_link, language="text")
        st.markdown("⚠️ **Note**: This link contains your API key. Share securely!")

# Main app
st.title("🔬 Retinal Image Analyzer")
st.markdown("Upload retinal images (OCT/fundus) for AI-powered analysis. ©Anand Singh Brar  [**@brar_retina**](https://www.instagram.com/brar_retina)")

# Main form
with st.form("analysis_form"):
    case_notes = st.text_area(
        "Case Scenario",
        height=100,
        placeholder="Enter any relevant clinical information... *bonus* use this box to ask follow up questions after the preliminary analysis report."
    )
    
    uploaded_files = st.file_uploader(
        "Upload retinal images (OCT/fundus)",
        accept_multiple_files=True,
        type=['png', 'jpg', 'jpeg']
    )
    
    submit_button = st.form_submit_button("Analyze Images/Generate Response")

if submit_button:
    if not st.session_state.api_key_configured:
        st.error("Please configure your API key first")
    elif not uploaded_files:
        st.error("Please upload at least one image")
    else:
        try:
            with st.spinner("Analyzing images..."):
                # Process images
                processed_images = [process_image(file) for file in uploaded_files]
                
                # Prepare message
                prompt = "Please analyze these retinal images."
                if case_notes:
                    prompt += f"\n\nClinical Notes: {case_notes}"
                
                # Create message parts
                message_parts = [prompt]
                message_parts.extend(processed_images)
                
                # Send message and get response
                response = st.session_state.chat_session.send_message(message_parts)
                
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
            st.error("Full error details:", exc_info=True)

# Usage instructions
with st.sidebar:
    st.markdown("### How to use")
    st.markdown("""
    1. Enter your Google API key (or use shared link)
    2. Select Gemini model version
    3. Upload retinal images (OCT/fundus)
    4. Add any relevant clinical notes
    5. Click 'Analyze Images'
    """)
    
    st.markdown("### Privacy Notice")
    st.markdown("""
    - Ensure no PHI (Protected Health Information) is uploaded
    - Data is not stored and is only used for analysis
    - Use appropriate de-identification methods
    - API keys in URLs should be shared securely
    """)
