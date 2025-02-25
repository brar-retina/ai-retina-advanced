import streamlit as st
from urllib.parse import quote, unquote
import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content
from PIL import Image
import io

# Configure page
st.set_page_config(page_title="Retinal Image Analyzer v2.0", layout="wide")

# Initialize session state
if 'api_key_configured' not in st.session_state:
    st.session_state.api_key_configured = False
if 'chat_session' not in st.session_state:
    st.session_state.chat_session = None
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'generation_config' not in st.session_state:
    st.session_state.generation_config = {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
    }
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

AVAILABLE_MODELS = {
    "Gemini 2.0 Flash": "gemini-2.0-flash-exp",
    "Gemini 2.0 Flash Thinking (Exp)": "gemini-2.0-flash-thinking-exp-01-21",
    "Gemini 2.0 Pro": "gemini-2.0-pro-exp-02-05",
    "LearnLM 1.5 Pro (Exp)": "learnlm-1.5-pro-experimental"
}

def initialize_model(api_key, model_name, generation_config=None):
    """Initialize the Gemini model with specific configurations"""
    try:
        genai.configure(api_key=api_key)
        
        # Use provided generation config or default from session state
        config = generation_config if generation_config else st.session_state.generation_config
        
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=config
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

def update_model_parameters():
    """Update model parameters without reinitializing the entire session"""
    if st.session_state.api_key_configured:
        try:
            # Create a new model with updated parameters
            model = genai.GenerativeModel(
                model_name=st.session_state.selected_model,
                generation_config=st.session_state.generation_config
            )
            
            # Keep chat history but update the model
            history = st.session_state.chat_session.history
            chat_session = model.start_chat(history=history)
            
            st.session_state.model = model
            st.session_state.chat_session = chat_session
            
            return True
        except Exception as e:
            st.sidebar.error(f"Failed to update parameters: {str(e)}")
            return False
    return False

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

def send_chat_message():
    if st.session_state.user_input and st.session_state.api_key_configured:
        user_message = st.session_state.user_input
        
        # Add user message to chat history
        st.session_state.chat_messages.append({"role": "user", "content": user_message})
        
        # Clear input box
        st.session_state.user_input = ""
        
        with st.spinner("Generating response..."):
            try:
                # Send message to the model
                response = st.session_state.chat_session.send_message(user_message)
                
                # Add assistant response to chat history
                st.session_state.chat_messages.append({"role": "assistant", "content": response.text})
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

# Sidebar configuration
with st.sidebar:
    st.title("⚙️ Configuration")
    
    if not st.session_state.api_key_configured:
        # Add model parameters configuration
        st.markdown("### Model Parameters")
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
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
        
        api_key = st.text_input("Enter your Google API Key", type="password")
        
        selected_model_name = st.selectbox(
            "Select Gemini Model",
            options=list(AVAILABLE_MODELS.keys()),
            index=0
        )
        
        if st.button("Configure API"):
            if api_key:
                try:
                    selected_model = AVAILABLE_MODELS[selected_model_name]
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
        
        # Dynamic parameters section (always visible after configuration)
        st.markdown("### Model Parameters")
        st.caption("Adjust parameters on-the-fly")
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=st.session_state.generation_config["temperature"],
            step=0.1,
            key="dyn_temperature"
        )
        
        top_p = st.slider(
            "Top P",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.generation_config["top_p"],
            step=0.05,
            key="dyn_top_p"
        )
        
        top_k = st.slider(
            "Top K",
            min_value=1,
            max_value=100,
            value=st.session_state.generation_config["top_k"],
            key="dyn_top_k"
        )
        
        max_tokens = st.slider(
            "Max Output Tokens",
            min_value=1000,
            max_value=16000,
            value=st.session_state.generation_config["max_output_tokens"],
            step=1000,
            key="dyn_max_tokens"
        )
        
        # Update button for dynamic parameters
        if st.button("Update Parameters"):
            # Update generation config
            st.session_state.generation_config = {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_output_tokens": max_tokens,
            }
            
            if update_model_parameters():
                st.success("Parameters updated successfully!")
        
        base_url = st.query_params.get('base_url', "ai-retina2advanced.streamlit.app")
        shareable_link = f"{base_url}?api_key={quote(st.session_state.api_key)}&model={st.session_state.selected_model}"
        st.markdown("### Shareable Link")
        st.code(shareable_link, language="text")
        st.markdown("⚠️ **Note**: This link contains your API key. Share securely!")

# Main app
st.title("🔬 Retinal Image Analyzer v2.0")
st.markdown("Upload retinal images (OCT/fundus) for AI-powered analysis. ©Anand Singh Brar  [**@brar_retina**](https://www.instagram.com/brar_retina)")

# Main form
with st.form("analysis_form"):
    case_notes = st.text_area(
        "Case Scenario",
        height=100,
        placeholder="Enter any relevant clinical information..."
    )
    
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
                
                # Add to chat history
                st.session_state.chat_messages = [
                    {"role": "user", "content": "Please analyze these uploaded images."},
                    {"role": "assistant", "content": response.text}
                ]
                
                st.session_state.analysis_complete = True
                
                # Display results
                st.success("Analysis complete!")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Full error details:", exc_info=True)

# Display analysis results and chat interface
if st.session_state.analysis_complete:
    # Display uploaded images
    st.markdown("### Uploaded Images")
    image_cols = st.columns(min(len(uploaded_files), 4) if 'uploaded_files' in locals() and uploaded_files else 1)
    if 'uploaded_files' in locals() and uploaded_files:
        for idx, uploaded_file in enumerate(uploaded_files):
            with image_cols[idx % len(image_cols)]:
                st.image(uploaded_file, caption=f"Image {idx + 1}")
    
    # Display chat messages
    st.markdown("### Analysis & Follow-up")
    
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_messages:
            if message["role"] == "user":
                st.markdown(f"**You**: {message['content']}")
            else:
                st.markdown(f"**AI Specialist**: {message['content']}")
    
    # Chat input
    st.text_input(
        "Ask follow-up questions or request clarification",
        key="user_input",
        on_change=send_chat_message
    )

# Usage instructions
with st.sidebar:
    st.markdown("### How to use")
    st.markdown("""
    1. Enter your Google API key (or use shared link)
    2. Select Gemini model version
    3. Upload retinal images (OCT/fundus)
    4. Add any relevant clinical notes
    5. Click 'Analyze Images'
    6. Ask follow-up questions in the chat box
    7. Adjust model parameters as needed
    """)
    
    st.markdown("### Privacy Notice")
    st.markdown("""
    - Ensure no PHI (Protected Health Information) is uploaded
    - Data is not stored and is only used for analysis
    - Use appropriate de-identification methods
    - API keys in URLs should be shared securely
    """)