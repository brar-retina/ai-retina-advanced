# Add this import at the top
from google.generativeai.types import Content

# Replace the try block in the submit button section with:
try:
    with st.spinner("Analyzing images..."):
        # Prepare message
        message = "Please analyze these retinal images."
        if case_notes:
            message += f"\n\nClinical Notes: {case_notes}"

        # Prepare images
        image_parts = []
        for uploaded_file in uploaded_files:
            try:
                image = Image.open(uploaded_file)
                st.write(f"Processing image: {image.size}, mode: {image.mode}")  # Debug info
                
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Convert image to bytes
                byte_stream = io.BytesIO()
                image.save(byte_stream, format='JPEG')
                image_bytes = byte_stream.getvalue()
                
                # Append in correct format
                image_parts.append({
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": image_bytes
                    }
                })
            except Exception as img_error:
                st.error(f"Error processing image: {str(img_error)}")
                continue

        # Prepare content parts
        content_parts = [
            {"text": message},
            *image_parts
        ]

        # Send message with correct format
        message_parts = [Content(parts=content_parts)]
        
        # Debug info
        st.write("Sending message with structure:", [type(p) for p in message_parts])
        
        # Get response using chat session
        response = st.session_state.chat_session.send_message(message_parts)
        
        # Process search results
        search_results = process_search_results(response)
        st.session_state.search_results = search_results
        
        # Display results
        st.success("Analysis complete!")
        st.markdown("### Analysis Results")
        st.write(response.text)
        
        # Display search results if available
        if search_results:
            st.markdown("### Related Research & References")
            for result in search_results:
                with st.expander(result['title']):
                    st.write(result['snippet'])
                    st.markdown(f"[Read more]({result['url']})")
        
        # Display uploaded images
        st.markdown("### Uploaded Images")
        cols = st.columns(len(uploaded_files))
        for idx, uploaded_file in enumerate(uploaded_files):
            with cols[idx]:
                st.image(uploaded_file, caption=f"Image {idx + 1}")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    # Add more detailed error information
    import traceback
    st.error(f"Full error trace: {traceback.format_exc()}")
