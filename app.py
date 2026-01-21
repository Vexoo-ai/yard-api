import streamlit as st
from document_processor import DocumentProcessor
from inference import InferenceAgent, LLMProvider

# Set page configuration
st.set_page_config(
    page_title="Document AI Assistant",
    page_icon="📚",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 4rem;
        white-space: pre-wrap;
        background-color: #FFFFFF;
        border-radius: 4px;
        color: #000000;
        padding: 1rem 2rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E88E5;
        color: #FFFFFF;
    }
    div[data-testid="stFileUploader"] {
        border: 2px dashed #1E88E5;
        border-radius: 8px;
        padding: 2rem;
        margin: 1rem 0;
    }
    .chat-container {
        border-radius: 8px;
        background-color: #F8F9FA;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        border-radius: 4px;
        height: 3rem;
        background-color: #1E88E5;
        color: white;
    }
    .upload-text {
        text-align: center;
        color: #666666;
        margin-bottom: 1rem;
    }
    .model-selector {
        padding: 1rem;
        background-color: #F8F9FA;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .provider-info {
        font-size: 0.8rem;
        color: #666666;
        margin-top: 0.5rem;
        font-style: italic;
    }
    .stSelectbox {
        margin-bottom: 0 !important;
    }
    .thinking-container {
        background-color: #F0F7FF;
        border: 1px solid #BFDEFF;
        border-radius: 4px;
        padding: 0.5rem;
        margin-bottom: 1rem;
        font-size: 0.9rem;
        color: #1E88E5;
    }
    .thinking-header {
        display: flex;
        align-items: center;
        margin-bottom: 0.5rem;
        font-weight: bold;
    }
    .thinking-content {
        padding: 0.5rem;
        background-color: white;
        border-radius: 4px;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.6; }
        100% { opacity: 1; }
    }
    .thinking-indicator {
        animation: pulse 1.5s infinite;
        margin-left: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session states
if 'doc_processor' not in st.session_state:
    st.session_state.doc_processor = DocumentProcessor()
if 'inference_agent' not in st.session_state:
    st.session_state.inference_agent = InferenceAgent()
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'llm_provider' not in st.session_state:
    st.session_state.llm_provider = LLMProvider.CLAUDE

def display_chat_messages():
    """Display chat messages in the Streamlit interface"""
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def handle_file_upload(uploaded_files):
    """Handle document file upload and processing"""
    if not uploaded_files:
        return
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("Process Documents 🚀"):
            with st.spinner("Processing your documents..."):
                try:
                    num_chunks = st.session_state.doc_processor.process_documents(uploaded_files)
                    st.success(f"✅ Successfully processed {num_chunks} document chunks!")
                except Exception as e:
                    st.error(f"❌ Error during processing: {str(e)}")

def display_model_selector():
    """Display the model selection interface"""
    st.markdown('<div class="model-selector">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("**🤖 Select Model:**")
    with col2:
        selected_provider = st.selectbox(
            "",
            options=[provider.value for provider in LLMProvider],
            index=0,
            key="model_selector"
        )
        st.session_state.llm_provider = LLMProvider(selected_provider)
        
        # Display model info based on selection
        if st.session_state.llm_provider == LLMProvider.CLAUDE:
            provider_info = "Using Claude 3.7 Sonnet with enhanced reasoning capabilities"
        else:  # Mistral
            provider_info = "Using Mistral Nemo for fast streaming responses"
            
        st.markdown(
            f'<p class="provider-info">{provider_info}</p>',
            unsafe_allow_html=True
        )
    st.markdown('</div>', unsafe_allow_html=True)

def handle_chat_interaction(prompt, context):
    """Handle chat interaction and response generation with streaming"""
    thinking_container = st.empty()
    response_container = st.chat_message("assistant")
    message_placeholder = response_container.empty()
    
    try:
        # Initialize containers for thinking and response
        thinking_content = ""
        final_response = ""
        is_thinking = True
        response_started = False
        
        # Get streaming response
        if st.session_state.llm_provider == LLMProvider.CLAUDE:
            # For Claude, we get parameters and create the stream in the app
            claude_params = st.session_state.inference_agent.generate_chat_response(
                st.session_state.chat_messages,
                context=context,
                provider=st.session_state.llm_provider
            )
            
            # Claude stream handling using context manager
            try:
                with claude_params["client"].messages.stream(
                    model=claude_params["model"],
                    max_tokens=claude_params["max_tokens"],
                    thinking=claude_params["thinking"],
                    messages=claude_params["messages"]
                ) as stream:
                    for event in stream:
                        if event.type == "content_block_start":
                            if event.content_block.type == "thinking":
                                is_thinking = True
                            elif event.content_block.type == "text":
                                is_thinking = False
                                response_started = True
                                
                        elif event.type == "content_block_delta":
                            if event.delta.type == "thinking_delta" and is_thinking:
                                # Add the thinking delta to the thinking content
                                thinking_content += event.delta.thinking
                                
                                # Clean up any potential HTML issues for display
                                display_thinking = thinking_content.replace('<', '&lt;').replace('>', '&gt;')
                                
                                # Display thinking content with proper formatting
                                thinking_container.markdown(f"""
                                    <div class="thinking-container">
                                        <div class="thinking-header">
                                            🤔 Thinking Process
                                            <span class="thinking-indicator">...</span>
                                        </div>
                                        <div class="thinking-content">
                                            {display_thinking}
                                        </div>
                                    </div>
                                """, unsafe_allow_html=True)
                                
                            elif event.delta.type == "text_delta" and (not is_thinking or response_started):
                                final_response += event.delta.text
                                message_placeholder.markdown(final_response + "▌")
                                
                        elif event.type == "content_block_stop":
                            if response_started:
                                message_placeholder.markdown(final_response)
            except Exception as e:
                st.error(f"Error processing Claude stream: {str(e)}")
                if not final_response:
                    message_placeholder.markdown("Sorry, there was an error generating a response from Claude.")
                    
        else:  # MISTRAL
            # For Mistral, we get parameters and create the stream in the app
            mistral_params = st.session_state.inference_agent.generate_chat_response(
                st.session_state.chat_messages,
                context=context,
                provider=st.session_state.llm_provider
            )
            
            # Mistral stream handling - Note: no thinking process for Mistral
            try:
                # Clear the thinking container since Mistral doesn't use it
                thinking_container.empty()
                
                # Mistral doesn't have thinking mode, so set it to false
                is_thinking = False
                response_started = True
                
                # Create the stream
                stream = mistral_params["client"].chat.stream(
                    messages=mistral_params["messages"],
                    model=mistral_params["model"],
                    max_tokens=mistral_params["max_tokens"],
                    temperature=mistral_params["temperature"]
                )
                
                # Process the stream
                for chunk in stream:
                    try:
                        # Extract content from the stream
                        if hasattr(chunk, 'data') and chunk.data:
                            completion_chunk = chunk.data
                            if hasattr(completion_chunk, 'choices') and completion_chunk.choices:
                                delta = completion_chunk.choices[0].delta
                                if hasattr(delta, 'content') and delta.content is not None:
                                    final_response += delta.content
                                    message_placeholder.markdown(final_response + "▌")
                    except Exception as e:
                        # Continue silently on error to avoid interrupting the stream
                        continue
                
                # Final update to response
                message_placeholder.markdown(final_response)
                
            except Exception as e:
                st.error(f"Error processing Mistral stream: {str(e)}")
                if not final_response:
                    message_placeholder.markdown("Sorry, there was an error generating a response from Mistral.")
        
        # Final update to response
        if final_response.strip():
            message_placeholder.markdown(final_response)
            # Add assistant response to chat history
            st.session_state.chat_messages.append({"role": "assistant", "content": final_response})
        
    except Exception as e:
        st.error(f"❌ Error generating response: {str(e)}")

def main():
    # App Header
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("""
            <h1 style='text-align: center; color: #1E88E5; margin-bottom: 2rem;'>
                📚 Vexoo Docs PlayGround
            </h1>
        """, unsafe_allow_html=True)
    
    # Create tabs with icons
    tab1, tab2 = st.tabs([
        "📤 Upload Documents",
        "💬 Chat"
    ])
    
    # Upload Documents Tab
    with tab1:
        st.markdown("""
            <h2 style='text-align: center; margin-bottom: 1rem;'>Upload Your Documents</h2>
        """, unsafe_allow_html=True)
        
        st.markdown('<p class="upload-text">Support for PDF, Excel, and PowerPoint files</p>', unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Drag and drop files here",
            type=['pdf', 'xlsx', 'ppt', 'pptx'],
            accept_multiple_files=True
        )
        
        handle_file_upload(uploaded_files)
    
    # Chat Tab
    with tab2:
        st.markdown("<h2 style='text-align: center; margin-bottom: 2rem;'>Chat with Your Documents</h2>", unsafe_allow_html=True)
        
        # Model selection
        display_model_selector()
        
        # Chat container
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Display chat messages
        display_chat_messages()
        
        # Chat input
        if prompt := st.chat_input("💭 Ask a question about your documents..."):
            # Add user message to chat history
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get relevant context from vector store
            context = None
            if st.session_state.doc_processor.get_vector_store() is not None:
                context = st.session_state.doc_processor.search_documents(prompt, k=3)
            
            # Generate streaming response
            handle_chat_interaction(prompt, context)
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
