import streamlit as st
import os
from dotenv import load_dotenv
from langchain_helper_Google_gemini import get_sql_agent, query_database
# from langchain_helper_IBM_watson import get_sql_agent, query_database
import logging

load_dotenv()

st.set_page_config(
    page_title="AtliQ T-Shirts Database Q&A",
    page_icon="👕",
    layout="wide"
)

logging.getLogger().setLevel(logging.WARNING)

if 'sql_agent' not in st.session_state:
    st.session_state.sql_agent = None
if 'agent_error' not in st.session_state:
    st.session_state.agent_error = None

def initialize_agent():
    """Initialize the SQL agent with error handling"""
    try:
        with st.spinner("Initializing AI agent... This may take a few moments."):
            agent = get_sql_agent()
            st.session_state.sql_agent = agent
            st.session_state.agent_error = None
            st.success("✅ AI agent initialized successfully!")
            return True
    except Exception as e:
        st.session_state.agent_error = str(e)
        st.error(f"❌ Failed to initialize AI agent: {str(e)}")
        return False


# Main UI
st.title("🏪 AtliQ T-Shirts: Database Q&A 👕")
st.markdown("Ask questions about t-shirt inventory, sales, and more!")

# Sidebar for configuration and help
with st.sidebar:
    st.header("🔧 Configuration")
    
    # Agent status
    if st.session_state.sql_agent is None and st.session_state.agent_error is None:
        st.info("Agent not initialized. Enter a question to start.")
    elif st.session_state.agent_error:
        st.error("Agent initialization failed")
        if st.button("🔄 Retry Initialization"):
            st.session_state.agent_error = None
            st.rerun()
    else:
        st.success("Agent ready!")
    
    st.header("💡 Example Questions")
    example_questions = [
        "How many t-shirts are in stock?",
        "What are the different sizes available?", 
        "Show me all white t-shirts",
        "What's the total inventory value?",
        "Which brand has the most stock?",
        "How many Levi's t-shirts do we have?",
        "What sizes are available in Nike?",
        "Show me t-shirts under $50"
    ]
    
    for i, example in enumerate(example_questions):
        if st.button(f"📝 {example}", key=f"example_{i}"):
            st.session_state.current_question = example

# Main question input
question = st.text_input(
    "❓ **Ask your question:**", 
    value=st.session_state.get('current_question', ''),
    placeholder="e.g., How many Adidas t-shirts are in stock?",
    help="Ask anything about the t-shirt inventory, sales, brands, sizes, etc."
)

# Clear the session state question after using it
if hasattr(st.session_state, 'current_question'):
    del st.session_state.current_question

if question:
    # Initialize agent if not already done
    if st.session_state.sql_agent is None and st.session_state.agent_error is None:
        if not initialize_agent():
            st.stop()
    
    # If agent failed to initialize, show error and stop
    if st.session_state.agent_error:
        st.error("Cannot process questions until the agent is properly initialized.")
        st.info("Please check your configuration and try the 'Retry Initialization' button in the sidebar.")
        st.stop()
    
    # Process the question
    with st.spinner("🤔 Analyzing your question and querying the database..."):
        try:
            result = query_database(question, st.session_state.sql_agent)
            
            if result["success"]:
                st.header("✅ Answer")
                st.write(result["answer"])
            else:
                st.header("❌ Error")
                st.error(f"Failed to process your question: {result['error']}")
                
                st.header("💡 Troubleshooting Tips")
                st.markdown("""
                - Make sure your question is about the t-shirt database
                - Try rephrasing your question more specifically  
                - Check if you're asking about existing columns/tables
                - Ensure your database connection is working
                """)
                
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            st.info("Please try again or rephrase your question.")

# Footer with additional information
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #66A; font-size: 0.8em;'>
    Secure database queries powered by AI | 
    Using Google Gemini 1.5 Flash | 
    Connected to AtliQ T-Shirts Database
</div>
""", unsafe_allow_html=True)
