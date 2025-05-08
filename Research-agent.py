import os
import logging
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import PubMedAPIWrapper
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from scholarly import scholarly
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Retrieve API keys from environment or secrets
pubmed_api_key = st.secrets["PUBMED_API_KEY"]
groq_api_key = st.secrets["GROQ_API_KEY"]

# Ensure the PubMed API key is available
if not pubmed_api_key:
    raise ValueError("PubMed API key is missing. Please add it to the environment variables.")

# Initialize PubMed API Wrapper
pubmed_wrapper = PubMedAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=300,
    api_key=pubmed_api_key
)

# Define PubMed Tool function
def pubmed_tool_func(query: str):
    return PubmedQueryRun(api_wrapper=pubmed_wrapper).run(query)

# Wrap PubMedQueryRun as a LangChain Tool (must use Tool here, not just a function)
pubmed_tool = Tool(
    name="PubMedQuery",
    description="Search PubMed for medical and scientific literature.",
    func=pubmed_tool_func,
    is_single_input=True  # Make sure that the tool works with single input
)

# Define Google Scholar Query Function
def google_scholar_query(query, num_results=10):
    search_results = scholarly.search_pubs(query)
    results = []
    try:
        for _ in range(num_results):
            result = next(search_results)
            if "bib" in result:  # Safeguard against missing keys
                results.append(result["bib"])
    except StopIteration:
        pass
    return results

# Wrap Google Scholar query function as a LangChain Tool (must use Tool here, not just a function)
google_scholar_tool = Tool(
    name="GoogleScholarQuery",
    description="Search Google Scholar for academic articles.",
    func=google_scholar_query,
    is_single_input=True  # Make sure that the tool works with single input
)

# Define the tools list (ensure they are wrapped as Tool instances)
tools = [pubmed_tool, google_scholar_tool]

# Initialize LangChain LLM
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="gemma2-9b-it",
    streaming=True
)

# Streamlit app setup
st.title("Research Agent")
st.write("This agent helps you search PubMed and Google Scholar")

# Sliders for user customization
st.write("Customize your search:")
top_k_results = st.slider(
    label="Top Results:",
    min_value=1,
    max_value=10,
    value=5,
    step=1,
    help="Select the number of top search results to display."
)
doc_content_chars_max = st.slider(
    label="Max Characters:",
    min_value=100,
    max_value=500,
    value=250,
    step=100,
    help="Set the maximum number of characters per document."
)

# Sidebar for API key input
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Please Enter your Groq API key:", type="password")

# Initialize message history for the chat
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "Assistant",
            "content": "Hi, I am your research assistant. How can I help you?"
        }
    ]

# Display previous messages in a scrollable container
with st.container():
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

# Chat input and prompt handling
if prompt := st.chat_input("Search: "):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Update PubMed settings with user customization
    pubmed_wrapper.top_k_results = top_k_results
    pubmed_wrapper.doc_content_chars_max = doc_content_chars_max
    logging.info(f"Top K Results: {top_k_results}, Max Characters: {doc_content_chars_max}")

    # Initialize the agent with tools (ensure that tools are wrapped as Tool objects)
    search_agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True
    )  

    with st.chat_message("assistant"):
        # Set up Streamlit callback handler
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

        # Run the agent with the user messages
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])

        # Add assistant's response to message history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })

        # Display the response
        st.write(response)

