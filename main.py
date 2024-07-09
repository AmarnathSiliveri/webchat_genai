import os
import streamlit as st
import pickle
from dotenv import load_dotenv


# Importing components from langchain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain

# Load environment variables from .env file
load_dotenv()
st.set_page_config(page_title="GenBot", page_icon='🚀')
# Define CSS styling
page_bg = """
<style>
[data-testid='stAppViewContainer'] {
    background-image: url("https://preview.redd.it/mountain-roads-3840x2160-v0-7rurxa9fxufb1.jpg?auto=webp&s=ed2dc535075f653670a91b3d576a064a5add9a3b");
    background-size: cover;
}
[data-testid="stHeader"] {
background-color: rgba(0,0,0,0);
}
</style>
"""

# Apply the styling using st.markdown
st.markdown(page_bg, unsafe_allow_html=True)

sidebar_bg = """
<style>
[data-testid="stSidebar"][aria-expanded="true"] {
    background-image: url("https://wallpapercave.com/wp/wp5746590.jpg");
    background-size: cover;
    background-repeat: no-repeat;
    background-position: center;
}
</style>
"""

# Apply the background image style to the sidebar using st.markdown
st.markdown(sidebar_bg, unsafe_allow_html=True)


# Set Streamlit app titles

st.title("GenBot: MULTI-WEB-chat 📈")
st.sidebar.title("WEB pages URLS")

if 'responses' not in st.session_state:
    st.session_state['responses'] = []

if 'querys' not in st.session_state:
    st.session_state['querys'] = []


# Input fields for URLs in the sidebar
n=st.sidebar.number_input("Enter number of URLS ", value=1, step=1,min_value=1)
urlss = [st.sidebar.text_input(f"URL 🔗{i+1}") for i in range(n)]
process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_googlegenai.pkl"



# Initialize Google Generative AI Chat model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    google_api_key=os.getenv('GOOGLE_API_KEY'),  # Fetch Google API key from environment variables
    temperature=0,  # Control the randomness of responses (0 = deterministic)
    max_tokens=None,  # Maximum number of tokens the model can generate in response
    timeout=None,  # Timeout duration for API calls
    max_retries=2,  # Maximum number of retries in case of API failures
)

# Process URLs button click event
if process_url_clicked:
    with st.spinner("Loading data..."):
        try:
            # Load data from provided URLs
            loader = UnstructuredURLLoader(urls=urlss)
            data = loader.load()

            # Split data into documents
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=3000
            )
            docs = text_splitter.split_documents(data)

            # Create embeddings using Google Generative AI embeddings
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vectorstore_genai = FAISS.from_documents(docs, embeddings)

            # Update status and notify completion
            st.success("Data loading and processing completed successfully!")
            
        except Exception as e:
            st.error(f"Error: {str(e)}")




# Query input field for user interaction
query = st.text_input("CHAT WITH WEB-PAGES")
st.session_state["querys"].append(("user",query))

if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            
            # Initialize the QA chain with language model and FAISS retriever
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            
            # Query the QA chain with user's question
            result = chain({"question": query}, return_only_outputs=True)
            
            # Display answer if available
            st.header("Answer")
            st.write(result["answer"])
            st.session_state["responses"].append(("BOT",result))
            
            # Display sources, if provided
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")
                for source in sources_list:
                    st.write(source)
                    
