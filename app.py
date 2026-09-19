import os
import streamlit as st
import numpy as np
import faiss

from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="VFSTR R22 Academic Assistant",
    page_icon="🎓",
    layout="centered"
)


# ==========================================
# LOAD API KEY
# ==========================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found in .env file.")
    st.stop()


# ==========================================
# PROJECT INFORMATION
# ==========================================

st.title("🎓 VFSTR R22 Academic Regulations Assistant")

st.write(
    "Ask questions about the VFSTR B.Tech R22 Academic Regulations."
)

st.info(
    "This assistant answers questions using the VFSTR R22 regulations document."
)


# ==========================================
# LOAD AND PROCESS PDF
# ==========================================

@st.cache_resource
def load_rag_system():

    PDF_PATH = "documents/vfstr_r22_regulations.pdf"

    # ------------------------------
    # Load PDF
    # ------------------------------

    reader = PdfReader(PDF_PATH)

    documents = []

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text()

        if text:

            documents.append({
                "text": text,
                "page": page_number
            })


    # ------------------------------
    # Split text into chunks
    # ------------------------------

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = []
    metadata = []

    for document in documents:

        page_chunks = text_splitter.split_text(
            document["text"]
        )

        for chunk in page_chunks:

            chunks.append(chunk)

            metadata.append({
                "page": document["page"]
            })


    # ------------------------------
    # Create embeddings
    # ------------------------------

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    embeddings = embedding_model.encode(
        chunks,
        show_progress_bar=False
    )

    embeddings = np.array(
        embeddings
    ).astype("float32")


    # ------------------------------
    # Create FAISS index
    # ------------------------------

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(embeddings)


    # ------------------------------
    # Load Gemini
    # ------------------------------

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=GOOGLE_API_KEY
    )


    return (
        chunks,
        metadata,
        embedding_model,
        index,
        llm
    )


# ==========================================
# LOAD RAG SYSTEM
# ==========================================

with st.spinner("Loading VFSTR regulations..."):

    (
        chunks,
        metadata,
        embedding_model,
        index,
        llm
    ) = load_rag_system()


# ==========================================
# QUESTION INPUT
# ==========================================

query = st.text_input(
    "Ask your question:",
    placeholder="Example: What is the minimum attendance requirement?"
)


# ==========================================
# ASK BUTTON
# ==========================================

if st.button("🔍 Ask"):

    if not query.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        # ------------------------------
        # Create query embedding
        # ------------------------------

        query_embedding = embedding_model.encode(
            [query]
        )

        query_embedding = np.array(
            query_embedding
        ).astype("float32")


        # ------------------------------
        # Search FAISS
        # ------------------------------

        distances, indices = index.search(
            query_embedding,
            3
        )


        # ------------------------------
        # Retrieve relevant chunks
        # ------------------------------

        retrieved_chunks = []

        for index_number in indices[0]:

            retrieved_chunks.append(
                chunks[index_number]
            )


        context = "\n\n".join(
            retrieved_chunks
        )


        # ------------------------------
        # Gemini prompt
        # ------------------------------

        prompt = f"""
You are a university academic regulations assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer is not present in the context, say:

"I could not find this information in the VFSTR R22 regulations."

Do not invent information.

Context:
{context}

User Question:
{query}

Give a clear and concise answer.
"""


        # ------------------------------
        # Generate answer
        # ------------------------------

        with st.spinner("Finding the answer..."):

            response = llm.invoke(
                prompt
            )


        # ------------------------------
        # Extract response text
        # ------------------------------

        if isinstance(
            response.content,
            list
        ):

            answer = ""

            for item in response.content:

                if (
                    isinstance(item, dict)
                    and item.get("type") == "text"
                ):

                    answer += item.get(
                        "text",
                        ""
                    )

        else:

            answer = response.content


        # ------------------------------
        # Display answer
        # ------------------------------

        st.subheader("Answer")

        st.write(answer)


        # ------------------------------
        # Display sources
        # ------------------------------

        st.subheader("Sources")

        source_pages = sorted(
            set(
                metadata[index_number]["page"]
                for index_number in indices[0]
            )
        )

        st.write(
            "VFSTR R22 Academic Regulations — "
            + ", ".join(
                f"Page {page}"
                for page in source_pages
            )
        )


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.header("About")

    st.write(
        "This project uses Retrieval-Augmented Generation (RAG) "
        "to answer questions from the VFSTR R22 Academic Regulations."
    )

    st.write("**Technologies used:**")

    st.write(
        """
        • Python  
        • Streamlit  
        • PyPDF  
        • Sentence Transformers  
        • FAISS  
        • Google Gemini
        """
    )

    st.divider()

    st.write(
        "**Document:** VFSTR B.Tech R22 Academic Regulations"
    )

    st.write(
        "**Purpose:** Academic regulation question answering"
    )