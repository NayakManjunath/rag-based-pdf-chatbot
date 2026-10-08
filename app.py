from pathlib import Path
import streamlit as st

from src.chat_history import (
    add_assistant_message,
    add_user_message,
    get_pdf_id,
    initialize_chat_history,
    reset_chat_history_for_new_pdf,
)

from src.embeddings import generate_embeddings
from src.llm import generate_answer
from src.pdf_processor import (
    UPLOAD_DIR,
    create_document_chunks,
    extract_pages_from_pdf,
)
from src.rag_pipeline import build_context
from src.retriever import embed_question, retrieve_relevant_chunks
from src.source_evidence import format_source_evidence
from src.vector_store import create_faiss_index


st.set_page_config(
    page_title="RAG Based PDF Chat Bot",
    page_icon="📄",
)


st.title("📄 RAG Based PDF Chat Bot")
st.write("Upload a PDF and ask questions about its contents.")


# ---------------------------------------------------------
# Initialize conversational state
# ---------------------------------------------------------

initialize_chat_history(st.session_state)

if "current_pdf_id" not in st.session_state:
    st.session_state["current_pdf_id"] = None


# ---------------------------------------------------------
# PDF upload
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"],
)


# ---------------------------------------------------------
# Detect a new PDF and start a new conversation
# ---------------------------------------------------------

if uploaded_file is not None:
    pdf_bytes = uploaded_file.getvalue()
    pdf_id = get_pdf_id(pdf_bytes)

    reset_chat_history_for_new_pdf(
        st.session_state,
        pdf_id,
    )

# ---------------------------------------------------------
# Question input
# ---------------------------------------------------------

question = st.text_input(
    "Ask a question",
    placeholder="Ask something about your PDF...",
)

ask_button = st.button("Ask")


# ---------------------------------------------------------
# Process question
# ---------------------------------------------------------

if ask_button:

    if uploaded_file is None:
        st.warning("Please upload a PDF first.")

    elif not question.strip():
        st.warning("Please enter a question.")

    else:
        # Store the user's question immediately.
        add_user_message(
            st.session_state,
            question,
        )

        try:
            with st.spinner("Reading the PDF and generating an answer..."):

                # -----------------------------------------
                # Save uploaded PDF
                # -----------------------------------------

                UPLOAD_DIR.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                filename = Path(uploaded_file.name).name
                pdf_path = UPLOAD_DIR / filename

                pdf_path.write_bytes(
                    uploaded_file.getvalue()
                )

                # -----------------------------------------
                # PDF → pages → chunks
                # -----------------------------------------

                pages = extract_pages_from_pdf(
                    pdf_path
                )

                chunks = create_document_chunks(
                    pages,
                    source_file=filename,
                )

                if not chunks:
                    raise ValueError(
                        "No extractable text was found in the PDF."
                    )

                # -----------------------------------------
                # Chunks → embeddings → FAISS
                # -----------------------------------------

                embeddings = generate_embeddings(
                    chunks
                )

                index = create_faiss_index(
                    embeddings
                )

                # -----------------------------------------
                # Question → embedding
                # -----------------------------------------

                question_embedding = embed_question(
                    question
                )

                # -----------------------------------------
                # Retrieve relevant chunks
                # -----------------------------------------

                retrieved_chunks = retrieve_relevant_chunks(
                    index,
                    chunks,
                    question_embedding,
                    k=3,
                )

                # -----------------------------------------
                # Source evidence
                # -----------------------------------------

                sources = format_source_evidence(
                    retrieved_chunks
                )

                # -----------------------------------------
                # Build grounded context
                # -----------------------------------------

                context = build_context(
                    retrieved_chunks
                )

                # -----------------------------------------
                # Generate grounded answer
                # -----------------------------------------

                answer = generate_answer(
                    context,
                    question,
                )

                # -----------------------------------------
                # Store assistant response + sources
                # -----------------------------------------

                add_assistant_message(
                    st.session_state,
                    answer,
                    sources,
                )

        except Exception as exc:
            st.error(
                f"Unable to process the question: {exc}"
            )


# ---------------------------------------------------------
# Display conversation history
# ---------------------------------------------------------

if st.session_state["chat_history"]:

    st.divider()
    st.subheader("💬 Conversation")

    for message in st.session_state["chat_history"]:

        role = message["role"]

        if role == "user":

            with st.chat_message("user"):
                st.write(message["content"])

        elif role == "assistant":

            with st.chat_message("assistant"):

                st.write(message["content"])

                sources = message.get(
                    "sources",
                    [],
                )

                if sources:

                    st.caption("📚 Sources")

                    source_file = sources[0][
                        "source_file"
                    ]

                    source_details = " · ".join(
                        f"Page {source['page_number']} "
                        f"({source['chunk_id']})"
                        for source in sources
                    )

                    st.caption(
                        f"📄 {source_file} — "
                        f"{source_details}"
                    )