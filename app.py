from pathlib import Path

import streamlit as st

from src.embeddings import generate_embeddings
from src.llm import generate_answer
from src.pdf_processor import UPLOAD_DIR, create_chunks, extract_text_from_pdf
from src.rag_pipeline import build_context
from src.retriever import embed_question, retrieve_relevant_chunks
from src.vector_store import create_faiss_index


st.set_page_config(
    page_title="RAG Based PDF Chat Bot",
    page_icon="📄",
)

st.title("📄 RAG Based PDF Chat Bot")
st.write("Upload a PDF and ask questions about its contents.")


uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"],
)

question = st.text_input(
    "Ask a question",
    placeholder="What would you like to know from the PDF?",
)

ask_button = st.button("Ask")


if ask_button:
    if uploaded_file is None:
        st.warning("Please upload a PDF first.")

    elif not question.strip():
        st.warning("Please enter a question.")

    else:
        try:
            with st.spinner("Processing your PDF and generating an answer..."):
                UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

                filename = Path(uploaded_file.name).name
                pdf_path = UPLOAD_DIR / filename
                pdf_path.write_bytes(uploaded_file.getvalue())

                text = extract_text_from_pdf(pdf_path)
                chunks = create_chunks(text)

                if not chunks:
                    st.error("No text could be extracted from the PDF.")
                else:
                    embeddings = generate_embeddings(chunks)
                    index = create_faiss_index(embeddings)

                    question_embedding = embed_question(question)

                    retrieved_chunks = retrieve_relevant_chunks(
                        index,
                        chunks,
                        question_embedding,
                        k=3,
                    )

                    context = build_context(retrieved_chunks)

                    answer = generate_answer(
                        context,
                        question,
                    )

                    st.session_state["answer"] = answer

        except Exception as exc:
            st.error("Unable to process the PDF and generate an answer.")
            st.exception(exc)


if "answer" in st.session_state:
    st.divider()
    st.subheader("Answer")
    st.write(st.session_state["answer"])