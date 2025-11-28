import streamlit as st
from YoutubeData import Get_Youtube_Data
from RagSystem import Use_Rag_System, Setup_Vector_Store

st.title("YouTube RAG Assistant 🎥🤖")

# --- Minimal session state just for vector_store ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# ------------- MODE 1: Ask for URL (no vector store yet) -------------
if st.session_state.vector_store is None:
    st.subheader("Step 1: Enter YouTube URL")

    youtube_url = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=W6NZfCO5SIk",
        key="youtube_url",
    )

    if st.button("Submit URL"):
        if youtube_url.strip():
            try:
                with st.spinner("Extracting transcript..."):
                    transcript = Get_Youtube_Data(youtube_url)

                with st.spinner("Building vector database..."):
                    st.session_state.vector_store = Setup_Vector_Store(transcript)

                st.success("Transcript stored in vector database. You can now ask questions about the video.")

                # Force rerun so that URL UI disappears and Q&A UI appears
                st.rerun()

            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.error("Please enter a valid YouTube link.")

# ------------- MODE 2: Q&A (vector store is ready) -------------
else:
    st.subheader("Ask a question about the video")

    question = st.text_input(
        "Your question",
        placeholder="Can you summarize the video?",
        key="question_input",
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        ask_clicked = st.button("Ask")
    with col2:
        reset_clicked = st.button("Change Video")

    if ask_clicked:
        if question.strip():
            try:
                with st.spinner("Generating answer..."):
                    answer = Use_Rag_System(st.session_state.vector_store, question.strip())
                st.markdown("### Answer:")
                st.write(answer)
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.error("Please enter a valid question.")

    if reset_clicked:
        # Clear vector store and go back to URL input mode
        st.session_state.vector_store = None
        st.rerun()
