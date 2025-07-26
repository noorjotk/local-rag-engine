import streamlit as st
import requests
import uuid  # ✅ For unique uploader key
import json
import time

API_URL = "http://localhost:8000"

st.set_page_config(page_title="📚 Advanced RAG Assistant", layout="wide")

#initializing session state keys
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = [] #The first time you upload a file, this key doesn’t exist yet, so you need to initialize it
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False
if "show_confirm" not in st.session_state:
    st.session_state.show_confirm = False
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = str(uuid.uuid4())  # ✅ Force reset
if "pending_question" not in st.session_state:
    st.session_state.pending_question = ""    
if "user_question" not in st.session_state:
    st.session_state.user_question = ""    
if "streaming_answer" not in st.session_state:
    st.session_state.streaming_answer = ""
if "stream_complete" not in st.session_state:
    st.session_state.stream_complete = False

st.title("📚 Your AI File Assistant : Upload, Ask, Learn")
st.write(
    "Analyze PDFs Securely — Your Data Never Leaves Your Machine"
)

# === Upload files ===
st.header("1️⃣ Upload your PDFs")
MAX_FILES = 3
uploaded_files = st.file_uploader(
    f"Upload up to {MAX_FILES} PDF files at once",
    type="pdf",
    accept_multiple_files=True,
    key=st.session_state.uploader_key,  # ✅ Use dynamic key to force clear
    disabled=st.session_state.is_processing
)
success_count = 0
failed_files = [] 
if uploaded_files:
    if len(uploaded_files) > MAX_FILES:
        st.warning(f"⚠️ You can only upload up to {MAX_FILES} files at once. Only the first {MAX_FILES} will be processed.")
        uploaded_files = uploaded_files[:MAX_FILES]  # ✅ Only keep first MAX_FILES

    new_files = [f for f in uploaded_files if f.name not in st.session_state.uploaded_files]    
    for f in new_files:
        with st.spinner(f"Processing {f.name}..."):
            files = {"file": (f.name, f, "application/pdf")}
            try:
                res = requests.post(f"{API_URL}/upload", files=files)
                if res.status_code == 200:
                    st.success(res.json().get("status", "✅ Uploaded!"))
                    st.session_state.uploaded_files.append(f.name)
                else:
                    failed_files.append(f.name) 
            except Exception as e:
                st.error(f"API request failed: {e}")
                failed_files.append(f.name) 
    if success_count > 0:
        st.info("✅ New files uploaded and ingested! You can now ask questions.")
    if failed_files:
        st.warning(f"⚠️ These files failed to upload: {', '.join(failed_files)}")
# === List all files ===
st.header("2️⃣ Choose PDF(s) to search")

try:
    files_res = requests.get(f"{API_URL}/files")
    if files_res.status_code == 200:
        file_list = files_res.json().get("files", [])
    else:
        st.error(f"Could not fetch file list: {files_res.status_code}")
        file_list = []
except Exception as e:
    st.warning(f"API not reachable: {e}")
    file_list = []

if file_list:
    select_all = st.checkbox("Select all files")

    if select_all:
        default_files = file_list
        select_files_box_disabled = True
    else:
        default_files = []
        select_files_box_disabled = False

    selected_files = st.multiselect(
        "Pick one or more files to ask questions about",
        options=file_list,
        default=default_files,
        disabled=select_files_box_disabled or st.session_state.is_processing
    )
else:
    st.warning("No files uploaded yet.")
    selected_files = []

# === Ask a question ===
st.header("3️⃣ Ask a question")
st.markdown(
    """
    <div style="background-color: #E3F2FD; padding: 15px; border-radius: 8px; border-left: 4px solid #2196F3; margin: 10px 0;">
        ⚠️ <strong>Note:</strong> Please ask meaningful questions related to the uploaded file(s).<br><br>
        ❌ Avoid random messages like <code>hello</code>, <code>how are you</code>, or generic questions — the assistant may not work correctly.<br><br>
    </div>
    """, 
    unsafe_allow_html=True
)

question = st.text_input("Your question", placeholder="Type your question here...",key="user_question", disabled=st.session_state.is_processing)#connect to session state!
# Button: disables when request in progress
get_answer = st.button(
    "Get Answer",
    disabled=st.session_state.is_processing
)

if get_answer:
    if not question or not question.strip():
        st.error("❌ Please enter a question before clicking 'Get Answer'.")
    # Validation: Check if no files are selected
    elif not selected_files:
        st.error("❌ Please select at least one PDF file to search before asking a question.")
    # All validations passed - proceed with the query
    else:
        # Clear previous answer when starting new query
        st.session_state.streaming_answer = ""
        st.session_state.final_answer = ""
        st.session_state.is_processing = True
        st.session_state.pending_question = question
        st.session_state.pending_files = selected_files
        st.session_state.stream_complete = False
        st.rerun()

# Show answer section if we have an answer (either streaming or completed)
if st.session_state.is_processing or st.session_state.get('final_answer', ''):
    st.subheader("Answer")
    
    # Create placeholder for content
    answer_placeholder = st.empty()
    
    if st.session_state.is_processing and not st.session_state.stream_complete:
        # Currently streaming
        payload = {
            "question": st.session_state.pending_question,
            "filenames": st.session_state.pending_files
        }
        try:
            with st.spinner("🔄 Generating answer..."):
                with requests.post(
                    f"{API_URL}/query_stream",
                    json=payload,
                    stream=True,
                    headers={'Accept': 'text/plain'}
                ) as response:
                    if response.status_code == 200:
                        for line in response.iter_lines(chunk_size=1, decode_unicode=True):
                            if line.startswith("data: "):
                                try:
                                    data = json.loads(line[6:])  # Remove "data: " prefix
                                    if "chunk" in data:
                                        st.session_state.streaming_answer += data["chunk"]
                                        answer_placeholder.markdown(st.session_state.streaming_answer)
                                        time.sleep(0.01)  # Small delay for better visual effect
                                    elif "done" in data and data["done"]:
                                        # Save final answer and reset processing state
                                        st.session_state.final_answer = st.session_state.streaming_answer
                                        st.session_state.stream_complete = True
                                        st.session_state.is_processing = False
                                        st.rerun()  # Rerun to update UI
                                        break
                                    elif "error" in data:
                                        st.session_state.final_answer = data["error"]
                                        st.session_state.stream_complete = True
                                        st.session_state.is_processing = False
                                        answer_placeholder.error(data["error"])
                                        break
                                except json.JSONDecodeError:
                                    continue
                    else:
                        error_msg = f"Query failed: {response.status_code}"
                        st.session_state.final_answer = error_msg
                        st.session_state.stream_complete = True
                        st.session_state.is_processing = False
                        answer_placeholder.error(error_msg)
        except Exception as e:
            error_msg = f"API request failed: {e}"
            st.session_state.final_answer = error_msg
            st.session_state.stream_complete = True
            st.session_state.is_processing = False
            answer_placeholder.error(error_msg)
    else:
        # Show completed answer (persists across reruns)
        if st.session_state.get('final_answer', ''):
            if st.session_state.final_answer.startswith(("Query failed:", "API request failed:")):
                answer_placeholder.error(st.session_state.final_answer)
            else:
                answer_placeholder.markdown(st.session_state.final_answer)



# === Delete selected file embeddings from vector db ===
st.header("4️⃣ Remove selected files from AI memory")

if selected_files:
    if st.button("🗑️ Delete selected file(s)"):
        st.session_state.show_confirm = True
    if st.session_state.show_confirm:
        confirm = st.radio(
            "Are you sure?",
            ("No", "Yes, delete them")
        )
        if confirm == "Yes, delete them":
            try:
                res = requests.post(f"{API_URL}/delete_file", json={"filenames": selected_files})
                if res.status_code == 200:
                    st.success(res.json().get("status", "Deleted!"))
                    st.session_state.uploaded_files = [
                        f for f in st.session_state.uploaded_files if f not in selected_files
                    ]
                    st.session_state.show_confirm = False
                    st.session_state.uploader_key = str(uuid.uuid4())  # ✅ Reset uploader!
                    st.rerun() #Force a rerun to refresh file list
                else:
                    st.error(f"Delete failed: {res.status_code}\n{res.text}")
                    st.session_state.show_confirm = False    
            except Exception as e:
                st.error(f"API request failed: {e}")
