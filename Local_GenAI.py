import tkinter as tk
from tkinter import scrolledtext
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from transformers import pipeline
import os
import PyPDF2
from bs4 import BeautifulSoup
import chardet

# Initialize RAG components (based on your RAG code)
# 1. Text Extraction Functions
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def extract_text_from_txt(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read()

def extract_text_from_html(html_path):
    # Read the file as binary to detect encoding
    with open(html_path, "rb") as f:
        raw_data = f.read()

    # Detect encoding
    result = chardet.detect(raw_data)
    encoding = result["encoding"]

    # Read the file with the detected encoding
    with open(html_path, "r", encoding=encoding, errors="replace") as f:
        soup = BeautifulSoup(f, "html.parser")
        return soup.get_text()

# 2. Load Data from Local Files
def load_local_data(directory):
    all_data = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(".pdf"):
                all_data.append(extract_text_from_pdf(file_path))
            elif file.endswith(".txt"):
                all_data.append(extract_text_from_txt(file_path))
            elif file.endswith(".htm"):
                all_data.append(extract_text_from_html(file_path))
    return all_data

# 3. Initialize Embedding Model and FAISS
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
local_data = load_local_data("C:\\DATA_PATH\\")
embeddings = np.array(embedding_model.encode(local_data, convert_to_tensor=False), dtype="float32")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# 4. Text Generation Model
generator = pipeline("text-generation", model="gpt2")

# 5. Retrieve Top-k Results and Generate Response
def retrieve_top_k(query, top_k=3):
    query_embedding = np.array(embedding_model.encode([query], convert_to_tensor=False), dtype="float32")
    distances, indices = index.search(query_embedding, top_k)
    results = [local_data[i] for i in indices[0]]
    return results

def generate_response(query):
    retrieved_docs = retrieve_top_k(query, top_k=2)
    context = " ".join(retrieved_docs)
    input_prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    response = generator(input_prompt, max_length=100, num_return_sequences=1)
    return response[0]["generated_text"]

# GUI Code with Tkinter
def ask_question():
    question = question_input.get("1.0", tk.END).strip()  # Get the user's question
    if not question:
        return
    answer = generate_response(question)  # Call the RAG pipeline
    output_area.config(state=tk.NORMAL)  # Enable editing for appending the response
    output_area.insert(tk.END, f"Q: {question}\nA: {answer}\n\n")
    output_area.config(state=tk.DISABLED)  # Disable editing
    question_input.delete("1.0", tk.END)  # Clear the input box

# Create the main application window
app = tk.Tk()
app.title("Local Generative AI - RAG")
app.geometry("600x400")

# Add components to the GUI
# 1. Label for instructions
label = tk.Label(app, text="Ask your question below:", font=("Arial", 12))
label.pack(pady=5)

# 2. Textbox for user input
question_input = tk.Text(app, height=3, width=70, font=("Arial", 10))
question_input.pack(pady=5)

# 3. Button to submit the question
ask_button = tk.Button(app, text="Ask", command=ask_question, font=("Arial", 10))
ask_button.pack(pady=5)

# 4. Scrollable text area for displaying answers
output_area = scrolledtext.ScrolledText(app, height=15, width=70, font=("Arial", 10))
output_area.pack(pady=5)
output_area.config(state=tk.DISABLED)  # Disable editing

# Run the application
app.mainloop()
