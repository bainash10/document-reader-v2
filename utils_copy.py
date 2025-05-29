import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pdfminer")

import pdfplumber
import re
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain
from transformers import pipeline
import torch

# Load PDF text
def load_pdf(file_path):
    try:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        return f"Error loading PDF: {e}"

# Create FAISS vectorstore
def create_vector_store(text):
    splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    chunks = splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    return vectorstore

# Load QA model chain
def get_qa_chain():
    device = 0 if torch.cuda.is_available() else -1
    model_names = ["google/flan-t5-large", "google/flan-t5-base", "google/flan-t5-small"]

    for model_name in model_names:
        try:
            print(f"Loading model: {model_name}")
            qa_pipeline = pipeline(
                "text2text-generation",
                model=model_name,
                tokenizer=model_name,
                device=device,
                max_length=512,
                do_sample=False,
                temperature=0
            )
            llm = HuggingFacePipeline(pipeline=qa_pipeline)
            chain = load_qa_chain(llm, chain_type="map_reduce")
            print(f"Model loaded: {model_name}")
            return chain
        except RuntimeError as e:
            print(f"Error loading {model_name}: {e}")
    return None

# Utility: gibberish detection
def is_gibberish(text, threshold=0.5):
    if not text:
        return True
    text_no_spaces = text.replace(" ", "")
    non_alpha = len(re.findall(r'[^a-zA-Z]', text_no_spaces))
    return (non_alpha / len(text_no_spaces)) > threshold

# Utility: question validation
def is_valid_question(text):
    if not text or len(text.strip()) < 5:
        return False
    words = re.findall(r'\b[a-zA-Z]{2,}\b', text)
    return len(words) >= 2 and not is_gibberish(text)

# Utility: answer refusal
def is_refusal_answer(answer):
    refusal_phrases = [
        "i don't know", "not sure", "no information", "no relevant",
        "out-of-scope", "unclear", "cannot answer", "don't have",
        "sorry", "cannot provide", "unknown", "i am unable"
    ]
    answer_lower = answer.lower().strip()
    if len(answer) < 10:
        return True
    if any(phrase in answer_lower for phrase in refusal_phrases):
        return True
    return is_gibberish(answer)

# Utility: clean model answer
def clean_answer(answer):
    answer = answer.strip()
    answer = re.sub(r'^(Answer:|A:)', '', answer, flags=re.IGNORECASE).strip()
    answer = re.sub(r'\n{2,}', '\n', answer)  # Collapse extra newlines
    return answer

# Main function to answer query
def answer_question(query, vectorstore, chain, top_k=6):
    if not is_valid_question(query):
        return "⚠️ Please ask a more detailed and meaningful question."

    if chain is None:
        return "❌ The QA model could not be loaded due to memory or system constraints."

    try:
        docs = vectorstore.similarity_search(query, k=top_k)
        if not docs:
            return "⚠️ Sorry, no matching information was found in the document."

        prompt_query = f"Answer this as clearly and logically as possible using the document: {query}. Provide reasoning or explanation if applicable."

        result = chain.invoke({
            "input_documents": docs,
            "question": prompt_query
        })

        answer = clean_answer(result.get("output_text", ""))

        if is_refusal_answer(answer):
            return "⚠️ The question might be unclear or not answered by the document. Please rephrase and try again."

        return answer

    except RuntimeError as e:
        return f"❌ Runtime Error: {str(e)}"
    except Exception as e:
        return f"❌ Unexpected Error: {str(e)}"

# Example usage
"""
file_path = "sample.pdf"
text = load_pdf(file_path)
if "Error" in text:
    print(text)
else:
    vectorstore = create_vector_store(text)
    qa_chain = get_qa_chain()
    query = "Describe the main theme discussed in the document, even if it is implied."
    print(answer_question(query, vectorstore, qa_chain))
"""
