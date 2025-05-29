from flask import Flask, request, render_template
from utils import load_pdf, create_vector_store, get_qa_chain, answer_question

app = Flask(__name__)

# Load PDF and initialize on app start
pdf_path = "uploads/project.pdf"
text = load_pdf(pdf_path)
vectorstore = create_vector_store(text)
qa_chain = get_qa_chain()

if qa_chain is None:
    raise RuntimeError("Failed to load QA model. Check memory or model configuration.")

@app.route('/', methods=['GET', 'POST'])
def ask_question():
    answer = None
    if request.method == 'POST':
        question = request.form.get('question', '').strip()
        if question:
            answer = answer_question(question, vectorstore, qa_chain)
        else:
            answer = "Please enter a valid question."
    return render_template('ask_queries.html', answer=answer)

if __name__ == "__main__":
    app.run(debug=True)
