from flask import Flask, request, render_template, redirect, url_for
import pickle
import fitz  # PyMuPDF for PDF handling
import os
import re

# Initialize Flask app
app = Flask(__name__)

# Load pickled models
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
model = pickle.load(open('clf.pkl', 'rb'))
label_encoder = pickle.load(open('encoder.pkl', 'rb'))

# Text cleaning function
def clean_resume(txt):
    txt = re.sub(r'http\S+\s', ' ', txt)
    txt = re.sub(r'RT|cc', ' ', txt)
    txt = re.sub(r'#\S+\s', ' ', txt)
    txt = re.sub(r'@\S+', ' ', txt)
    txt = re.sub(r'[%s]' % re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', txt)
    txt = re.sub(r'[^\x00-\x7f]', ' ', txt)
    txt = re.sub(r'\s+', ' ', txt)
    return txt.strip()

# Function to predict the category
def predict_category(text):
    cleaned_text = clean_resume(text)
    vectorized_text = tfidf.transform([cleaned_text]).toarray()
    prediction = model.predict(vectorized_text)
    return label_encoder.inverse_transform(prediction)[0]

# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text

# Home route
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        file = request.files["resume"]
        if file and (file.filename.endswith(".txt") or file.filename.endswith(".pdf")):
            file_path = os.path.join("uploads", file.filename)
            file.save(file_path)

            # Read the file content
            if file.filename.endswith(".pdf"):
                resume_text = extract_text_from_pdf(file_path)
            else:
                resume_text = file.read().decode("utf-8")

            # Predict category
            predicted_category = predict_category(resume_text)

            # Remove the file after processing
            os.remove(file_path)

            return render_template("index.html", prediction=predicted_category)

    return render_template("index.html", prediction=None)

# Run the app
if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)
