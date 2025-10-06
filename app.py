from flask import Flask, request, jsonify, render_template, send_file, session, redirect, url_for, flash
import requests
from io import BytesIO
from fpdf import FPDF
from transformers import pipeline
import html
import json
import PyPDF2
from werkzeug.utils import secure_filename
import os
import pandas as pd
import datetime
from googletrans import Translator
import time
import random
import google.generativeai as genai
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from jinja2 import Environment, FileSystemLoader
from flask_babel import Babel, _, get_locale 


app = Flask(__name__)
app.secret_key = 'manjula143#'
app.config['LANGUAGES'] = {
    'en': 'English',
    'hi': 'Hindi',
    'te': 'Telugu'
}
babel = Babel(app, locale_selector=lambda: session.get('language', 'en'))


# Your API keys
KANOON_API_KEY = "4dbc8b0b8ed8e19162e08bbe008897a6c58c1b5f"
GROQ_API_KEY = "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"
GEMINI_API_KEY = "AIzaSyCgLBTviDMc5aXmxe2KYQT5Od2lLXTIocY" 
genai.configure(api_key=GEMINI_API_KEY)
translator = Translator()
@app.route('/set_language/<lang_code>')
def set_language(lang_code):
    session['language'] = lang_code
    print(f"DEBUG: Language set in session: {session.get('language')}") # ADD THIS LINE
    return redirect(request.referrer or url_for('home'))

# Indian Kanoon API Base URL
KANOON_API_URL = "https://api.indiankanoon.org/search/"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Function for searching data from Indian Kanoon
# Find and replace your old search_kanoon function
# In app.py, modify this function
# In app.py, replace your entire search_kanoon function with this one

# In app.py, replace your search_kanoon function with this final version

# In app.py, replace your search_kanoon function with this TEST version

# Restore the function to this final, clean version
# --- REPLACE the old CSV loading block with this NEW JSON loading block ---
model_path = "models/trained_model"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def generate_text(prompt):
    # Encode prompt
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    # Generate output tokens
    outputs = model.generate(
        inputs,
        max_length=inputs.shape[1] + 200,
        temperature=0.8,
        do_sample=True,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id,
    )
    # Decode generated tokens
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove prompt from output
    return generated_text[len(prompt):].strip()
env = Environment(loader=FileSystemLoader('templates'))

def render_template_doc(doc_type, context):
    template_map = {
        "Partnership Agreement": "partnership_agreement_template.txt",
        "Rental Agreement": "rental_agreement_template.txt",
        "Employment Contract": "employment_contract_template.txt",
        "Non-Disclosure Agreement": "nda_template.txt",
        "Service Agreement": "service_agreement_template.txt",
        "Will Testament": "will_testament_template.txt"
    }
    template_file = template_map.get(doc_type)
    if not template_file:
        raise ValueError(f"No template found for document type: {doc_type}")
    template = env.get_template(template_file)
    return template.render(context)

# --- REPLACE your old document_drafting_page route with this ---
# ---------------------
def search_kanoon(query):
    headers = {"Authorization": f"Token {KANOON_API_KEY}"}
    payload = {"query": query, "num": 10}
    try:
        response = requests.post(KANOON_API_URL, headers=headers, data=payload)
        if response.status_code == 200:
            try:
                results = response.json()
                if "errmsg" in results: return []
                return results.get("results", [])
            except ValueError: return []
        return []
    except requests.RequestException: return []
# Function for generating explanation using Groq
def get_groq_explanation(user_question, kanoon_data, max_retries=2):
    """
    Using Google Gemini Pro with improved error handling
    """
    
    for attempt in range(max_retries):
        try:
            print(f"🔄 Gemini attempt {attempt + 1}/{max_retries}")
            
            # Create Gemini Pro model
            model = genai.GenerativeModel('gemini-flash-latest')  # ✅ NEW NAME

            # Create a safer, more specific prompt
            prompt = f"""You are a helpful legal information assistant. Provide educational information about legal concepts in simple terms.

Question: {user_question}
Context: {kanoon_data}

Please provide an informative and educational response about this legal topic."""

            # Generate response with relaxed safety settings
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    candidate_count=1,
                    max_output_tokens=1500,  # Reduced tokens
                    temperature=0.3,  # Lower temperature for more consistent responses
                ),
                safety_settings={
                    genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                }
            )
            
            # Check if response was blocked
            if response.prompt_feedback:
                if response.prompt_feedback.block_reason:
                    print(f"❌ Prompt blocked: {response.prompt_feedback.block_reason}")
                    continue
            
            if response and response.text:
                print(f"✅ Gemini success on attempt {attempt + 1}")
                return response.text.strip()
            elif response.candidates and response.candidates[0].finish_reason:
                print(f"❌ Response blocked: {response.candidates[0].finish_reason}")
                # Try simpler prompt
                if attempt == 0:  # Only on first attempt
                    simple_response = model.generate_content(f"Explain this legal concept simply: {user_question}")
                    if simple_response and simple_response.text:
                        return simple_response.text.strip()
            else:
                print(f"❌ Empty Gemini response on attempt {attempt + 1}")
                
        except Exception as e:
            error_msg = str(e)
            print(f"💥 Gemini error on attempt {attempt + 1}: {error_msg}")
            
            # Check for specific error types
            if "quota" in error_msg.lower():
                return "The AI service has reached its usage limit for now. Please try again in a few minutes."
            elif "safety" in error_msg.lower():
                return "I cannot provide information on this topic due to safety restrictions. Please try rephrasing your question."
            elif "authentication" in error_msg.lower():
                return "There is an issue with API authentication. Please contact support."
                
            if attempt < max_retries - 1:
                time.sleep(2)  # Longer wait
    
    print(f"❌ All Gemini attempts failed")
    return """I'm having trouble connecting to the AI service right now. This could be due to:

1. **High demand** - Try again in 1-2 minutes
2. **Content restrictions** - Try rephrasing your question
3. **Network issues** - Check your internet connection

Please try again shortly!"""

def list_gemini_models():
    print("\n--- Listing available Gemini models ---")
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"  Model Name: {m.name}")
        print("--- End of model list ---")
        return True
    except Exception as e:
        print(f"Error listing models: {e}")
        print("--- End of model list (with error) ---")
        return False

# Call this function once when the app starts
list_gemini_models()

def translate_with_gemini(text, target_language):
    """
    Use Gemini to translate text instead of Google Translate
    """
    try:
        model = genai.GenerativeModel('gemini-flash-latest')
        
        language_names = {
            'telugu': 'Telugu',
            'hindi': 'Hindi'
        }
        
        target_lang_name = language_names.get(target_language, target_language)
        
        prompt = f"""Translate the following English text to {target_lang_name}. Maintain the formatting, structure, and meaning. Only provide the translation, no additional text:

{text}"""

        response = model.generate_content(prompt)
        
        if response and response.text:
            return response.text.strip()
        else:
            return None
            
    except Exception as e:
        print(f"Gemini translation error: {e}")
        return None



# Sample legal dictionary (this can be expanded)
legal_dict = {
    "tort": "A tort is a civil wrong that causes harm or loss to someone, and for which the law provides a remedy.",
    "contract": "A contract is an agreement between two or more parties that is enforceable by law.",
    "statute": "A statute is a written law passed by a legislative body.",
    "plaintiff": "A plaintiff is a person who brings a case against another in a court of law.",
    "defendant": "A defendant is a person who is being accused or sued in a court of law."
}


def render_partnership_agreement(context):
    env = Environment(loader=FileSystemLoader('templates'))
    template = env.get_template('partnership_agreement_template.txt')
    return template.render(context)
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        if email == "admin@example.com" and password == "password123":
            session['user'] = email
            return redirect(url_for('home'))
        else:
            flash("Invalid email or password. Please try again.")
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash("You have been logged out.")
    return redirect(url_for('login'))

# --- PROTECTED PAGE ROUTES ---

@app.route("/home")
def home():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template("index.html")
@app.route('/knowledge_hub')
def knowledge_hub():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('knowledge_hub.html')
@app.route('/legal_dictionary_search', methods=['POST'])
def legal_dictionary_search():
    # ✅ 3. SECURED API
    if 'user' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    term = request.json.get('term')
    if not term:
        return jsonify({"response": "Please enter a legal term."}), 400
    definition = legal_dict.get(term.lower())
    if definition:
        return jsonify({"response": definition})
    else:
        explanation = get_groq_explanation(term, "")
        return jsonify({"response": explanation})


# Route for the home page


# Route for the chatbot page
@app.route("/chatbot")
def chatbot():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    # --- IMPORTANT CHANGE HERE ---
    current_flask_locale = str(get_locale()) # Get the current locale as a string (e.g., 'en', 'hi', 'te')
    print(f"DEBUG: Flask rendering chatbot.html with locale: {current_flask_locale}")
    return render_template("chatbot.html", current_lang=current_flask_locale) # <--- MODIFIED THIS LINE

# Route to render the Document Drafting Page
# In app.py
# Delete the old /document_drafting route and add this new one.

@app.route('/draft_document', methods=['POST'])
def draft_document():
    # ✅ 3. SECURED API
    if 'user' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    data = request.get_json()
    doc_type = data.get('document_type') or data.get('doc_type') or ''
    answers = data.get('answers') or data
    if not doc_type:
        return jsonify({"error": "Document type not provided."}), 400
    try:
        document_text = render_template_doc(doc_type, answers)
        return jsonify({'document': document_text})
    except Exception as e:
        return jsonify({"error": "Document generation failed.", "details": str(e)}), 500



@app.route('/document_drafting')
def document_drafting_page():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('document_drafting.html')

@app.route("/chat", methods=["POST"])
def chat():
    if 'user' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"response": "Please send a message."}), 400

    # Step 1: Get the explanation from Gemini (this will be in English)
    explanation = get_groq_explanation(user_message, "")

    # Step 2: Check the user's preferred language from the session
    current_language_code = session.get('language', 'en') # Default to 'en' if not set

    # Step 3: If the language is not English, attempt to translate the response
    if current_language_code != 'en':
        print(f"Attempting to translate bot response to {current_language_code}...")
        # Your translate_with_gemini expects names like 'Hindi' or 'Telugu', not codes
        lang_name_map = {'hi': 'Hindi', 'te': 'Telugu', 'en': 'English'}
        target_lang_name = lang_name_map.get(current_language_code, 'English') # Default to English name

        translated_explanation = translate_with_gemini(explanation, target_lang_name)

        if translated_explanation:
            explanation = translated_explanation
            print(f"Successfully translated to {current_language_code}.")
        else:
            print(f"Failed to translate to {current_language_code}. Sending English response.")
            # Optionally, inform the user about the translation failure
            # explanation += f"\n\n(Translation to {target_lang_name} failed. Showing English.)"
            # flash(f"Translation to {target_lang_name} failed.", "warning")

    return jsonify({"response": explanation})

# In app.py, replace the old /legal_aid_checker route with this one.

@app.route('/legal_aid_checker', methods=['GET', 'POST'])
def legal_aid_checker():
    if 'user' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    if request.method == 'POST':
        try:
            state = request.json.get('state')
            situation = request.json.get('situation')

            if not state or not situation:
                return jsonify({"error": "State and situation description are required."}), 400

            # --- AI PROMPT ENGINEERING ---
            # This is the detailed prompt we send to the Groq LLM
            prompt = f"""
            Act as an expert legal aid advisor in India, assessing eligibility based on the Legal Services Authorities Act, 1987.

            Here is the user's situation:
            - State: {state}
            - Description: "{situation}"

            Based on this information, please provide a clear and helpful assessment in three parts:

            1.  **Eligibility Assessment:** Start with a clear "Likely Eligible" or "Likely Not Eligible" conclusion.
            2.  **Reasoning:** Briefly explain why, based on the user's description and the known criteria (e.g., being a woman, a child, a member of SC/ST, or meeting the state-specific income threshold).
            3.  **Actionable Next Steps:** Provide the name of the State Legal Services Authority for the user's state (e.g., "Maharashtra State Legal Services Authority"). Instruct them to contact this authority and provide the official website for it.

            Format the response clearly with headings for each part.
            """
            # ---------------------------

            # Use our existing Groq function to get the AI-powered assessment
            ai_response = get_groq_explanation("Legal Aid Eligibility Assessment", prompt)

            return jsonify({"response": ai_response})

        except Exception as e:
            return jsonify({"error": "An AI error occurred", "details": str(e)}), 500

    # This handles the initial page load (GET request)
    return render_template("legal_aid_checker.html")
@app.route("/legal_dictionary")
def legal_dictionary():
    # ✅ 2. ADDED PROTECTION
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template("legal_dictionary.html")
# In app.py, add this new route after deleting the old case law finder routes.
# Make sure you have 'import datetime' at the top of your file.

# In app.py, replace your /case_law_finder route with this one

# In app.py, replace your /case_law_finder route with this one

# In app.py, replace your /case_law_finder route with this one last time.

# In app.py
# Make sure 'get_groq_explanation' function is available in this file.

# This is the new, AI-powered version of your case law finder route
@app.route('/legal_aid_checker')
def legal_aid_checker_page():
    # ✅ 2. ADDED PROTECTION
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template("legal_aid_checker.html")
@app.route('/case_law_finder')
def case_law_finder_page():
    # ✅ 2. ADDED PROTECTION
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template("case_law_finder.html")
@app.route('/case_law_finder', methods=['GET', 'POST'])
def case_law_finder():
    if 'user' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    if request.method == 'POST':
        try:
            query = request.json.get('query', '')
            year = request.json.get('year', '')

            if not query:
                return jsonify({"error": "No query provided"}), 400
            
            # --- AI PROMPT ENGINEERING ---
            # We create a detailed prompt for the Groq LLM
            prompt = f"""
            Act as an expert legal researcher for the Supreme Court of India.
            Find up to 5 notable cases related to "{query}" from the year {year if year else 'the last few years'}.

            For each case, provide the following in a clean format:
            1.  **Case Title:** (e.g., P. Yuvaprakash v. State)
            2.  **Summary:** A concise 1-2 sentence summary of the main issue and the court's final decision.

            Do not include any introductory or concluding sentences. Just provide the list of cases.
            """
            # ---------------------------

            # We use our existing Groq function to get the answer
            case_results = get_groq_explanation("Legal Case Search", prompt)

            return jsonify({"cases": case_results})

        except Exception as e:
            return jsonify({"error": "An AI error occurred", "details": str(e)}), 500

    # This handles the initial page load (GET request)
    return render_template("case_law_finder.html")
# ---- Config ----
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

# Create uploads folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ---- Helpers ----
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_pdf_text(pdf_path):
    """Extract text from PDF file"""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        return None, str(e)
    return text.strip(), None

# Example: Hugging Face pipeline (already in your code)
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)
@app.route("/document_scanner")
def document_scanner_page():
    # ✅ 2. ADDED PROTECTION
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template("document_scanner.html")
# ---- Routes ----
@app.route("/document_scanner", methods=['GET', 'POST'])
def document_scanner():
    if 'user' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    if request.method == 'GET':
        # Render the HTML page
        return render_template("document_scanner.html")

    # Handle POST (AJAX) request
    data = request.get_json()
    document_text = data['document']
    language = data.get('language', 'en')

    # --- Your analysis logic here ---
    result = analyze_document(document_text)

    # Translate if needed
    if language != 'en':
        translated = translator.translate(result, dest=language).text
        result = translated

    return jsonify({'response': result})
translator = Translator()
@app.route('/analyze_document', methods=['POST'])
def analyze_document():
    document_text = request.json.get('document_text')
    selected_language = request.json.get('language', 'english')  # Get language parameter
    
    if not document_text:
        return jsonify({"error": "No document text provided"}), 400
    
    # --- Part 1: Get Content Analysis from Gemini (always in English first) ---
    analysis_prompt = f"""
    Please analyze this legal document and provide:
    1. Document Type (contract, agreement, etc.)
    2. Key Parties Involved
    3. Main Terms and Clauses
    4. Important Dates or Deadlines
    5. Rights and Obligations
    6. Potential Risks or Red Flags
    7. Overall Summary
    
    Document Text:
    {document_text}
    """
    
    content_analysis = get_groq_explanation("Document Analysis", analysis_prompt)
    
    # --- Part 2: Get Sentiment Analysis from BERT ---
    try:
        sentiment_results = sentiment_pipeline(document_text[:512])
        sentiment = sentiment_results[0]
    except Exception as e:
        sentiment = {"label": "UNKNOWN", "score": 0.0}
    
    # --- Part 3: Translate if needed ---
    final_analysis = content_analysis
    
    if selected_language != 'english':
        try:
            print(f"🔄 Translating to {selected_language} using Gemini...")
            
            # Try Gemini translation first
            gemini_translation = translate_with_gemini(content_analysis, selected_language)
            
            if gemini_translation:
                final_analysis = gemini_translation
                print(f"✅ Gemini translation to {selected_language} successful!")
            else:
                # Fallback to Google Translate
                lang_codes = {'telugu': 'te', 'hindi': 'hi'}
                target_lang = lang_codes.get(selected_language, 'en')
                
                if target_lang != 'en':
                    print(f"🔄 Fallback to Google Translate...")
                    translated = translator.translate(content_analysis[:2000], dest=target_lang)
                    final_analysis = translated.text
                    print(f"✅ Google Translate successful!")
                else:
                    final_analysis = content_analysis
                    
        except Exception as e:
            print(f"Translation error: {e}")
            final_analysis = content_analysis
    
    # --- Part 4: Return Results ---
    return jsonify({
        "analysis": final_analysis,
        "document_length": len(document_text.split()),
        "sentiment": {
            "label": sentiment['label'],
            "score": round(sentiment['score'], 4)
        },
        "language_used": selected_language
    })


# ADD this new route for PDF upload:
@app.route('/analyze_pdf', methods=['POST'])
def analyze_pdf():
    if 'pdf_file' not in request.files:
        return jsonify({"error": "No PDF file uploaded"}), 400
    
    file = request.files['pdf_file']
    selected_language = request.form.get('language', 'english')  # Get language from form data
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Extract text from PDF
        extracted_text, error = extract_pdf_text(filepath)
        
        # Clean up uploaded file
        os.remove(filepath)
        
        if error:
            return jsonify({"error": f"Failed to read PDF: {error}"}), 400
        
        if not extracted_text or len(extracted_text.strip()) < 50:
            return jsonify({"error": "Could not extract readable text from PDF. Please try a text-based PDF or paste the content manually."}), 400
        
        # Use same analysis as text input
        analysis_prompt = f"""
        Please analyze this legal document and provide:
        1. Document Type (contract, agreement, etc.)
        2. Key Parties Involved  
        3. Main Terms and Clauses
        4. Important Dates or Deadlines
        5. Rights and Obligations
        6. Potential Risks or Red Flags
        7. Overall Summary
        
        Document Text:
        {extracted_text}
        """
        
        analysis = get_groq_explanation("Document Analysis", analysis_prompt)  # Note: variable is 'analysis'
        
        # --- Translation Logic (Fixed variable name) ---
        final_analysis = analysis  # Start with the original analysis
        
        if selected_language != 'english':
            try:
                print(f"🔄 Translating to {selected_language} using Gemini...")
                
                # Try Gemini translation first
                gemini_translation = translate_with_gemini(analysis, selected_language)  # Use 'analysis' not 'content_analysis'
                
                if gemini_translation:
                    final_analysis = gemini_translation
                    print(f"✅ Gemini translation to {selected_language} successful!")
                else:
                    # Fallback to Google Translate
                    lang_codes = {'telugu': 'te', 'hindi': 'hi'}
                    target_lang = lang_codes.get(selected_language, 'en')
                    
                    if target_lang != 'en':
                        print(f"🔄 Fallback to Google Translate...")
                        translated = translator.translate(analysis[:2000], dest=target_lang)  # Use 'analysis'
                        final_analysis = translated.text
                        print(f"✅ Google Translate successful!")
                    else:
                        final_analysis = analysis
                        
            except Exception as e:
                print(f"Translation error: {e}")
                final_analysis = analysis  # Use 'analysis' not 'content_analysis'
        
        return jsonify({
            "analysis": final_analysis,  # Use translated version
            "document_length": len(extracted_text.split()),
            "extracted_text": extracted_text[:1000] + "..." if len(extracted_text) > 1000 else extracted_text,
            "language_used": selected_language  # Add this for UI display
        })
    
    return jsonify({"error": "Invalid file format. Please upload a PDF file."}), 400


# Find this route in app.py and replace it with the code below

@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    document_text = request.json.get('document_text')
    
    if not document_text:
        return jsonify({"error": "No text provided"}), 400
    
    # The BERT model pipeline does all the work here
    try:
        results = sentiment_pipeline(document_text)
        # The result is a list of dictionaries, we just need the first one
        sentiment = results[0]
        return jsonify({
            "label": sentiment['label'],
            "score": round(sentiment['score'], 4)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
