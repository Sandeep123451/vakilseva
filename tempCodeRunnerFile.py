from flask import Flask, request, jsonify, render_template, send_file, session, redirect, url_for, flash, g
import requests
from io import BytesIO
from fpdf import FPDF
from transformers import pipeline
import html
import json
import PyPDF2
from werkzeug.utils import secure_filename
import os
import pandas as pd # Not currently used, but keeping it
import datetime
from googletrans import Translator # Keeping as fallback, though Gemini is preferred
import time
import random
import google.generativeai as genai
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from jinja2 import Environment, FileSystemLoader
from flask_babel import Babel, _, get_locale, lazy_gettext as _l
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'manjula143#' # Keep your existing secret key
app.config['LANGUAGES'] = {
    'en': 'English',
    'hi': 'Hindi',
    'te': 'Telugu'
}
babel = Babel(app, locale_selector=lambda: session.get('language', 'en'))

# Your API keys
KANOON_API_KEY = "4dbc8b0b8ed8e19162e08bbe008897a6c58c1b5f"
GROQ_API_KEY = "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"
GEMINI_API_KEY = "AIzaSyAciInnPnj4BGzLUMt1cWKJfEYi4ioT6aY" 
genai.configure(api_key=GEMINI_API_KEY)
# We'll keep the old Translator instance as a fallback for now, though Gemini is preferred
g_translator = Translator() # Renamed to avoid conflict with potential local use in functions

@app.route('/set_language/<lang_code>')
def set_language(lang_code):
    session['language'] = lang_code
    print(f"DEBUG: Language set in session: {session.get('language')}")
    return redirect(request.referrer or url_for('home'))

# Indian Kanoon API Base URL
KANOON_API_URL = "https://api.indiankanoon.org/search/"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions" # This is a placeholder, as you're using Gemini now


# --- NEW: User Management System ---
# In-memory user store for demonstration purposes.
# In a real application, this would be a database (e.g., SQLAlchemy with SQLite/PostgreSQL).
users = {} # {user_id: UserObject}
next_user_id = 1

class User:
    def __init__(self, id, username, email, password_hash):
        self.id = id
        self.username = username
        self.email = email
        self.password_hash = password_hash

    def __repr__(self):
        return f"<User {self.username}>"

    @staticmethod
    def get(user_id):
        return users.get(user_id)

    @staticmethod
    def get_by_email(email):
        for user in users.values():
            if user.email == email:
                return user
        return None

# Add a dummy admin user for testing (password: password123)
# Hashed password for 'password123'
initial_password_hash = generate_password_hash('password123')
admin_user = User(next_user_id, 'Admin User', 'admin@example.com', initial_password_hash)
users[next_user_id] = admin_user
next_user_id += 1
# --- END NEW: User Management System ---

# --- UPDATED: login_required Decorator ---
# This decorator ensures a user is logged in before accessing a route.
# It also loads the current user into `g.user` for easy access in templates/views.
# --- UPDATED: login_required Decorator ---
# This decorator ensures a user is logged in before accessing a route.
# It also loads the current user into `g.user` for easy access in templates/views.
from functools import wraps # <--- ADD THIS IMPORT at the top of app.py

def login_required(f):
    @wraps(f) # <--- ADD THIS DECORATOR
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash(_('Please log in to access this page.'), 'error')
            return redirect(url_for('login'))
        
        # Load user object into g
        g.user = User.get(session['user_id']) 
        if not g.user: # If user_id exists but user object can't be found (e.g., server restart)
            session.pop('user_id', None)
            flash(_l('Your session has expired. Please log in again.'), 'error')
            return redirect(url_for('login'))
        
        return f(*args, **kwargs)
    return decorated_function

# --- END UPDATED: login_required Decorator ---


# --- Functions (Existing) ---
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

def get_groq_explanation(user_question, kanoon_data, max_retries=2):
    """
    Using Google Gemini Pro with improved error handling
    """
    
    for attempt in range(max_retries):
        try:
            print(f"🔄 Gemini attempt {attempt + 1}/{max_retries}")
            
            # Create Gemini Pro model
            model = genai.GenerativeModel('gemini-1.5-flash')

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
                    max_output_tokens=1500,
                    temperature=0.3,
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
                return _l("The AI service has reached its usage limit for now. Please try again in a few minutes.")
            elif "safety" in error_msg.lower():
                return _l("I cannot provide information on this topic due to safety restrictions. Please try rephrasing your question.")
            elif "authentication" in error_msg.lower():
                return _l("There is an issue with API authentication. Please contact support.")
                
            if attempt < max_retries - 1:
                time.sleep(2)
    
    print(f"❌ All Gemini attempts failed")
    return _l("""I'm having trouble connecting to the AI service right now. This could be due to:

1. **High demand** - Try again in 1-2 minutes
2. **Content restrictions** - Try rephrasing your question
3. **Network issues** - Check your internet connection

Please try again shortly!""")

def translate_with_gemini(text, target_language_name): # Renamed target_language to target_language_name
    """
    Use Gemini to translate text instead of Google Translate
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""Translate the following English text to {target_language_name}. Maintain the formatting, structure, and meaning. Only provide the translation, no additional text:

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

# --- END Functions (Existing) ---


# --- AUTHENTICATION ROUTES (UPDATED & NEW) ---

@app.route('/', methods=['GET', 'POST'])
def login():
    if 'user_id' in session: # Check for 'user_id' in session, not 'user'
        return redirect(url_for('home')) # Already logged in

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.get_by_email(email) # Use the new User model

        if user and check_password_hash(user.password_hash, password):
            session['user_id'] = user.id # Store user_id, not just email
            flash(_l('Login successful!'), 'success')
            return redirect(url_for('home'))
        else:
            flash(_l('Invalid email or password. Please try again.'), 'error')
            # return redirect(url_for('login')) # No need to redirect on POST error, render template instead
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    global next_user_id # Declare that we intend to modify the global variable

    if 'user_id' in session:
        return redirect(url_for('home')) # Already logged in

    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if not username or not email or not password or not confirm_password:
            flash(_l('All fields are required.'), 'error')
            return render_template('register.html', username=username, email=email)

        if password != confirm_password:
            flash(_l('Passwords do not match.'), 'error')
            return render_template('register.html', username=username, email=email)

        if User.get_by_email(email):
            flash(_l('An account with this email already exists. Please log in.'), 'error')
            return render_template('register.html', username=username, email=email)

        # Hash the password
        hashed_password = generate_password_hash(password)

        # Create new user and add to our in-memory store
        new_user = User(next_user_id, username, email, hashed_password)
        users[next_user_id] = new_user
        next_user_id += 1 # Increment for the next user

        flash(_l('Registration successful! Please log in.'), 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None) # Clear the user_id from session
    flash(_l('You have been logged out.'), 'info')
    return redirect(url_for('login')) # Redirect to login page after logout

# --- END AUTHENTICATION ROUTES ---


# --- PROTECTED PAGE ROUTES (APPLYING login_required) ---

@app.route("/home")
@login_required # Apply the decorator here
def home():
    # g.user is available here thanks to the decorator
    return render_template("index.html")

@app.route('/knowledge_hub')
@login_required # Apply the decorator here
def knowledge_hub():
    return render_template('knowledge_hub.html')

@app.route('/legal_dictionary_search', methods=['POST'])
@login_required # Apply the decorator here for API endpoint
def legal_dictionary_search():
    term = request.json.get('term')
    if not term:
        return jsonify({"response": _l("Please enter a legal term.")}), 400
    definition = legal_dict.get(term.lower())
    if definition:
        return jsonify({"response": definition})
    else:
        explanation = get_groq_explanation(term, "")
        return jsonify({"response": explanation})

@app.route("/chatbot")
@login_required # Apply the decorator here
def chatbot():
    current_flask_locale = str(get_locale())
    print(f"DEBUG: Flask rendering chatbot.html with locale: {current_flask_locale}")
    return render_template("chatbot.html", current_lang=current_flask_locale)

@app.route('/draft_document', methods=['POST'])
@login_required # Apply the decorator here for API endpoint
def draft_document():
    data = request.get_json()
    doc_type = data.get('document_type') or data.get('doc_type') or ''
    answers = data.get('answers') or data
    if not doc_type:
        return jsonify({"error": _l("Document type not provided.")}), 400
    try:
        document_text = render_template_doc(doc_type, answers)
        return jsonify({'document': document_text})
    except Exception as e:
        return jsonify({"error": _l("Document generation failed."), "details": str(e)}), 500

@app.route('/document_drafting')
@login_required # Apply the decorator here
def document_drafting_page():
    return render_template('document_drafting.html')

@app.route("/chat", methods=["POST"])
@login_required # Apply the decorator here for API endpoint
def chat():
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"response": _l("Please send a message.")}), 400

    explanation = get_groq_explanation(user_message, "")

    current_language_code = session.get('language', 'en')

    if current_language_code != 'en':
        print(f"Attempting to translate bot response to {current_language_code}...")
        lang_name_map = {'hi': 'Hindi', 'te': 'Telugu', 'en': 'English'}
        target_lang_name = lang_name_map.get(current_language_code, 'English')

        translated_explanation = translate_with_gemini(explanation, target_lang_name)

        if translated_explanation:
            explanation = translated_explanation
            print(f"Successfully translated to {current_language_code}.")
        else:
            print(f"Failed to translate to {current_language_code}. Sending English response.")
            # flash(f"Translation to {target_lang_name} failed.", "warning") # Flash messages are for full page loads, not AJAX

    return jsonify({"response": explanation})

@app.route('/legal_aid_checker', methods=['GET', 'POST'])
@login_required # Apply the decorator here for both GET and POST
def legal_aid_checker():
    if request.method == 'POST':
        try:
            state = request.json.get('state')
            situation = request.json.get('situation')

            if not state or not situation:
                return jsonify({"error": _l("State and situation description are required.")}), 400

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
            ai_response = get_groq_explanation("Legal Aid Eligibility Assessment", prompt)

            return jsonify({"response": ai_response})

        except Exception as e:
            return jsonify({"error": _l("An AI error occurred"), "details": str(e)}), 500

    return render_template("legal_aid_checker.html") # For GET requests

@app.route("/legal_dictionary")
@login_required # Apply the decorator here
def legal_dictionary():
    return render_template("legal_dictionary.html")

@app.route('/case_law_finder')
@login_required # Apply the decorator here
def case_law_finder_page():
    return render_template("case_law_finder.html")

@app.route('/case_law_finder_api', methods=['POST']) # Renamed to _api to differentiate from the GET route
@login_required # Apply the decorator here for API endpoint
def case_law_finder_api():
    try:
        query = request.json.get('query', '')
        year = request.json.get('year', '')

        if not query:
            return jsonify({"error": _l("No query provided")}), 400
        
        prompt = f"""
        Act as an expert legal researcher for the Supreme Court of India.
        Find up to 5 notable cases related to "{query}" from the year {year if year else 'the last few years'}.

        For each case, provide the following in a clean format:
        1.  **Case Title:** (e.g., P. Yuvaprakash v. State)
        2.  **Summary:** A concise 1-2 sentence summary of the main issue and the court's final decision.

        Do not include any introductory or concluding sentences. Just provide the list of cases.
        """
        case_results = get_groq_explanation("Legal Case Search", prompt)

        return jsonify({"cases": case_results})

    except Exception as e:
        return jsonify({"error": _l("An AI error occurred"), "details": str(e)}), 500

# ---- Document Scanner Config & Helpers ----
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_pdf_text(pdf_path):
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        return None, str(e)
    return text.strip(), None

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

@app.route("/document_scanner")
@login_required # Apply the decorator here
def document_scanner_page():
    return render_template("document_scanner.html")

@app.route("/document_scanner_analyze_text", methods=['POST']) # Renamed this route to avoid conflict and be more descriptive
@login_required # Apply the decorator here for API endpoint
def document_scanner_analyze_text():
    data = request.get_json()
    document_text = data.get('document')
    language = data.get('language', 'en')

    if not document_text:
        return jsonify({"error": _l("No document text provided")}), 400

    # --- Your analysis logic here ---
    # This will now call the analyze_document logic (which internally uses Gemini and BERT)
    # Re-using the logic from the /analyze_document route for consistency
    response_data = analyze_document_logic(document_text, language)

    return jsonify(response_data)


# --- NEW: Refactored document analysis logic into a function ---
# This function encapsulates the core analysis and translation,
# so it can be called by both text input and PDF input routes.
def analyze_document_logic(document_text, selected_language):
    if not document_text:
        return {"error": _l("No document text provided"), "status": 400}
    
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
        sentiment_results = sentiment_pipeline(document_text[:512]) # Limit text for BERT
        sentiment = sentiment_results[0]
    except Exception as e:
        print(f"Sentiment analysis error: {e}")
        sentiment = {"label": "UNKNOWN", "score": 0.0}
    
    # --- Part 3: Translate if needed ---
    final_analysis = content_analysis
    
    if selected_language != 'english':
        try:
            print(f"🔄 Translating to {selected_language} using Gemini...")
            lang_name_map = {'telugu': 'Telugu', 'hindi': 'Hindi', 'english': 'English'}
            target_lang_name = lang_name_map.get(selected_language, 'English')
            
            gemini_translation = translate_with_gemini(content_analysis, target_lang_name)
            
            if gemini_translation:
                final_analysis = gemini_translation
                print(f"✅ Gemini translation to {selected_language} successful!")
            else:
                # Fallback to Google Translate if Gemini fails
                lang_codes = {'telugu': 'te', 'hindi': 'hi'}
                target_lang = lang_codes.get(selected_language, 'en')
                
                if target_lang != 'en':
                    print(f"🔄 Fallback to Google Translate...")
                    translated = g_translator.translate(content_analysis[:2000], dest=target_lang) # Use g_translator
                    final_analysis = translated.text
                    print(f"✅ Google Translate successful!")
                else:
                    final_analysis = content_analysis
                    
        except Exception as e:
            print(f"Translation error: {e}")
            final_analysis = content_analysis
    
    # --- Part 4: Return Results ---
    return {
        "analysis": final_analysis,
        "document_length": len(document_text.split()),
        "sentiment": {
            "label": sentiment['label'],
            "score": round(sentiment['score'], 4)
        },
        "language_used": selected_language
    }

# --- END NEW: Refactored document analysis logic ---

@app.route('/analyze_document', methods=['POST']) # This route now just calls the shared logic
@login_required
def analyze_document_route(): # Renamed to avoid conflict with the function analyze_document_logic
    document_text = request.json.get('document_text')
    selected_language = request.json.get('language', 'english')
    
    if not document_text:
        return jsonify({"error": _l("No document text provided")}), 400

    response_data = analyze_document_logic(document_text, selected_language)
    
    # Handle error case from analyze_document_logic
    if "error" in response_data:
        return jsonify(response_data), response_data.get("status", 400)
    
    return jsonify(response_data)


@app.route('/analyze_pdf', methods=['POST'])
@login_required # Apply the decorator here
def analyze_pdf():
    if 'pdf_file' not in request.files:
        return jsonify({"error": _l("No PDF file uploaded")}), 400
    
    file = request.files['pdf_file']
    selected_language = request.form.get('language', 'english')
    
    if file.filename == '':
        return jsonify({"error": _l("No file selected")}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        extracted_text, error = extract_pdf_text(filepath)
        os.remove(filepath) # Clean up uploaded file
        
        if error:
            return jsonify({"error": _l("Failed to read PDF: ") + error}), 400
        
        if not extracted_text or len(extracted_text.strip()) < 50:
            return jsonify({"error": _l("Could not extract readable text from PDF. Please try a text-based PDF or paste the content manually.")}), 400
        
        # Call the shared analysis logic
        response_data = analyze_document_logic(extracted_text, selected_language)
        
        # Handle error case from analyze_document_logic
        if "error" in response_data:
            return jsonify(response_data), response_data.get("status", 400)

        # Add extracted text to the response (truncated if long)
        response_data['extracted_text'] = extracted_text[:1000] + "..." if len(extracted_text) > 1000 else extracted_text

        return jsonify(response_data)
    
    return jsonify({"error": _l("Invalid file format. Please upload a PDF file.")}), 400

@app.route('/analyze_sentiment', methods=['POST'])
@login_required # Apply the decorator here for API endpoint
def analyze_sentiment():
    document_text = request.json.get('document_text')
    
    if not document_text:
        return jsonify({"error": _l("No text provided")}), 400
    
    try:
        results = sentiment_pipeline(document_text)
        sentiment = results[0]
        return jsonify({
            "label": sentiment['label'],
            "score": round(sentiment['score'], 4)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500 # Added status code

# Run the app
if __name__ == "__main__":
    app.run(debug=True)