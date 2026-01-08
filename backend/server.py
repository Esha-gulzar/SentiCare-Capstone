from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper
from gtts import gTTS
import io
import base64
import random
import os
import tempfile

app = Flask(__name__)
CORS(app)

# Load Whisper model (using base model for better performance)
print("Loading Whisper model...")
try:
    model = whisper.load_model("base")
    print("Whisper model loaded successfully!")
except Exception as e:
    print(f"Error loading Whisper model: {e}")
    print("Make sure ffmpeg is installed: https://ffmpeg.org/download.html")
    model = None

# Predefined responses in both languages
RESPONSES = {
    'en': {
        'greetings': [
            "Assalam o Alaikum! I'm so glad you're here. How can I help you today?",
            "Hello there! It's wonderful to see you. What would you like to talk about?",
            "Assalam o Alaikum! Welcome! How are you doing today?",
            "Hi! I'm here and ready to chat with you. How's your day going?"
        ],
        'how_are_you': [
            "I'm doing great, thank you for asking! How are you feeling today?",
            "I'm wonderful! It's nice of you to ask. How about you?",
            "I'm here and ready to help! How are you doing?"
        ],
        'help': [
            "I'm here to chat with you! You can ask me questions or just have a conversation. What's on your mind?",
            "I can help answer questions, have conversations, or just listen. What would you like to do?",
            "Feel free to talk to me about anything! I'm here to help."
        ],
        'thanks': [
            "You're most welcome! It's my pleasure to help!",
            "Anytime! I'm happy to assist you.",
            "Glad I could help! Is there anything else you'd like to know?"
        ],
        'weather': [
            "I don't have access to real-time weather data, but I hope it's nice where you are!",
            "I can't check the weather right now, but I hope you're having a pleasant day!",
            "Unfortunately, I cannot access weather information at the moment."
        ],
        'name': [
            "I'm your friendly AI Assistant, here to help you anytime!",
            "You can call me AI Assistant. I'm here to chat and help you!",
            "I'm an AI Assistant created to assist and chat with you!"
        ],
        'default': [
            "That's interesting! Tell me more about that.",
            "I understand. What else would you like to discuss?",
            "Thank you for sharing that with me. How can I help you further?",
            "I'm listening. Please go on.",
            "That's a good point. What do you think about it?",
            "I appreciate you talking with me. What else is on your mind?"
        ]
    },
    'ur': {
        'greetings': [
            "السلام علیکم! میں بہت خوش ہوں کہ آپ یہاں ہیں۔ آج میں آپ کی کیسے مدد کر سکتا ہوں؟",
            "ہیلو! آپ کو دیکھ کر بہت اچھا لگا۔ آپ کس بارے میں بات کرنا چاہیں گے؟",
            "السلام علیکم! خوش آمدید! آج آپ کیسا محسوس کر رہے ہیں؟",
            "ہائے! میں یہاں ہوں اور آپ سے بات کرنے کے لیے تیار ہوں۔ آپ کا دن کیسا گزر رہا ہے؟"
        ],
        'how_are_you': [
            "میں بہت اچھا ہوں، پوچھنے کا شکریہ! آج آپ کیسا محسوس کر رہے ہیں؟",
            "میں بہترین ہوں! آپ کا پوچھنا بہت اچھا لگا۔ آپ کا کیا حال ہے؟",
            "میں یہاں ہوں اور مدد کے لیے تیار ہوں! آپ کیسے ہیں؟"
        ],
        'help': [
            "میں آپ سے بات کرنے کے لیے یہاں ہوں! آپ مجھ سے سوالات پوچھ سکتے ہیں یا صرف گفتگو کر سکتے ہیں۔ آپ کے ذہن میں کیا ہے؟",
            "میں سوالات کے جوابات دینے، گفتگو کرنے، یا صرف سننے میں مدد کر سکتا ہوں۔ آپ کیا کرنا چاہیں گے؟",
            "کسی بھی چیز کے بارے میں مجھ سے بات کریں! میں مدد کے لیے یہاں ہوں۔"
        ],
        'thanks': [
            "خوش آمدید! آپ کی مدد کرنا میری خوشی ہے!",
            "کبھی بھی! میں آپ کی مدد کرکے خوش ہوں۔",
            "خوشی ہے کہ میں مدد کر سکا! کیا آپ کو کچھ اور جاننا ہے؟"
        ],
        'weather': [
            "میرے پاس حقیقی وقت کے موسم کا ڈیٹا نہیں ہے، لیکن امید ہے کہ آپ کے یہاں موسم اچھا ہوگا!",
            "میں ابھی موسم چیک نہیں کر سکتا، لیکن امید ہے کہ آپ کا دن اچھا گزر رہا ہے!",
            "بدقسمتی سے، میں اس وقت موسم کی معلومات تک رسائی حاصل نہیں کر سکتا۔"
        ],
        'name': [
            "میں آپ کا دوستانہ AI اسسٹنٹ ہوں، ہر وقت آپ کی مدد کے لیے حاضر!",
            "آپ مجھے AI اسسٹنٹ کہہ سکتے ہیں۔ میں آپ سے بات کرنے اور مدد کرنے کے لیے یہاں ہوں!",
            "میں ایک AI اسسٹنٹ ہوں جو آپ کی مدد اور گفتگو کے لیے بنایا گیا ہے!"
        ],
        'default': [
            "یہ دلچسپ ہے! مجھے اس کے بارے میں مزید بتائیں۔",
            "میں سمجھتا ہوں۔ آپ اور کیا بات کرنا چاہیں گے؟",
            "میرے ساتھ یہ شیئر کرنے کا شکریہ۔ میں آپ کی مزید کیسے مدد کر سکتا ہوں؟",
            "میں سن رہا ہوں۔ براہ کرم جاری رکھیں۔",
            "یہ اچھا نکتہ ہے۔ آپ اس کے بارے میں کیا سوچتے ہیں؟",
            "میں آپ سے بات کرنے کی تعریف کرتا ہوں۔ آپ کے ذہن میں اور کیا ہے؟"
        ]
    }
}

def get_response_category(message, language):
    """Determine which category of response to use based on the message"""
    message_lower = message.lower()
    
    # Check for greetings
    greeting_words = {
        'en': ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening'],
        'ur': ['سلام', 'ہیلو', 'السلام علیکم', 'صبح بخیر', 'شام بخیر']
    }
    
    # Check for "how are you"
    how_are_you_words = {
        'en': ['how are you', 'how do you do', 'how are u', 'whats up', "what's up"],
        'ur': ['کیسے ہو', 'کیسے ہیں', 'کیا حال ہے', 'کیسا ہے']
    }
    
    # Check for help requests
    help_words = {
        'en': ['help', 'what can you do', 'assist', 'support'],
        'ur': ['مدد', 'آپ کیا کر سکتے ہیں', 'مدد کریں']
    }
    
    # Check for thanks
    thanks_words = {
        'en': ['thank', 'thanks', 'appreciate', 'grateful'],
        'ur': ['شکریہ', 'شکر ہے', 'نوازش']
    }
    
    # Check for weather
    weather_words = {
        'en': ['weather', 'temperature', 'forecast', 'rain', 'sunny'],
        'ur': ['موسم', 'بارش', 'درجہ حرارت']
    }
    
    # Check for name questions
    name_words = {
        'en': ['your name', 'who are you', 'what are you'],
        'ur': ['آپ کا نام', 'آپ کون ہیں', 'تم کون ہو']
    }
    
    # Check each category
    for word in greeting_words[language]:
        if word in message_lower:
            return 'greetings'
    
    for phrase in how_are_you_words[language]:
        if phrase in message_lower:
            return 'how_are_you'
    
    for word in help_words[language]:
        if word in message_lower:
            return 'help'
    
    for word in thanks_words[language]:
        if word in message_lower:
            return 'thanks'
    
    for word in weather_words[language]:
        if word in message_lower:
            return 'weather'
    
    for phrase in name_words[language]:
        if phrase in message_lower:
            return 'name'
    
    return 'default'

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """Transcribe audio using Whisper"""
    temp_path = None
    try:
        # Check if Whisper model is loaded
        if model is None:
            return jsonify({'error': 'Whisper model not loaded. Please install ffmpeg.'}), 500
        
        if 'audio' not in request.files:
            print("ERROR: No audio file in request")
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        language = request.form.get('language', 'en')
        
        print(f"Received audio file: {audio_file.filename}")
        print(f"Content type: {audio_file.content_type}")
        print(f"Language: {language}")
        
        # Use system temp directory instead of current directory
        temp_dir = tempfile.gettempdir()
        file_ext = '.webm' if 'webm' in str(audio_file.content_type) else '.wav'
        temp_path = os.path.join(temp_dir, f'temp_audio_{os.getpid()}{file_ext}')
        
        print(f"Saving audio to: {temp_path}")
        
        # Save the file
        audio_file.save(temp_path)
        
        # Verify file exists and has content
        if not os.path.exists(temp_path):
            print(f"ERROR: File not saved: {temp_path}")
            return jsonify({'error': 'Failed to save audio file'}), 500
        
        file_size = os.path.getsize(temp_path)
        print(f"Audio file saved successfully ({file_size} bytes)")
        
        if file_size < 100:
            print("ERROR: Audio file too small (probably empty)")
            os.remove(temp_path)
            return jsonify({'error': 'Audio file is empty or too short. Please speak for at least 1-2 seconds.'}), 400
        
        # Transcribe with Whisper
        whisper_lang = 'en' if language == 'en' else 'ur'
        print(f"Starting transcription with language: {whisper_lang}")
        
        result = model.transcribe(temp_path, language=whisper_lang, fp16=False)
        
        transcribed_text = result['text'].strip()
        print(f"Transcription successful: '{transcribed_text}'")
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
            print("Temp file cleaned up")
        
        if not transcribed_text:
            return jsonify({'error': 'Could not transcribe audio. Please speak more clearly.'}), 400
        
        return jsonify({'text': transcribed_text})
    
    except Exception as e:
        error_msg = str(e)
        print(f"ERROR in transcribe: {error_msg}")
        import traceback
        traceback.print_exc()
        
        # Clean up on error
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        
        # Provide more helpful error messages
        if "ffmpeg" in error_msg.lower() or "avconv" in error_msg.lower():
            return jsonify({'error': 'FFmpeg is not installed. Please install FFmpeg to use voice features.'}), 500
        
        return jsonify({'error': f'Transcription failed: {error_msg}'}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Generate chatbot response"""
    try:
        data = request.json
        user_message = data.get('message', '')
        language = data.get('language', 'en')
        
        print(f"Received message: {user_message} (Language: {language})")
        
        # Get appropriate response category
        category = get_response_category(user_message, language)
        
        # Select a random response from the category
        response_text = random.choice(RESPONSES[language][category])
        
        print(f"Generating response: {response_text}")
        
        # Generate audio using gTTS
        # Use correct language codes for gTTS
        if language == 'ur':
            # gTTS uses 'ur' for Urdu
            tts = gTTS(text=response_text, lang='ur', slow=False, tld='com')
        else:
            tts = gTTS(text=response_text, lang='en', slow=False, tld='com')
        
        # Save to memory buffer
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        # Encode to base64
        audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')
        
        print(f"Audio generated successfully (Language: {language})")
        
        return jsonify({
            'response': response_text,
            'audio': audio_base64
        })
    
    except Exception as e:
        print(f"Error in chat: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'Server is running'})

@app.route('/speak', methods=['POST'])
def speak():
    """Generate speech audio without chat logic - for greetings"""
    try:
        data = request.json
        text = data.get('text', '')
        language = data.get('language', 'en')
        
        print(f"Generating speech: {text} (Language: {language})")
        
        # Generate audio using gTTS
        if language == 'ur':
            tts = gTTS(text=text, lang='ur', slow=False, tld='com')
        else:
            tts = gTTS(text=text, lang='en', slow=False, tld='com')
        
        # Save to memory buffer
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        # Encode to base64
        audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')
        
        return jsonify({'audio': audio_base64})
    
    except Exception as e:
        print(f"Error in speak: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask server...")
    print("Server will be available at http://localhost:5000")
    print("Make sure to install requirements: pip install flask flask-cors openai-whisper gtts")
    print("\n" + "="*50)
    print("🚀 Backend server is running!")
    print("Keep this terminal open while using the app")
    print("="*50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)