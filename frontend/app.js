let currentLanguage = 'en';
let isRecording = false;
let mediaRecorder = null;
let audioChunks = [];

const translations = {
    en: {
        title: 'SentiCare',
        subtitle: 'Here to help',
        greeting: 'Assalamualaikum!',
        instruction: 'How are you feeling today? Start typing or press the microphone button to speak.',
        placeholder: 'Type your message here...',
        recording: 'Recording... Speak now',
        processing: 'Processing your message...',
        speaking: 'AI is speaking...'
    },
    ur: {
        title: 'سینٹی کیئر',
        subtitle: 'مدد کے لیے حاضر',
        greeting: 'السلام علیکم!',
        instruction: 'آج آپ کیسا محسوس کر رہے ہیں؟ ٹائپ کریں یا بولنے کے لیے مائیکروفون بٹن دبائیں۔',
        placeholder: '...یہاں اپنا پیغام لکھیں',
        recording: 'ریکارڈنگ... اب بولیں',
        processing: '...آپ کے پیغام پر کارروائی ہو رہی ہے',
        speaking: '...AI بول رہا ہے'
    }
};

function setLanguage(lang) {
    currentLanguage = lang;
    
    // Update UI text
    document.getElementById('headerTitle').textContent = translations[lang].title;
    document.getElementById('headerSubtitle').textContent = translations[lang].subtitle;
    document.getElementById('greeting').textContent = translations[lang].greeting;
    document.getElementById('instruction').textContent = translations[lang].instruction;
    document.getElementById('textInput').placeholder = translations[lang].placeholder;
    
    // Update button states
    document.getElementById('btnEn').classList.toggle('active', lang === 'en');
    document.getElementById('btnUr').classList.toggle('active', lang === 'ur');
    
    // Update text direction
    if (lang === 'ur') {
        document.getElementById('textInput').style.direction = 'rtl';
        document.getElementById('textInput').style.textAlign = 'right';
    } else {
        document.getElementById('textInput').style.direction = 'ltr';
        document.getElementById('textInput').style.textAlign = 'left';
    }
    
    // Clear chat history and show new greeting
    const chatHistory = document.getElementById('chatHistory');
    chatHistory.innerHTML = '';
    
    const greetingText = lang === 'en' 
        ? 'Assalam o Alaikum! How are you doing today?' 
        : 'السلام علیکم! آج آپ کیسے ہیں؟';
    
    addMessage(greetingText, false);
    
    // Play greeting audio in new language
    playGreeting(greetingText, lang);
}

async function playGreeting(text, language) {
    try {
        const response = await fetch('http://localhost:5000/speak', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                text: text,
                language: language
            })
        });
        
        if (response.ok) {
            const data = await response.json();
            if (data.audio) {
                const audio = new Audio('data:audio/mp3;base64,' + data.audio);
                audio.play();
            }
        }
    } catch (error) {
        console.log('Could not play greeting audio:', error);
    }
}

function addMessage(text, isUser) {
    const chatHistory = document.getElementById('chatHistory');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user' : 'bot'}`;
    messageDiv.textContent = text;
    chatHistory.appendChild(messageDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

function setStatus(message, isActive = false) {
    const statusIndicator = document.getElementById('statusIndicator');
    statusIndicator.textContent = message;
    statusIndicator.className = isActive ? 'status-indicator active' : 'status-indicator';
}

async function toggleRecording() {
    const micButton = document.getElementById('micButton');
    
    // Check if browser supports recording
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert('Your browser does not support voice recording. Please use Chrome, Edge, or Firefox.');
        return;
    }
    
    if (!isRecording) {
        try {
            console.log('Requesting microphone access...');
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            console.log('Microphone access granted!');
            
            // Check if MediaRecorder is supported
            if (!window.MediaRecorder) {
                alert('MediaRecorder is not supported in your browser.');
                stream.getTracks().forEach(track => track.stop());
                return;
            }
            
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];
            
            mediaRecorder.ondataavailable = (event) => {
                console.log('Audio data received:', event.data.size, 'bytes');
                audioChunks.push(event.data);
            };
            
            mediaRecorder.onstop = async () => {
                console.log('Recording stopped. Total chunks:', audioChunks.length);
                const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                console.log('Audio blob created:', audioBlob.size, 'bytes');
                await processAudio(audioBlob);
                stream.getTracks().forEach(track => track.stop());
            };
            
            mediaRecorder.onerror = (event) => {
                console.error('MediaRecorder error:', event.error);
                alert('Recording error: ' + event.error);
            };
            
            mediaRecorder.start();
            console.log('Recording started...');
            isRecording = true;
            micButton.classList.add('recording');
            setStatus(translations[currentLanguage].recording, true);
        } catch (error) {
            console.error('Error accessing microphone:', error);
            alert('Could not access microphone. Error: ' + error.message + '\n\nPlease:\n1. Check browser permissions\n2. Make sure microphone is connected\n3. Try reloading the page');
        }
    } else {
        console.log('Stopping recording...');
        mediaRecorder.stop();
        isRecording = false;
        micButton.classList.remove('recording');
        setStatus(translations[currentLanguage].processing, true);
    }
}

async function processAudio(audioBlob) {
    try {
        console.log('Processing audio blob...');
        const formData = new FormData();
        formData.append('audio', audioBlob, 'recording.webm');
        formData.append('language', currentLanguage);
        
        console.log('Sending audio to server...');
        const response = await fetch('http://localhost:5000/transcribe', {
            method: 'POST',
            body: formData
        });
        
        console.log('Server response status:', response.status);
        
        if (!response.ok) {
            const errorText = await response.text();
            console.error('Server error:', errorText);
            throw new Error(`Server error: ${response.status} - ${errorText}`);
        }
        
        const data = await response.json();
        console.log('Transcription result:', data);
        
        if (data.text) {
            addMessage(data.text, true);
            await getChatbotResponse(data.text);
        } else if (data.error) {
            setStatus('Error: ' + data.error);
            console.error('Transcription error:', data.error);
        } else {
            setStatus('Could not understand audio. Please try again.');
        }
    } catch (error) {
        console.error('Error processing audio:', error);
        setStatus('⚠️ Server not running! Start server.py first.');
        alert('Cannot connect to server. Error: ' + error.message + '\n\nPlease make sure:\n1. You ran: python server.py\n2. Server is running on http://localhost:5000\n3. Check the terminal for errors');
    }
}

async function sendMessage() {
    const textInput = document.getElementById('textInput');
    const message = textInput.value.trim();
    
    if (message) {
        addMessage(message, true);
        textInput.value = '';
        await getChatbotResponse(message);
    }
}

async function getChatbotResponse(userMessage) {
    try {
        setStatus(translations[currentLanguage].processing, true);
        
        const response = await fetch('http://localhost:5000/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: userMessage,
                language: currentLanguage
            })
        });
        
        const data = await response.json();
        
        if (data.response) {
            addMessage(data.response, false);
            
            // Play audio response
            if (data.audio) {
                setStatus(translations[currentLanguage].speaking, true);
                const audio = new Audio('data:audio/mp3;base64,' + data.audio);
                audio.onended = () => setStatus('');
                audio.play();
            } else {
                setStatus('');
            }
        }
    } catch (error) {
        console.error('Error getting chatbot response:', error);
        setStatus('Error connecting to server. Please try again.');
        addMessage('Sorry, I am having trouble connecting. Please try again.', false);
    }
}

function handleKeyPress(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
}

// Initialize with a greeting
window.addEventListener('load', async () => {
    setTimeout(async () => {
        // Greeting based on current language
        const greetingText = currentLanguage === 'en' 
            ? 'Assalam o Alaikum! How are you doing today?' 
            : 'السلام علیکم! آج آپ کیسے ہیں؟';
        
        addMessage(greetingText, false);
        
        // Play audio greeting
        try {
            const response = await fetch('http://localhost:5000/speak', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    text: greetingText,
                    language: currentLanguage
                })
            });
            
            if (response.ok) {
                const data = await response.json();
                if (data.audio) {
                    const audio = new Audio('data:audio/mp3;base64,' + data.audio);
                    audio.play();
                }
            }
        } catch (error) {
            console.log('Could not play greeting audio:', error);
            // It's okay if audio fails, at least show the text
        }
    }, 800);
});