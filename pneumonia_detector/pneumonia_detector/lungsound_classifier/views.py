from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.core.files.storage import FileSystemStorage
from django.contrib import messages
from .models import SoundPredictionHistory
import tensorflow as tf
import numpy as np
import librosa
import librosa.display
import io
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 1. Load your trained lung sound model
LUNG_SOUND_MODEL_PATH = 'ml_models/lung_sound_model.h5' 
sound_model = tf.keras.models.load_model(LUNG_SOUND_MODEL_PATH)

# 2. Define the class names your model was trained on
CLASS_NAMES = ['crackles', 'normal', 'rhonchi', 'wheezes']
MODEL_INPUT_SIZE = (224, 224) 

def audio_to_spectrogram(audio_path):
    # ... (audio processing function remains the same) ...
    try:
        y, sr = librosa.load(audio_path, duration=5)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)
        fig, ax = plt.subplots()
        librosa.display.specshow(S_dB, sr=sr, fmax=8000, ax=ax)
        ax.axis('off')
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        return img
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

@login_required
def classify_sound_view(request):
    if request.method == 'POST' and request.FILES.get('audio_upload'):
        audio_upload = request.FILES['audio_upload']
        
        fss = FileSystemStorage()
        file_name = fss.save(audio_upload.name, audio_upload)
        file_path = fss.path(file_name)

        spectrogram_img = audio_to_spectrogram(file_path)

        if spectrogram_img is None:
            messages.error(request, "Could not process the audio file. It might be corrupted or in an unsupported format.")
            return render(request, 'lungsound_classifier/lungsound_upload.html')

        img_resized = spectrogram_img.resize(MODEL_INPUT_SIZE)
        img_array = tf.keras.utils.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)

        prediction = sound_model.predict(img_array)
        predicted_class_index = np.argmax(prediction[0])
        
        # --- NEW: Get the simple result string ---
        result_label_str = CLASS_NAMES[predicted_class_index].capitalize()
        # ----------------------------------------
        
        confidence = prediction[0][predicted_class_index]
        final_result = f"{result_label_str} (Confidence: {confidence:.2%})"

        SoundPredictionHistory.objects.create(
            user=request.user,
            audio_file=audio_upload,
            prediction_result=final_result
        )

        messages.success(request, 'Lung sound analysis complete!')
        context = {
            'prediction': final_result,
            'uploaded_audio_url': fss.url(file_name),
            'result_type': result_label_str # Pass simple string for report link
        }
        return render(request, 'lungsound_classifier/lungsound_result.html', context)

    return render(request, 'lungsound_classifier/lungsound_upload.html')

@login_required
def sound_history_view(request):
    history = SoundPredictionHistory.objects.filter(user=request.user).order_by('-timestamp')
    return render(request, 'lungsound_classifier/sound_history.html', {'history': history})

# --- NEW FUNCTION: GENERATE SOUND REPORT VIEW ---
@login_required
def generate_sound_report_view(request, result, lang):
    # Base messages for abnormal sounds
    ABNORMAL_ADVICE = "Abnormal lung sounds were detected. Please consult a qualified medical professional immediately for accurate diagnosis and treatment."
    ABNORMAL_TITLE = "Abnormality Detected. Immediate Consultation Required."

    if lang == 'telugu':
        MESSAGES = {
            'Normal': {
                'title': "మీ శ్వాస ధ్వనులు సాధారణంగా ఉన్నాయి.",
                'advice': "మీ ఊపిరితిత్తుల నుండి వచ్చే శబ్దాలు స్పష్టంగా ఉన్నాయి. ఇది మంచి సంకేతం. మీ మంచి ఆరోగ్య అలవాట్లను కొనసాగించండి.",
                'color': "#28a745"
            },
            'Wheezes': {
                'title': ABNORMAL_TITLE,
                'advice': f"గుర్తించిన ధ్వని: ఊపిరి పీల్చుకునేటప్పుడు ఈల (Wheezes). శ్వాస మార్గము సంకోచించినట్లు సూచిస్తుంది. {ABNORMAL_ADVICE}",
                'color': "#dc3545"
            },
            'Crackles': {
                'title': ABNORMAL_TITLE,
                'advice': f"గుర్తించిన ధ్వని: గరగర (Crackles). చిన్న శ్వాస మార్గాలలో ద్రవం లేదా శ్లేష్మం ఉన్నట్లు సూచిస్తుంది. {ABNORMAL_ADVICE}",
                'color': "#dc3545"
            },
            'Rhonchi': {
                'title': ABNORMAL_TITLE,
                'advice': f"గుర్తించిన ధ్వని: పెద్ద శ్వాస మార్గాలలో అడ్డంకి (Rhonchi). ఇది మందపాటి శ్లేష్మం కారణంగా జరగవచ్చు. {ABNORMAL_ADVICE}",
                'color': "#dc3545"
            }
        }
    else: # Default to English
        MESSAGES = {
            'Normal': {
                'title': "Excellent News! Your Lung Sounds Are Clear.",
                'advice': "Your analysis indicates clear, normal respiratory sounds. This is a positive result. Continue maintaining good health practices.",
                'color': "#28a750"
            },
            'Wheezes': {
                'title': ABNORMAL_TITLE,
                'advice': f"Detected Sound: Wheezes (a whistling sound). This suggests airway constriction. {ABNORMAL_ADVICE}",
                'color': "#dc3545"
            },
            'Crackles': {
                'title': ABNORMAL_TITLE,
                'advice': f"Detected Sound: Crackles (short, popping sounds). This suggests fluid or mucus in the small airways. {ABNORMAL_ADVICE}",
                'color': "#dc3545"
            },
            'Rhonchi': {
                'title': ABNORMAL_TITLE,
                'advice': f"Detected Sound: Rhonchi (low-pitched, rumbling). This suggests large airway blockage, often due to thick mucus. {ABNORMAL_ADVICE}",
                'color': "#dc3545"
            }
        }
    
    message_data = MESSAGES.get(result, MESSAGES['Normal'])

    context = {
        'title': message_data['title'],
        'advice': message_data['advice'],
        'color': message_data['color'],
        'result_type': result
    }
    
    # We will use a new template for audio reports
    return render(request, 'lungsound_classifier/sound_report_page.html', context)