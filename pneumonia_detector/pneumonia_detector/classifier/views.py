from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from django.contrib.auth.decorators import login_required
from .models import PredictionHistory
import tensorflow as tf
import numpy as np
from django.contrib import messages
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocessor

# --- Public Landing Page View ---
def home_view(request):
    return render(request, 'classifier/home.html')

# --- Load All Necessary Models ---
VALIDATOR_MODEL_PATH = 'ml_models/xray_validator_model.h5'
validator_model = tf.keras.models.load_model(VALIDATOR_MODEL_PATH)
PNEUMONIA_MODEL_PATH = 'ml_models/model_resnet.h5' 
pneumonia_model = tf.keras.models.load_model(PNEUMONIA_MODEL_PATH)


@login_required
def classify_image(request):
    if request.method == 'POST' and request.FILES.get('image_upload'):
        image_upload = request.FILES['image_upload']
        fss = FileSystemStorage()
        file_name = fss.save(image_upload.name, image_upload)
        file_path = fss.path(file_name)

        try:
            # Validator Logic
            img_validator = tf.keras.utils.load_img(file_path, target_size=(128, 128))
            img_array_validator = tf.keras.utils.img_to_array(img_validator)
            img_array_validator = np.expand_dims(img_array_validator, axis=0)
            validation_score = validator_model.predict(img_array_validator)[0][0]

            if validation_score < 0.8: # Threshold set to 80%
                messages.error(request, "Invalid image detected. Please upload a chest X-ray.")
                return redirect('classify_image')

        except Exception as e:
            messages.error(request, "Could not process the uploaded file. Please try another image.")
            return redirect('classify_image')

        # Prediction Logic
        img_pneumonia = tf.keras.utils.load_img(file_path, target_size=(256, 256))
        img_array_pneumonia = tf.keras.utils.img_to_array(img_pneumonia)
        img_array_pneumonia = np.expand_dims(img_array_pneumonia, axis=0)
        preprocessed_img = resnet_preprocessor(img_array_pneumonia)
        prediction = pneumonia_model.predict(preprocessed_img)
        score = tf.nn.sigmoid(prediction[0][0]).numpy()
        
        # Determine result string and confidence
        if score > 0.5:
            result_label_str = "Pneumonia"
            result_confidence = f"Pneumonia Detected (Confidence: {score:.2%})"
        else:
            result_label_str = "Normal"
            result_confidence = f"Normal (Confidence: {1-score:.2%})"

        # Save the full result with confidence to history
        PredictionHistory.objects.create(
            user=request.user,
            image=image_upload,
            prediction_result=result_confidence
        )
        
        messages.success(request, 'Image analysis complete and saved to your history!')
        
        context = {
            'prediction': result_confidence,
            'uploaded_image_url': fss.url(file_name),
            'result_type': result_label_str # Pass the simple string for the report link
        }
        return render(request, 'classifier/result.html', context)

    return render(request, 'classifier/upload.html')


@login_required
def prediction_history_view(request):
    history = PredictionHistory.objects.filter(user=request.user).order_by('-timestamp')
    return render(request, 'classifier/history.html', {'history': history})


# --- NEW FUNCTION: GENERATE REPORT VIEW ---
@login_required
def generate_report_view(request, result, lang):
    # Determine the language dictionary
    if lang == 'telugu':
        MESSAGES = {
            'Normal': {
                'title': "శుభవార్త! మీ ఊపిరితిత్తులు ఆరోగ్యంగా ఉన్నాయి.",
                'advice': "మీ X-rayలో ఎటువంటి సమస్యలు లేవు. మీరు ఆరోగ్యంగా ఉన్నారు. ఈ మంచి అలవాట్లను కొనసాగించండి.",
                'color': "#28a745"
            },
            'Pneumonia': {
                'title': "అనారోగ్య సూచన. దయచేసి వెంటనే వైద్యుడిని సంప్రదించండి.",
                'advice': "మీ X-rayలో నిమోనియా లక్షణాలు ఉన్నట్లు గుర్తించబడింది. ఖచ్చితమైన నిర్ధారణ కోసం ఆలస్యం చేయకుండా వైద్య నిపుణులను సంప్రదించడం అత్యవసరం.",
                'color': "#dc3545"
            }
        }
    else: # Default to English
        MESSAGES = {
            'Normal': {
                'title': "Great News! Your Lungs Appear Healthy.",
                'advice': "Your analysis indicates no apparent issues in the X-ray. Your lungs are clear. Continue maintaining good health practices.",
                'color': "#28a745"
            },
            'Pneumonia': {
                'title': "Abnormality Detected. Immediate Consultation Required.",
                'advice': "Signs consistent with Pneumonia were identified in the X-ray. It is essential to consult a qualified medical professional immediately for accurate diagnosis and treatment.",
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
    
    return render(request, 'classifier/report_page.html', context)