from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import cv2
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'oral_lesion_detection_secret_key_2024')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff', 'tif', 'bmp'}

# Global variables for model and processing tools
model = None
tokenizer = None
class_names = None
IMG_SIZE = (128, 128)
MAX_TEXT_LENGTH = 100
VOCAB_SIZE = 5000


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_artifacts():
    """Load the trained model and processing artifacts"""
    global model, tokenizer, class_names

    try:
        # Load model
        model = load_model('best_hybrid_model.h5')
        print("✅ Model loaded successfully")

        # Load tokenizer
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        print("✅ Tokenizer loaded successfully")

        # Load class names
        with open('class_names.pickle', 'rb') as handle:
            class_names = pickle.load(handle)
        print("✅ Class names loaded successfully")

        return True
    except Exception as e:
        print(f"❌ Error loading artifacts: {e}")
        return False


def preprocess_image(image_path):
    """Preprocess image for model prediction"""
    try:
        # Read and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            return None

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize
        img = cv2.resize(img, IMG_SIZE)

        # Normalize
        img = img.astype(np.float32) / 255.0

        return np.expand_dims(img, axis=0)
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None


def preprocess_text(description):
    """Preprocess text description for model prediction"""
    try:
        # Clean description - handle empty/None descriptions
        if not description or str(description).strip() in ['', 'nan', 'none', 'no description',
                                                           'no description available']:
            desc_clean = "no clinical description provided"
        else:
            desc_clean = str(description).lower().strip()

        # Convert to sequence and pad
        sequence = tokenizer.texts_to_sequences([desc_clean])
        padded_sequence = pad_sequences(sequence, maxlen=MAX_TEXT_LENGTH, padding='post', truncating='post')

        return padded_sequence
    except Exception as e:
        print(f"Error preprocessing text: {e}")
        return None


def predict_lesion(image_path, description):
    """Make prediction using the hybrid model"""
    try:
        # Preprocess inputs
        processed_image = preprocess_image(image_path)
        processed_text = preprocess_text(description)

        if processed_image is None:
            return None, "Error processing image"

        if processed_text is None:
            return None, "Error processing text description"

        # Make prediction
        predictions = model.predict([processed_image, processed_text], verbose=0)

        # Get top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_probabilities = predictions[0][top_3_indices]

        results = []
        for i, (idx, prob) in enumerate(zip(top_3_indices, top_3_probabilities)):
            results.append({
                'rank': i + 1,
                'class_name': class_names[idx],
                'confidence': float(prob * 100),
                'confidence_percentage': f"{prob * 100:.2f}%"
            })

        return results, "Success"

    except Exception as e:
        print(f"Prediction error: {e}")
        return None, str(e)


def create_confidence_bar(confidence):
    """Create HTML for confidence bar"""
    width = min(100, confidence)
    color_class = "high-confidence" if confidence > 70 else "medium-confidence" if confidence > 40 else "low-confidence"

    return f'''
    <div class="confidence-bar-container">
        <div class="confidence-bar {color_class}" style="width: {width}%"></div>
        <span class="confidence-text">{confidence:.1f}%</span>
    </div>
    '''


# Load artifacts when app starts
@app.before_request
def before_first_request():
    global model, tokenizer, class_names
    if model is None:
        print("🚀 Initializing Oral Lesion Detection App...")
        if not load_artifacts():
            flash("Error loading model artifacts. Please check if model files are available.", "error")


@app.route('/')
def index():
    """Home page with upload form"""
    return render_template('index.html', class_names=class_names)


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    if request.method == 'POST':
        # Check if image file is present
        if 'image' not in request.files:
            flash('No image file selected', 'error')
            return redirect(request.url)

        file = request.files['image']

        # Validate file
        if file.filename == '':
            flash('No image selected', 'error')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            try:
                # Get description (optional)
                description = request.form.get('description', '').strip()

                # Save uploaded file
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                unique_filename = f"{timestamp}_{filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(filepath)

                # Make prediction
                results, status = predict_lesion(filepath, description)

                if results is None:
                    flash(f'Prediction error: {status}', 'error')
                    return redirect(request.url)

                # Prepare results for display
                top_prediction = results[0]

                # Create confidence bars for all results
                for result in results:
                    result['confidence_bar'] = create_confidence_bar(result['confidence'])

                # Generate result message based on confidence
                confidence = top_prediction['confidence']
                if confidence > 80:
                    confidence_level = "High Confidence"
                    confidence_class = "high-confidence-text"
                elif confidence > 60:
                    confidence_level = "Moderate Confidence"
                    confidence_class = "medium-confidence-text"
                else:
                    confidence_level = "Low Confidence - Clinical Correlation Recommended"
                    confidence_class = "low-confidence-text"

                return render_template('results.html',
                                       filename=unique_filename,
                                       description=description if description else "No description provided",
                                       results=results,
                                       top_prediction=top_prediction,
                                       confidence_level=confidence_level,
                                       confidence_class=confidence_class,
                                       timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            except Exception as e:
                flash(f'Error during prediction: {str(e)}', 'error')
                return redirect(request.url)
        else:
            flash('Invalid file type. Please upload an image file (PNG, JPG, JPEG, TIFF, BMP).', 'error')
            return redirect(request.url)

    return redirect(url_for('index'))


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    if 'image' not in request.files:
        return jsonify({'error': 'Image file is required'}), 400

    file = request.files['image']
    description = request.form.get('description', '')

    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid image file'}), 400

    try:
        # Save temporary file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{filename}")
        file.save(filepath)

        # Make prediction
        results, status = predict_lesion(filepath, description)

        # Clean up temporary file
        if os.path.exists(filepath):
            os.remove(filepath)

        if results is None:
            return jsonify({'error': status}), 500

        return jsonify({
            'success': True,
            'predictions': results,
            'top_prediction': results[0]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/about')
def about():
    """About page"""
    return render_template('about.html', class_names=class_names)


@app.route('/guidelines')
def guidelines():
    """Clinical guidelines page"""
    return render_template('guidelines.html')


@app.errorhandler(413)
def too_large(e):
    flash('File too large. Please upload images smaller than 16MB.', 'error')
    return redirect(request.url)


@app.errorhandler(500)
def internal_error(error):
    flash('Internal server error. Please try again.', 'error')
    return redirect(url_for('index'))


@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404


if __name__ == '__main__':
    print("🔬 Starting Oral Lesion Detection Web Application...")
    print("📁 Model files check:")
    print(f"   - Model file: {'✅ Found' if os.path.exists('best_hybrid_model.h5') else '❌ Missing'}")
    print(f"   - Tokenizer: {'✅ Found' if os.path.exists('tokenizer.pickle') else '❌ Missing'}")
    print(f"   - Class names: {'✅ Found' if os.path.exists('class_names.pickle') else '❌ Missing'}")

    # Load artifacts
    load_artifacts()

    # Run app
    app.run(debug=True, host='0.0.0.0', port=5000)