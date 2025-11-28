import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, url_for
from tensorflow.keras.models import load_model
from preprocess import detect_and_crop_face, preprocess_image
from werkzeug.utils import secure_filename
import base64

app = Flask(__name__)
app.config['SECRET_KEY'] = 'tu_clave_secreta_aqui'

# Configuración de uploads
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Crear directorio de uploads si no existe
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Cargar modelo
try:
    model = load_model('models/emotion_cnn.h5')
    print("✅ Modelo cargado exitosamente")
except Exception as e:
    print(f"❌ Error cargando modelo: {e}")
    model = None

# Configuración de clases
class_labels = {0: "Enojado", 1: "Triste"}
class_emojis = {0: "😠", 1: "😢"}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_emotion(image_path):
    """Analiza la emoción en una imagen"""
    try:
        # Leer imagen
        img = cv2.imread(image_path)
        if img is None:
            return None, "Error: No se pudo leer la imagen"
        
        # Detectar y recortar rostro
        face = detect_and_crop_face(img)
        
        if face is None:
            return None, "No se detectó ningún rostro en la imagen"
        
        if model is None:
            return None, "Modelo no disponible"
        
        # Preprocesar y predecir
        p = preprocess_image(face)
        x = np.expand_dims(p, axis=0)
        probs = model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        
        emotion = class_labels.get(idx, "Desconocido")
        emoji = class_emojis.get(idx, "")
        confidence = float(probs[idx])
        
        # Dibujar resultado en la imagen
        result_img = img.copy()
        gray = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml").detectMultiScale(gray, 1.1, 5)
        
        for (x, y, w, h) in faces:
            # Color basado en confianza
            if confidence > 0.7:
                color = (0, 255, 0)  # Verde
            elif confidence > 0.5:
                color = (0, 255, 255)  # Amarillo
            else:
                color = (0, 165, 255)  # Naranja
            
            # Dibujar rectángulo alrededor del rostro
            cv2.rectangle(result_img, (x, y), (x+w, y+h), color, 2)
            
            # Texto con resultado
            text = f"{emoji} {emotion}: {confidence:.2f}"
            cv2.putText(result_img, text, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Guardar imagen con resultados
        result_filename = 'result_' + os.path.basename(image_path)
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        cv2.imwrite(result_path, result_img)
        
        return {
            'emotion': emotion,
            'emoji': emoji,
            'confidence': confidence,
            'result_image': result_filename,
            'original_image': os.path.basename(image_path)
        }, None
        
    except Exception as e:
        return None, f"Error en el análisis: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Endpoint para subir y analizar imágenes"""
    try:
        # Verificar si se envió un archivo
        if 'file' not in request.files:
            return jsonify({'error': 'No se seleccionó ningún archivo'}), 400
        
        file = request.files['file']
        
        # Verificar si se seleccionó un archivo
        if file.filename == '':
            return jsonify({'error': 'No se seleccionó ningún archivo'}), 400
        
        # Verificar tipo de archivo
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Analizar emoción
            result, error = analyze_emotion(filepath)
            
            if error:
                # Eliminar archivo si hay error
                if os.path.exists(filepath):
                    os.remove(filepath)
                return jsonify({'error': error}), 400
            
            return jsonify({
                'success': True,
                'result': result,
                'result_image_url': url_for('static', filename=f'uploads/{result["result_image"]}'),
                'original_image_url': url_for('static', filename=f'uploads/{filename}')
            })
        else:
            return jsonify({'error': 'Tipo de archivo no permitido. Use PNG, JPG o JPEG'}), 400
            
    except Exception as e:
        return jsonify({'error': f'Error del servidor: {str(e)}'}), 500

@app.route('/clear_uploads', methods=['POST'])
def clear_uploads():
    """Limpiar archivos temporales"""
    try:
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        return jsonify({'success': 'Archivos eliminados'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
