import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, url_for
from tensorflow.keras.models import load_model
from preprocess import detect_and_crop_face, preprocess_image
from werkzeug.utils import secure_filename
import base64
import traceback

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
    # Verificar la arquitectura del modelo
    print("📊 Resumen del modelo:")
    model.summary()
except Exception as e:
    print(f"❌ Error cargando modelo: {e}")
    print("🔍 Verificando archivos en el directorio...")
    if os.path.exists('models'):
        print("📁 Contenido de models/:")
        for file in os.listdir('models'):
            print(f"   - {file}")
    else:
        print("❌ La carpeta models/ no existe")
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
        print(f"🖼️ Procesando imagen: {image_path}")
        
        # Leer imagen
        img = cv2.imread(image_path)
        if img is None:
            print("❌ No se pudo leer la imagen con OpenCV")
            return None, "Error: No se pudo leer la imagen"
        
        print("👤 Detectando rostros...")
        # Detectar y recortar rostro
        face = detect_and_crop_face(img)
        
        if face is None:
            print("❌ No se detectó ningún rostro")
            return None, "No se detectó ningún rostro en la imagen"
        
        if model is None:
            print("❌ Modelo no disponible")
            return None, "Modelo no disponible"
        
        print("🧠 Realizando predicción...")
        # Preprocesar y predecir
        p = preprocess_image(face)
        x = np.expand_dims(p, axis=0)
        probs = model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        
        emotion = class_labels.get(idx, "Desconocido")
        emoji = class_emojis.get(idx, "")
        confidence = float(probs[idx])
        
        print(f"🎯 Resultado: {emotion} {emoji} - {confidence:.2f}")
        
        # Dibujar resultado en la imagen
        result_img = img.copy()
        gray = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
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
        success = cv2.imwrite(result_path, result_img)
        
        if not success:
            print("❌ Error guardando imagen resultado")
            return None, "Error guardando imagen resultado"
            
        print(f"💾 Imagen resultado guardada: {result_filename}")
        
        return {
            'emotion': emotion,
            'emoji': emoji,
            'confidence': confidence,
            'result_image': result_filename,
            'original_image': os.path.basename(image_path)
        }, None
        
    except Exception as e:
        print(f"💥 Error en analyze_emotion: {str(e)}")
        traceback.print_exc()
        return None, f"Error en el análisis: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Endpoint para subir y analizar imágenes"""
    try:
        print("📨 Recibiendo solicitud de upload...")
        
        # Verificar si se envió un archivo
        if 'file' not in request.files:
            print("❌ No se encontró 'file' en request.files")
            return jsonify({'error': 'No se seleccionó ningún archivo'}), 400
        
        file = request.files['file']
        print(f"📄 Archivo recibido: {file.filename}")

        # Verificar si se seleccionó un archivo
        if file.filename == '':
            print("❌ Nombre de archivo vacío")
            return jsonify({'error': 'No se seleccionó ningún archivo'}), 400

        # Verificar tipo de archivo
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print(f"💾 Guardando archivo en: {filepath}")
            
            file.save(filepath)
            
            # Verificar que el archivo se guardó correctamente
            if not os.path.exists(filepath):
                return jsonify({'error': 'Error guardando el archivo'}), 500
                
            print("🔍 Analizando emoción en la imagen...")
            
            # Analizar emoción
            result, error = analyze_emotion(filepath)
            
            if error:
                print(f"❌ Error en análisis: {error}")
                # Eliminar archivo si hay error
                if os.path.exists(filepath):
                    os.remove(filepath)
                return jsonify({'error': error}), 400
            
            print(f"✅ Análisis exitoso: {result['emotion']} - {result['confidence']:.2f}")
            
            response_data = {
                'success': True,
                'result': result,
                'result_image_url': url_for('static', filename=f'uploads/{result["result_image"]}'),
                'original_image_url': url_for('static', filename=f'uploads/{filename}')
            }
            
            return jsonify(response_data)
        else:
            print("❌ Tipo de archivo no permitido")
            return jsonify({'error': 'Tipo de archivo no permitido. Use PNG, JPG o JPEG'}), 400
            
    except Exception as e:
        print(f"💥 Error crítico en upload_file: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Error del servidor: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint para verificar que la aplicación está funcionando"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'upload_folder_exists': os.path.exists(app.config['UPLOAD_FOLDER'])
    })

@app.route('/clear_uploads', methods=['POST'])
def clear_uploads():
    """Limpiar archivos temporales"""
    try:
        count = 0
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                count += 1
        print(f"🗑️ Eliminados {count} archivos")
        return jsonify({'success': f'{count} archivos eliminados'})
    except Exception as e:
        print(f"❌ Error limpiando uploads: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)