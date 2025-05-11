import cv2
import requests
import json

# --- CONFIGURACIÓN ---
ROBOFLOW_API_KEY = ""  # Reemplaza con tu clave de API
ROBOFLOW_MODEL_ID = ""  # Reemplaza con el ID de tu modelo
ROBOFLOW_API_URL = f"https://detect.roboflow.com/{ROBOFLOW_MODEL_ID}?api_key={ROBOFLOW_API_KEY}"
CONFIDENCE_THRESHOLD = 0.5  # Umbral de confianza para la detección

def capturar_foto():
    """Captura una foto desde la cámara web."""
    cap = cv2.VideoCapture(0)  # 0 indica la cámara predeterminada
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return None

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("No se pudo capturar la foto.")
        return None

    return frame

def analizar_imagen_roboflow(imagen):
    """Envía la imagen a la API de Roboflow para su análisis."""
    _, img_encoded = cv2.imencode('.jpg', imagen)
    files = {'image': ('image.jpg', img_encoded.tobytes(), 'image/jpeg')}
    try:
        response = requests.post(ROBOFLOW_API_URL, files=files)
        response.raise_for_status()  # Lanza una excepción para códigos de error HTTP
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error al comunicarse con la API de Roboflow: {e}")
        return None

def mostrar_resultados(resultados):
    """Muestra los resultados de la detección."""
    hay_persona = False
    objetos_detectados = []

    if resultados and 'predictions' in resultados:
        for prediction in resultados['predictions']:
            if prediction['confidence'] > CONFIDENCE_THRESHOLD:
                label = prediction['class']
                confidence = prediction['confidence']
                objetos_detectados.append(f"{label} ({confidence:.2f})")
                if label.lower() == 'person':
                    hay_persona = True

    if hay_persona:
        print("¡Se detectó una persona en la foto!")
    else:
        print("No se detectó ninguna persona en la foto.")

    if objetos_detectados:
        print("Otros objetos detectados:")
        for objeto in objetos_detectados:
            print(f"- {objeto}")
    else:
        print("No se detectaron otros objetos con la confianza especificada.")

if __name__ == "__main__":
    print("Abriendo la cámara web. Por favor, espera...")
    foto = capturar_foto()

    if foto is not None:
        print("Foto capturada. Analizando la imagen...")
        resultados = analizar_imagen_roboflow(foto)

        if resultados:
            print("Resultados del análisis:")
            mostrar_resultados(resultados)
        else:
            print("No se pudieron obtener resultados del análisis.")