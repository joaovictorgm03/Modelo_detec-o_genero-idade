import cv2
from ultralytics import YOLO
from deepface import DeepFace
import logging
import os

# Desativa logs detalhados do TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# --- INICIALIZAÇÃO DOS MODELOS ---

# Carrega o modelo YOLOv8 pré-treinado especificamente para detecção de ROSTOS
try:
    model_yolo = YOLO('yolov8m-face.pt')
    print("Modelo YOLOv8-Face carregado com sucesso.")
except Exception as e:
    print(f"Erro ao carregar o modelo YOLOv8-Face: {e}")
    # Mensagem de erro corrigida para refletir o modelo correto
    print("Certifique-se de que o arquivo 'yolov8m-face.pt' está na pasta do projeto.")
    exit()

# Aquecimento do DeepFace (opcional, mas recomendado)
try:
    print("Aquecendo o modelo DeepFace...")
    # Criar uma imagem de teste ou usar uma existente pode acelerar a primeira análise
    DeepFace.analyze(cv2.imread("face_test.jpg", cv2.IMREAD_GRAYSCALE), actions=['age', 'gender'], enforce_detection=False)
    print("Modelos DeepFace prontos.")
except ValueError:
    print("Modelos DeepFace foram baixados e estão prontos.")


# --- PROCESSAMENTO DA WEBCAM ---

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro: Não foi possível abrir a webcam.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Roda a inferência do YOLOv8-Face no quadro
    results = model_yolo(frame)

    # Processa os resultados
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Pega as coordenadas da bounding box do rosto
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Recorta a imagem do rosto detectado com um pequeno preenchimento (padding)
            face_img = frame[max(0, y1-10):min(frame.shape[0], y2+10), max(0, x1-10):min(frame.shape[1], x2+10)]

            if face_img.size == 0:
                continue

            try:
                # Roda a análise de idade e gênero com DeepFace
                analysis = DeepFace.analyze(face_img, actions=['age', 'gender'], enforce_detection=False)

                # >>> INÍCIO DO BLOCO MODIFICADO <<<
                if isinstance(analysis, list) and len(analysis) > 0:
                    face_data = analysis[0]
                    
                    # Pega o gênero dominante e a confiança associada a ele
                    dominant_gender = face_data['dominant_gender']
                    gender_confidence = face_data['gender'][dominant_gender]

                    # CONDIÇÃO: só mostra a análise se a confiança for maior que 95%
                    if gender_confidence > 95:
                        age = face_data['age']
                        gender_map = {'Man': 'Homem', 'Woman': 'Mulher'}
                        gender = gender_map.get(dominant_gender, 'N/A')

                        # Texto com a análise detalhada e a confiança
                        text = f"{gender}, {age} anos ({gender_confidence:.1f}%)"
                        
                        # Desenha a bounding box VERDE (alta confiança)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # Coloca o texto com os resultados
                        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        # Se a confiança for baixa, mostra um status diferente
                        text = "Analisando..."
                        
                        # Desenha a bounding box AMARELA (baixa confiança)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        # Coloca o texto genérico
                        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                # >>> FIM DO BLOCO MODIFICADO <<<

            except Exception as e:
                # Se der erro na análise do DeepFace, desenha uma caixa vermelha
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "Rosto", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    # Mostra o quadro resultante
    cv2.imshow('Detector de Idade e Genero - YOLOv8-Face + DeepFace', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera os recursos
cap.release()
cv2.destroyAllWindows()