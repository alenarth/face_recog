import face_recognition
import cv2
import numpy as np
import os

def carregar_encodings(caminho):
    encodings = []
    nomes = []
    for nome_pasta in os.listdir(caminho):
        pasta = os.path.join(caminho, nome_pasta)
        if os.path.isdir(pasta):
            for arquivo in os.listdir(pasta):
                caminho_completo = os.path.join(pasta, arquivo)
                imagem = face_recognition.load_image_file(caminho_completo)
                encodings_da_imagem = face_recognition.face_encodings(imagem)
                if encodings_da_imagem:
                    encoding = encodings_da_imagem[0]
                    encodings.append(encoding)
                    nomes.append(nome_pasta)
                else:
                    print(f"Nenhum rosto encontrado em {arquivo}.")
    return encodings, nomes

caminho_base = 'known_faces'
encodings_conhecidos, nomes_conhecidos = carregar_encodings(caminho_base)

video_capture = cv2.VideoCapture(0)
video_capture.set(3, 640)
video_capture.set(4, 480)

processar_cada_frame = 5
frame_count = 0

while True:
    ret, frame = video_capture.read()
    frame_count += 1
    pequeno_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_pequeno_frame = pequeno_frame[:, :, ::-1]

    if frame_count % processar_cada_frame == 0:
        localizacoes_rostos = face_recognition.face_locations(rgb_pequeno_frame)
        encodings_rostos = face_recognition.face_encodings(rgb_pequeno_frame, localizacoes_rostos)

        nomes_rostos = []
        for face_encoding in encodings_rostos:
            matches = face_recognition.compare_faces(encodings_conhecidos, face_encoding)
            nome = "Desconhecido"

            if True in matches:
                primeiro_match_index = matches.index(True)
                nome = nomes_conhecidos[primeiro_match_index]

            nomes_rostos.append(nome)
    else:
        continue  # Se não for um frame para processar, pule para o próximo frame

    for (top, right, bottom, left), nome in zip(localizacoes_rostos, nomes_rostos):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, nome, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    cv2.imshow('Vídeo', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
