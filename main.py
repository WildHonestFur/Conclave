import cv2
import mediapipe as mp
import random
from PIL import ImageFont, ImageDraw, Image
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

cap = cv2.VideoCapture(0)

roles = ["The Hacker", "The Muscle", "The Decoy", "The Mastermind", "The Leader", "The Strategist", "The Illusionist", "The Wildcard"]
random.shuffle(roles)

def draw_text_custom(frame, text, position, font_path, font_size=32, color=(255, 255, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color)

    bbox = font.getbbox(text)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR), text_width

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=8,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    assigned_roles = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape

        results = face_mesh.process(rgb)

        face_boxes = []
        if results.multi_face_landmarks:
            face_boxes = []

            for face_landmarks in results.multi_face_landmarks:
                x_vals = []
                y_vals = []

                for lm in face_landmarks.landmark:
                    x_vals.append(int(lm.x * w))
                    y_vals.append(int(lm.y * h))

                x_min = max(min(x_vals) - 10, 0)
                y_min = max(min(y_vals) - 10, 0)
                x_max = min(max(x_vals) + 10, w)
                y_max = min(max(y_vals) + 10, h)

                face_boxes.append((x_min, y_min, x_max, y_max))

            if not assigned_roles or len(assigned_roles) != len(face_boxes):
                random.shuffle(roles)
                assigned_roles = roles[:len(face_boxes)]

            for i, (x_min, y_min, x_max, y_max) in enumerate(face_boxes):
                cv2.line(frame, ((x_min+x_max)//2, y_min), ((x_min+x_max)//2+30, y_min-30), (255, 0,  0), 2)
                frame, text_width = draw_text_custom(
                    frame,
                    assigned_roles[i],
                    ((x_min+x_max)//2+45, y_min-30-30),
                    #font_path=r"Fonts\Share_Tech_Mono\ShareTechMono-Regular.ttf",
                    font_path=r"Fonts\VT323\VT323-Regular.ttf",
                    font_size=30,
                    color=(0, 0, 255)
                )
                cv2.line(frame, ((x_min+x_max)//2+30, y_min-30), ((x_min+x_max)//2+30+text_width+20, y_min-30), (255, 0,  0), 2)
                

        cv2.imshow('AI Booth', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
