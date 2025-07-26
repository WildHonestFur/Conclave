import cv2
import mediapipe as mp
import random
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import mysql.connector as m

mp_face_mesh = mp.solutions.face_mesh
con = m.connect(user='root', password='****', host='localhost', database='conclave', autocommit=True)
cur = con.cursor()

cap = cv2.VideoCapture(0)
bg = cv2.imread("AI_Template.jpg")

roles = ["The Hacker", "The Muscle", "The Decoy", "The Mastermind", "The Leader", "The Strategist", "The Illusionist", "The Wildcard"]
random.shuffle(roles)

adjectives = [
    "Neural", "Quantum", "Cognitive", "Adaptive", "Sentient",
    "Synthetic", "Deep", "Autonomous", "Intelligent", "Digital",
    "Algorithmic", "Futuristic", "Machine", "Virtual", "Cyber",
    "Smart", "Learning", "Automated", "Data-Driven", "Robotic"
]

nouns = [
    "Synths", "Bots", "Ciphers", "Agents", "Processors",
    "Networks", "Intellects", "Minds", "Circuits", "Systems",
    "Analyzers", "Engines", "Constructs", "Sentinels", "Architects",
    "Coders", "Explorers", "Guardians", "Rangers", "Optimizers"
]

def draw_text_custom(frame, text, position, font_path, font_size=32, color=(255, 255, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color)

    bbox = font.getbbox(text)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR), text_width

face_boxes = []

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=8,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    assigned_roles = []

    while cap.isOpened():
        face_boxes = []
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        new = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape

        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                x_vals = []
                y_vals = []

                for lm in face_landmarks.landmark:
                    x_vals.append(int(lm.x * w))
                    y_vals.append(int(lm.y * h))

                x_min = max(min(x_vals) - 20, 0)
                y_min = max(min(y_vals) - 60, 0)
                x_max = min(max(x_vals) + 20, w)
                y_max = min(max(y_vals) + 15, h)

                face_boxes.append((x_min, y_min, x_max, y_max))

            if not assigned_roles or len(assigned_roles) != len(face_boxes):
                random.shuffle(roles)
                assigned_roles = roles[:len(face_boxes)]

            for i, (x_min, y_min, x_max, y_max) in enumerate(face_boxes):
                cv2.line(frame, ((x_min+x_max)//2, y_min), ((x_min+x_max)//2+30, y_min-30), (0, 0,  255), 2)
                frame, text_width = draw_text_custom(
                    frame,
                    assigned_roles[i],
                    ((x_min+x_max)//2+45, y_min-30-30),
                    font_path=r"VT323-Regular.ttf",
                    font_size=30,
                    color=(255, 0, 0)
                )
                cv2.line(frame, ((x_min+x_max)//2+30, y_min-30), ((x_min+x_max)//2+30+text_width+20, y_min-30), (0, 0,  255), 2)
                

        cv2.imshow('AI Booth', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):

            query = "SELECT name FROM leader;"
            cur.execute(query)
            info = cur.fetchall()
            names = []
            for i in info:
                names.append(i[0])

            teamname = random.choice(adjectives)+" "+random.choice(nouns)
            while teamname in names:
                  teamname = random.choice(adjectives)+" "+random.choice(nouns)

            query = f"INSERT INTO leader VALUES ('{teamname}', 100);"
            cur.execute(query)
            
            cv2.imwrite(f"static\Teams\{teamname}.jpg", frame)
            desired_width = 1000
            scale = desired_width / frame.shape[1]
            new_size = (desired_width, int(frame.shape[0] * scale))
            frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

            h, w, _ = bg.shape
            x, y = int(22*3.3), 300

            px, py = (w-desired_width)//2, int(230*3)

            bg, text_width = draw_text_custom(
                bg,
                "Team name: ",
                (x, y),
                font_path=r"VT323-Regular.ttf",
                font_size=100,
                color=(255, 0, 0)
            )
            bg, __ = draw_text_custom(
                bg,
                teamname,
                (x+text_width+15, y),
                font_path=r"VT323-Regular.ttf",
                font_size=100,
                color=(255, 225, 255)
            )
            bg, text_width = draw_text_custom(
                bg,
                "Species: ",
                (x, y+100),
                font_path=r"VT323-Regular.ttf",
                font_size=100,
                color=(255, 0, 0)
            )
            bg, __ = draw_text_custom(
                bg,
                random.choice(["Human", "Unknown"]),
                (x+text_width+15, y+100),
                font_path=r"VT323-Regular.ttf",
                font_size=100,
                color=(255, 225, 255)
            )
            bg, text_width = draw_text_custom(
                bg,
                "Threat Level: ",
                (x, y+200),
                font_path=r"VT323-Regular.ttf",
                font_size=100,
                color=(255, 0, 0)
            )
            bg, __ = draw_text_custom(
                bg,
                random.choice(["Extreme", "High", "Moderate", "Infinite"]),
                (x+text_width+15, y+200),
                font_path=r"VT323-Regular.ttf",
                font_size=100,
                color=(255, 225, 255)
            )

            bg[py:py+frame.shape[0], px:px+frame.shape[1]] = frame

            bg, text_width = draw_text_custom(
                bg,
                "Members: ",
                (x, py+frame.shape[0]+50),
                font_path=r"VT323-Regular.ttf",
                font_size=100,
                color=(255, 0, 0)
            )

            num = len(face_boxes)
            fixed_width = 200
            fixed_height = 210
            startx = (w-(num*fixed_width+(num-1)*50))//2

            for i, (x_min, y_min, x_max, y_max) in enumerate(face_boxes):
                face_crop = new[y_min:y_max, x_min:x_max]
                face_resized = cv2.resize(face_crop, (fixed_width, fixed_height))
                bg[py+frame.shape[0]+190:py+frame.shape[0]+190+fixed_height, startx+i*(fixed_width+50):startx+(1+i)*(fixed_width+50)-50] = face_resized
                _, width = draw_text_custom(
                    bg,
                    assigned_roles[i],
                    (startx+i*(fixed_width+50), py+frame.shape[0]+190+fixed_height+20),
                    font_path=r"VT323-Regular.ttf",
                    font_size=35,
                    color=(255, 0, 0)
                )
                bg, __ = draw_text_custom(
                    bg,
                    assigned_roles[i],
                    (startx+i*(fixed_width+50)+((fixed_width-width)//2), py+frame.shape[0]+190+fixed_height+20),
                    font_path=r"VT323-Regular.ttf",
                    font_size=35,
                    color=(255, 0, 0)
                )
            
            cv2.imwrite(f"static\TeamPhotos\{teamname}.jpg", bg)
            break

cap.release()
cv2.destroyAllWindows()
con.close()
