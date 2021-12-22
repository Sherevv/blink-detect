import math
import time

import cv2 as cv
import mediapipe as mp
import numpy as np
from playsound import playsound

# Input an existing mp3 filename
mp3File = "alarm.mp3"

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

WIDTH_CAM, HEIGHT_CAM = 640, 480

LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# constants
CLOSED_EYES_FRAME = 1

BLINK_RATIO_THRESHOLD = 3.5

BLINK_PER_MINUTE_THRESHOLD = 24

PLAYSOUND_DEBOUNCE = 4  # seconds

PLAYSOUND_BLINK_TIME = 10  # seconds


def get_landmarks_mesh(img, results, draw=False):
    img_height, img_width = img.shape[:2]
    # list[(x,y), (x,y)....]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in
                  results.multi_face_landmarks[0].landmark]
    if draw:
        [cv.circle(img, p, 2, (0, 255, 0), -1) for p in mesh_coord]

    # returning the list of tuples for each landmarks
    return mesh_coord


def eye_ratio(landmarks, indices):
    # horizontal line
    right = landmarks[indices[0]]
    left = landmarks[indices[8]]
    # vertical line
    top = landmarks[indices[12]]
    bottom = landmarks[indices[4]]

    h_distance = distance(right, left)
    v_distance = distance(top, bottom)
    if v_distance != 0:
        ratio = h_distance / v_distance
    else:
        ratio = 100
    return ratio


def blink_ratio(landmarks, right_indices, left_indices):
    r_ratio = eye_ratio(landmarks, right_indices)
    l_ratio = eye_ratio(landmarks, left_indices)

    ratio = (r_ratio + l_ratio) / 2
    return ratio


def distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def eye_extract(img, eye_coords):
    # converting color image to  scale image
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # getting the dimension of image
    dim = gray.shape

    # creating mask from gray scale dim
    mask = np.zeros(dim, dtype=np.uint8)

    # drawing Eyes Shape on mask with white color
    cv.fillPoly(mask, [np.array(eye_coords, dtype=np.int32)], 255)

    # showing the mask
    # cv.imshow('mask', mask)

    # draw eyes image on mask, where white shape is
    eyes = cv.bitwise_and(gray, gray, mask=mask)
    # change black color to gray other than eys
    # cv.imshow('eyes draw', eyes)
    eyes[mask == 0] = 155

    # getting minium and maximum x and y
    top = (max(eye_coords, key=lambda item: item[0]))[0]
    bottom = (min(eye_coords, key=lambda item: item[0]))[0]
    right = (max(eye_coords, key=lambda item: item[1]))[1]
    left = (min(eye_coords, key=lambda item: item[1]))[1]

    # croping the eyes from mask
    cropped = eyes[left: right, bottom: top]

    # returning the cropped eye
    return cropped


def draw_text(img, text,
              font=cv.FONT_HERSHEY_PLAIN,
              pos=(0, 0),
              font_scale=2,
              font_thickness=2,
              text_color=(0, 255, 0),
              text_color_bg=(0, 0, 0)
              ):
    x, y = pos
    text_size, _ = cv.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv.rectangle(img, pos, (x + text_w + 20, y + text_h + 20), text_color_bg, -1)
    cv.putText(img, text, (x + 10, y + text_h + font_scale - 1 + 10), font, font_scale, text_color, font_thickness)

    return text_size


def get_iris_points(landmark_point):
    left_eye_points = [
        landmark_point[468],
        landmark_point[469],
        landmark_point[470],
        landmark_point[471],
        landmark_point[472],
    ]
    right_eye_points = [
        landmark_point[473],
        landmark_point[474],
        landmark_point[475],
        landmark_point[476],
        landmark_point[477],
    ]

    left_eye_info = calc_min_enc_losingCircle(left_eye_points)
    right_eye_info = calc_min_enc_losingCircle(right_eye_points)

    return left_eye_info, right_eye_info


def calc_min_enc_losingCircle(landmark_list):
    center, radius = cv.minEnclosingCircle(np.array(landmark_list))
    center = (int(center[0]), int(center[1]))
    radius = int(radius)

    return center, radius


def blink_detect():
    total_blinks = 0
    cef_counter = 0
    prev_time = 0
    blink_times = []
    cap = cv.VideoCapture(0)
    cap.set(3, WIDTH_CAM)
    cap.set(4, HEIGHT_CAM)
    start_time = time.time()
    playsound_time = 0
    frame_counter = 0  # frame counter
    with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5,
                               refine_landmarks=True) as face_mesh:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue
            frame_counter += 1
            # Flip the image horizontally for a selfie-view display.
            image = cv.flip(image, 1)

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            results = face_mesh.process(image)

            # Draw the face detection annotations on the image.
            # image.flags.writeable = True
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
            if results.multi_face_landmarks:
                mesh_coords = get_landmarks_mesh(image, results, False)
                ratio = blink_ratio(mesh_coords, RIGHT_EYE, LEFT_EYE)
                # cv.putText(image, f'ratio {ratio}', (300, 20), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 2)

                if ratio > BLINK_RATIO_THRESHOLD:
                    cef_counter += 1
                    # cv.putText(image, 'Blink', (300, 50), cv.FONT_HERSHEY_PLAIN, 1.3, (0, 255, 0), 2)
                else:
                    if cef_counter > CLOSED_EYES_FRAME:
                        total_blinks += 1
                        cef_counter = 0
                        blink_times.append(time.time())
                        print(total_blinks)
                draw_text(image, f'Blinks:{total_blinks}', pos=(20, 60))

                cv.polylines(image, [np.array([mesh_coords[p] for p in LEFT_EYE], dtype=np.int32)], True, (0, 0, 255),
                             1, cv.LINE_AA)
                cv.polylines(image, [np.array([mesh_coords[p] for p in RIGHT_EYE], dtype=np.int32)], True, (0, 0, 255),
                             1, cv.LINE_AA)

                left_eye, right_eye = get_iris_points(mesh_coords)
                cv.circle(image, left_eye[0], left_eye[1], (0, 255, 0), 2)
                cv.circle(image, right_eye[0], right_eye[1], (0, 255, 0), 2)

            cur_time = time.time()
            fps = 1 / (cur_time - prev_time)
            prev_time = cur_time
            draw_text(image, f'FPS:{str(int(fps))}', pos=(20, 10))

            diff_time = cur_time - start_time
            if diff_time < 60:  # if start app just now
                bpm = 60 / diff_time * total_blinks
                bpm = len(blink_times)
            else:
                # Count blinks in last minute
                blink_times[:] = [t for t in blink_times if cur_time - t < 60]
                bpm = len(blink_times)

            # Alarm when rarely blink
            if bpm < BLINK_PER_MINUTE_THRESHOLD:
                draw_text(image, f'blink more often', pos=(20, HEIGHT_CAM - 60))
                # Play sound
                # if diff_time > 60 and cur_time - playsound_time > PLAYSOUND_DEBOUNCE:
                #     playsound(mp3File, False)
                #     playsound_time = cur_time
            if blink_times \
                    and cur_time - blink_times[-1] > PLAYSOUND_BLINK_TIME \
                    and cur_time - playsound_time > PLAYSOUND_DEBOUNCE:
                playsound(mp3File, False)
                playsound_time = cur_time

            draw_text(image, f'BPM:{str(int(bpm))}', pos=(20, 110))
            draw_text(image, f'{str(int(diff_time))}', pos=(20, 160))

            cv.imshow('MediaPipe Blink Detection', image)

            if cv.waitKey(5) & 0xFF == ord('q'):
                break
    # Finally release back the camera resources
    cv.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    blink_detect()
