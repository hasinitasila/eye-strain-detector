import numpy as np

# Eye landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def get_eye_points(landmarks, eye_indices, frame_width, frame_height):
    points = []
    for idx in eye_indices:
        x = int(landmarks[idx].x * frame_width)
        y = int(landmarks[idx].y * frame_height)
        points.append((x, y))
    return points


def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def compute_ear(eye_points):
    # vertical distances
    v1 = euclidean_distance(eye_points[1], eye_points[5])
    v2 = euclidean_distance(eye_points[2], eye_points[4])

    # horizontal distance
    h = euclidean_distance(eye_points[0], eye_points[3])

    ear = (v1 + v2) / (2.0 * h)
    return ear