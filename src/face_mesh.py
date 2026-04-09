import mediapipe as mp

class FaceMeshDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            refine_landmarks=True,
            max_num_faces=1
        )

    def get_landmarks(self, rgb_frame):
        results = self.face_mesh.process(rgb_frame)
        return results.multi_face_landmarks