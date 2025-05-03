import cv2 as cv
import mediapipe as mp

def distance(point1, point2):
    (x1, y1), (x2, y2) = point1, point2
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


class FaceDetector:
    def __init__(self,
                 static_image_mode=False,
                 max_num_faces=2,
                 model_complexity=1,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh

        # ✅ Corrected initialization of FaceMesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=True,  # Optional: Improves landmark accuracy
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def landmarks(self, img, draw=False):
        """Detects face landmarks and returns a list of coordinates."""
        lst = []  # List to store facial landmarks
        rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb_img)

        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                face_points = []
                for lm in face_landmarks.landmark:
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    face_points.append((cx, cy))  # Append each landmark (x, y)

                lst.append(face_points)  # Store landmarks for each detected face

                if draw:
                    self.mp_draw.draw_landmarks(img, face_landmarks, self.mp_face_mesh.FACEMESH_CONTOURS)

        return lst


def beautify_all(img, landmarks_list, radius=2, color=(0, 255, 0), thickness=-1):
    """Draws circles on all detected facial landmarks."""
    for landmarks in landmarks_list:
        for point in landmarks:
            cv.circle(img, point, radius=radius, color=color, thickness=thickness)


if __name__ == "__main__":
    face_detector = FaceDetector()

    video = cv.VideoCapture(0)

    while True:
        is_true, frame = video.read()
        if not is_true:
            break

        faces = face_detector.landmarks(frame, draw=True)

        if len(faces) > 0:
            beautify_all(frame, faces)

        cv.imshow("Face Mesh Detection", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv.destroyAllWindows()
