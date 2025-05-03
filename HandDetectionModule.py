import cv2 as cv
import mediapipe as mp

def distance(point1, point2):
    (x1, y1), (x2, y2) = point1, point2
    return ((x1-x2)**2 + (y1-y2)**2)**0.5


class HandDetector:

    def __init__(self,
                 static_image_mode=False,
                 max_num_hands=2,
                 model_complexity=1,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):

        self.mpdraw = mp.solutions.drawing_utils
        self.mphands = mp.solutions.hands

        stImgMode = static_image_mode
        noHands = max_num_hands
        model_complexity = model_complexity
        minDetectionCon = min_detection_confidence
        minTrackingCon = min_tracking_confidence
        self.lmCoordinates = []
        self.hands = self.mphands.Hands(stImgMode, noHands,
                                   model_complexity, minDetectionCon, minTrackingCon)

    def landmarks(self, img, draw=False):
        lst = {"Left": [], "Right": []}  # Initialize the dictionary for left and right hands
        rgbImg = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        result = self.hands.process(rgbImg)
        lmarks = result.multi_hand_landmarks
        handedness = result.multi_handedness  # Get handedness information

        if lmarks and handedness:
            for i, lmark in enumerate(lmarks):
                hand_key = handedness[i].classification[0].label  # "Left" or "Right"
                hand_landmarks = []

                for id, lm in enumerate(lmark.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    hand_landmarks.append((cx, cy))  # Append each landmark (x, y)

                lst[hand_key] = hand_landmarks  # Store landmarks in the respective key

                if draw:
                    self.mpdraw.draw_landmarks(img, lmark, self.mphands.HAND_CONNECTIONS)

        return lst

    def fingersUp(self, landmarks):
        """
        Determines which fingers are up based on landmark positions.

        Parameters:
        landmarks (list): A list of (x, y) coordinates for the 21 hand landmarks.

        Returns:
        list: A list of 5 binary values (1 for up, 0 for down) representing the thumb and four fingers.
        """
        fingers = [0] * 5  # Initialize fingers list with all 0s

        if not landmarks or len(landmarks) < 21:
            return fingers  # Return default if no landmarks

        # Thumb: Compare tip (landmark 4) with knuckle (landmark 2)
        if landmarks[4][0] < landmarks[2][0]:  # Left hand
            fingers[0] = 1 if landmarks[4][0] < landmarks[3][0] else 0
        else:  # Right hand
            fingers[0] = 1 if landmarks[4][0] > landmarks[3][0] else 0

        # Other fingers: Compare tip with PIP joint
        for i, (tip, pip) in enumerate([(8, 6), (12, 10), (16, 14), (20, 18)]):
            fingers[i + 1] = 1 if landmarks[tip][1] < landmarks[pip][1] else 0  # Tip is above PIP

        return fingers


def beutifyAll( img, lms, radius=3, color=(0, 255, 0), thickness=-1):
    for point in lms:
        cv.circle(img, point, radius=radius, color=color, thickness=thickness)


if __name__ == "__main__":
    hands = HandDetector()

    video = cv.VideoCapture(0)

    while True:
        isTrue, frame = video.read()
        lm = hands.landmarks(frame, True)
        if len(lm) > 0:
            cv.circle(frame, lm[4], radius=10, color=(0, 255, 0), thickness=-1)
            cv.circle(frame, lm[8], radius=10, color=(0, 255, 0), thickness=-1)
        # beutifyAll(frame, lm)


        cv.imshow("Camera", frame)
        cv.waitKey(1)


    video.release()
    cv.destroyAllWindows()
