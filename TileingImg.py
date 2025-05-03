import cv2 as cv
import numpy as np
import HandDetectionModule as Hdm
import FaceDetectionModule as Fdm

class Tile:
    def __init__(self, width, height, positionx=0, positiony=0):
        self.width, self.height = width, height
        self.px, self.py = positionx, positiony

    def contains_landmark(self, landmark, expansion_factor=5):
        """Check if the landmark is inside this tile, with expanded area."""
        expanded_px = self.px - int(self.width * (expansion_factor - 1) / 2)
        expanded_py = self.py - int(self.height * (expansion_factor - 1) / 2)
        expanded_width = int(self.width * expansion_factor)
        expanded_height = int(self.height * expansion_factor)

        return expanded_px <= landmark[0] < expanded_px + expanded_width and \
               expanded_py <= landmark[1] < expanded_py + expanded_height


if __name__ == '__main__':
    hands = Hdm.HandDetector()
    faces = Fdm.FaceDetector()
    video = cv.VideoCapture(0)

    _, frame = video.read()
    shape = frame.shape
    canvas = np.zeros_like(frame)
    tw, th = shape[1] // 20, shape[0] // 15  # Tile width and height

    while True:
        _, frame = video.read()
        canvas[:] = 0  # Reset canvas every frame

        # Get hand landmarks
        lm1 = hands.landmarks(frame, False)
        left_hand = lm1["Left"]  # List of (x, y) landmarks
        right_hand = lm1["Right"]  # List of (x, y) landmarks

        lm2 = faces.landmarks(frame, False)
        face = [] if not lm2 else lm2[0]
        # if face:
        #     cv.circle(frame, face[468], 2, (0,0,255), -1)
        #     cv.circle(frame, face[473], 2, (0,0,255), -1)
        #     cv.circle(frame, face[151], 7, (0,0,255), -1)
        # Create tiles
        tiles = []
        # -------------------------------------
        # in stright sequencical order :
        for xcordinate in range(0, shape[1], tw):
            for ycordinate in range(0, shape[0], th):
                tiles.append(Tile(tw, th, xcordinate, ycordinate))
        # ----------------------------------------
        # Create tiles in reversed order (bottom to top)
        # for xcordinate in range(0, shape[1], tw):  # X remains same
        #     for ycordinate in range(shape[0] - th, -1, -th):  # Reverse Y order
        #         tiles.append(Tile(tw, th, xcordinate, ycordinate))
        # ----------------------------------------
        # for xcordinate in range(shape[1], 0, tw):  # X remains same
        #     for ycordinate in range(shape[0] - th, -1, -th):  # Reverse Y order
        #         tiles.append(Tile(tw, th, xcordinate, ycordinate))



        # Expansion factor for visibility (1.5x tile size)
        expansion_factor = 3

        # Show only tiles that contain landmarks or are nearby
        for tile in tiles:
            if any(tile.contains_landmark(lm, expansion_factor) for lm in left_hand) or \
               any(tile.contains_landmark(lm, expansion_factor) for lm in right_hand) or \
               any(tile.contains_landmark(lm, expansion_factor) for lm in face):

                # Copy tile from frame to canvas
                canvas[tile.py:tile.py + tile.height, tile.px:tile.px + tile.width] = \
                    frame[tile.py:tile.py + tile.height, tile.px:tile.px + tile.width]

                cv.rectangle(canvas, (tile.px, tile.py), (tile.px + tile.width, tile.py + tile.height), (0, 0, 0), 2)


        cv.imshow("video", frame)
        cv.imshow("canvas", canvas)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv.destroyAllWindows()
