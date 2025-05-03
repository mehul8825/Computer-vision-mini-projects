import cv2 as cv
import numpy as np
import HandDetectionModule as hdm

colors = [
        [(0,0,0),"Eraser"],# eraser
        [(0, 0, 255),"Red"],# red
        [(255, 0, 0),"Blue"], # blue
        [(0, 255, 255),"Yellow"], # Yellow
        [(0, 255, 0),"Green"] # green
        ]

def Highlight(frame, coordinate):
    x, y = coordinate
    cv.rectangle(frame, (x-1, y-1), (x + 70 + 1, y + 50 + 1), (255,255,255), thickness=5)

def draw(canvas, frame, position, color):
    cv.circle(canvas, position, radius=thickness, color=color, thickness=-1)
    # cv.circle(frame, position, radius=2, color=color, thickness=-1)  # Indicator

# navigator graphics
def Graphics(frame, highlight):

    x, y = 560, 10

    for color in enumerate(colors):
        cv.rectangle(frame, (x, y), (x + 70, y + 50), color[1][0], thickness=cv.FILLED)
        cv.putText(frame, color[1][1], (x + 5, y + 35), cv.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), thickness=1)
        if color[0] == highlight:
            Highlight(frame, (x, y))
        y += 70



if __name__ == "__main__":
    detector = hdm.HandDetector()
    video = cv.VideoCapture(0)

    isTrue, frame = video.read()

    # Drawing Canvas
    canvas = np.zeros(frame.shape, dtype=np.uint8)  # Canvas for drawing

    #by default thickness:
    thickness=4
    current_color = 2
    while True:
        isTrue, frame = video.read()
        if not isTrue:
            break

        frame = cv.flip(frame, 1)  # Flip the frame for a mirror view
        lm = detector.landmarks(frame, False)

        if lm["Left"]:
            thumb_tip = lm["Left"][4]
            index_tip = lm["Left"][8]
            radius = int(hdm.distance(thumb_tip, index_tip) / 2)
            thickness = radius
            center = int((thumb_tip[0] + index_tip[0]) / 2), int((thumb_tip[1] + index_tip[1]) / 2)
            cv.circle(frame, center=center, radius=radius, color=(255, 255, 255), thickness=4)

        if lm["Right"]:
            index_tip = lm["Right"][8]
            middle_tip = lm["Right"][12]
            selcting_distance = int(hdm.distance(index_tip,middle_tip))
            # we are drawing
            if selcting_distance > 45:

                draw(canvas, frame, index_tip, colors[current_color][0])
                # Drawing logic
                gray_canvas = cv.cvtColor(canvas, cv.COLOR_BGR2GRAY)
                _, invert_mask = cv.threshold(gray_canvas, 20, 255, cv.THRESH_BINARY_INV)
                invert_mask = cv.cvtColor(invert_mask, cv.COLOR_GRAY2BGR)
                combined_frame = cv.bitwise_and(frame, invert_mask)
                frame = cv.add(combined_frame, canvas)
            # we are selecting the color by 2 fingers
            else:
                cv.rectangle(frame, index_tip, middle_tip, color=(255, 255, 255), thickness=cv.FILLED)
                xcoordinate, ycoordinate = 540, 20

                xrange = range(xcoordinate, xcoordinate + 70)
                if index_tip[0] in xrange and middle_tip[0] in xrange:
                    cv.rectangle(frame, index_tip, middle_tip, color=(0, 255, 255), thickness=cv.FILLED)
                    for i in enumerate(colors):
                        yrange = range(ycoordinate, ycoordinate+50)
                        if index_tip[1] in yrange and middle_tip[1] in yrange:
                            cv.rectangle(frame, index_tip, middle_tip, color=i[1][0], thickness=cv.FILLED)
                            current_color = int(i[0])
                        ycoordinate+=70


        Graphics(frame, current_color)
        cv.imshow("Paint on Camera", frame)

        # Exit on pressing 'q'
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv.destroyAllWindows()
