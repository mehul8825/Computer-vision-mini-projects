import cv2 as cv
import numpy as np

def distance_matrix(frame1, frame2):
    """Calculate pixel-wise Euclidean distance between two frames."""
    diff = frame1.astype(np.int16) - frame2.astype(np.int16)
    return np.sqrt(np.sum(diff ** 2, axis=2))

def lerp(a, b, t):
    """Linear interpolation."""
    return (1 - t) * a + t * b

if __name__ == "__main__":
    video = cv.VideoCapture(0)

    if not video.isOpened():
        print("Error: Could not open camera.")
        exit()

    # Read the first frame
    isTrue, frame = video.read()
    if not isTrue:
        print("Error: Could not read frame.")
        video.release()
        exit()

    preframe = frame.copy()
    threshold = 50  # Lower threshold for better sensitivity
    smooth_x, smooth_y = None, None  # Smoothed position of the circle

    while True:
        isTrue, frame = video.read()
        if not isTrue:
            print("Error: Failed to grab frame.")
            break

        canvas = np.zeros_like(frame, dtype=np.uint8)  # Initialize the motion detection canvas
        diff_matrix = distance_matrix(preframe, frame)

        # Create a binary mask for motion detection
        mask = diff_matrix > threshold

        # Apply morphological operations to clean up the mask
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))  # Adjustable kernel size
        mask = cv.morphologyEx(mask.astype(np.uint8), cv.MORPH_CLOSE, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

        # Update the canvas
        canvas[mask > 0] = [255, 255, 255]

        # Find the coordinates of white pixels
        white_coords = np.column_stack(np.where(mask > 0))

        if white_coords.size > 0:
            # Calculate the average position of white pixels
            center_y, center_x = np.mean(white_coords, axis=0)

            # Initialize smoothed position if not set
            if smooth_x is None or smooth_y is None:
                smooth_x, smooth_y = center_x, center_y

            # Smoothly transition to the new position
            smooth_x = lerp(smooth_x, center_x, 0.1)
            smooth_y = lerp(smooth_y, center_y, 0.1)

            # Draw a circle at the smoothed center position
            cv.circle(frame, (int(center_x), int(center_y)), 10, (0, 255, 0), cv.FILLED)

        # Apply blur to smooth the canvas for visualization
        canvas_blurred = cv.GaussianBlur(canvas, (15, 15), 0)

        # Display the live camera feed and the smoothed canvas
        cv.imshow("Camera", frame)
        cv.imshow("Motion Detection Canvas", canvas_blurred)

        # Update the previous frame
        preframe = frame.copy()

        # Break the loop on pressing 'q'
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv.destroyAllWindows()
