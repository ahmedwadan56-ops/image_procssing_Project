import cv2
import numpy as np

def apply_filter(frame, mode):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if mode == 'grayscale':
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif mode == 'gaussian':
        return cv2.GaussianBlur(frame, (15, 15), 0)
    elif mode == 'median':
        return cv2.medianBlur(frame, 15)
    elif mode == 'sobel_x':
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        return cv2.cvtColor(np.uint8(np.absolute(sobelx)), cv2.COLOR_GRAY2BGR)
    elif mode == 'sobel_y':
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        return cv2.cvtColor(np.uint8(np.absolute(sobely)), cv2.COLOR_GRAY2BGR)
    elif mode == 'canny':
        edges = cv2.Canny(gray, 100, 200)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    elif mode == 'min_filter':
        kernel = np.ones((5, 5), np.uint8)
        return cv2.erode(frame, kernel)
    elif mode == 'max_filter':
        kernel = np.ones((5, 5), np.uint8)
        return cv2.dilate(frame, kernel)
    else:
        return frame

def main():
    cap = cv2.VideoCapture(0)
    mode = 'original'

    print("Real-time OpenCV Video Filter App")
    print("Keys: 'g': Grayscale, 'b': Gaussian Blur, 'm': Median Filter, 'x': Sobel X, 'y': Sobel Y, 'c': Canny, 'n': Min Filter, 'x': Max Filter, 'r': Reset, 'q': Quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        filtered_frame = apply_filter(frame, mode)
        
        cv2.putText(filtered_frame, f"Mode: {mode}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Video Processing - Ahmed Wadan", filtered_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('g'): mode = 'grayscale'
        elif key == ord('b'): mode = 'gaussian'
        elif key == ord('m'): mode = 'median'
        elif key == ord('x'): mode = 'sobel_x'
        elif key == ord('y'): mode = 'sobel_y'
        elif key == ord('c'): mode = 'canny'
        elif key == ord('n'): mode = 'min_filter'
        elif key == ord('p'): mode = 'max_filter'
        elif key == ord('r'): mode = 'original'

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
