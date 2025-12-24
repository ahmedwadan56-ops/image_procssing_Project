import numpy as np              # Import NumPy for array operations
import cv2                      # Import OpenCV for image/video processing

# -----------------------------
# Video Capture
# -----------------------------
vc = cv2.VideoCapture(0)        # Open webcam (device 0)

# -----------------------------
# Default settings
# -----------------------------
mode = 'o'                       # Current filter mode ('o' = original)
kernel_size = 3                  # Kernel/window size for filters
sobel_state = 0                  # Sobel state (0=X, 1=Y, 2=Magnitude)

# -----------------------------
# Filtering Functions
# -----------------------------
def arithmetic_mean(img, k):
    return cv2.blur(img, (k, k))             # Arithmetic mean filter (average)

def geometric_mean(img, k):
    img = img.astype(np.float32) + 1e-6      # Convert to float to avoid log(0)
    log_img = np.log(img)                     # Take logarithm of pixels
    mean = cv2.blur(log_img, (k, k))         # Average the log values
    return np.exp(mean).astype(np.uint8)     # Exponentiate and convert back to uint8

def harmonic_mean(img, k):
    img = img.astype(np.float32) + 1e-6
    inv = 1.0 / img                           # Inverse of each pixel
    mean = cv2.blur(inv, (k, k))             # Average inverses
    return (1.0 / mean).astype(np.uint8)     # Reciprocal to get result

def contraharmonic_mean(img, k, Q=1.5):
    img = img.astype(np.float32) + 1e-6
    num = cv2.blur(img**(Q+1), (k, k))       # Numerator
    den = cv2.blur(img**Q, (k, k))           # Denominator
    return (num / den).astype(np.uint8)      # Compute final value

def midpoint_filter(img, k):
    max_f = cv2.dilate(img, np.ones((k,k)))  # Maximum in the window
    min_f = cv2.erode(img, np.ones((k,k)))   # Minimum in the window
    return ((max_f + min_f)/2).astype(np.uint8)  # Average of max & min

def alpha_trimmed_mean_gray(img_gray, k, d=2):
    small = cv2.resize(img_gray, (0,0), fx=0.5, fy=0.5)  # Downsample
    out = np.zeros_like(small)                            # Output small image
    pad = k//2
    padded = np.pad(small, ((pad,pad),(pad,pad)), mode='edge')  # Pad edges

    for i in range(small.shape[0]):                       # Loop rows
        for j in range(small.shape[1]):                   # Loop columns
            window = padded[i:i+k, j:j+k].flatten()      # Flatten window
            window.sort()                                # Sort pixels
            trimmed = window[d:-d] if len(window)>2*d else window  # Remove extremes
            out[i,j] = np.mean(trimmed)                 # Average remaining pixels

    J = cv2.resize(out, (img_gray.shape[1], img_gray.shape[0]))  # Upsample
    return J.astype(np.uint8)                             # Return as uint8

# -----------------------------
# Get filter name for display
# -----------------------------
def get_mode_name(mode, sobel_state, kernel_size):
    names = {
        'o': "Original", 'g': "Gray", 'b': f"Blur (kernel={kernel_size})",
        'c': "Canny", 'a': "Arithmetic Mean", 'e': "Geometric Mean",
        'h': "Harmonic Mean", 'k': "Contraharmonic Mean", 'm': "Median Filter",
        'x': "Max Filter", 'n': "Min Filter", 'd': "Midpoint Filter",
        't': "Alpha Trimmed Mean", 'l': "Low Pass (Smoothing)", 'p': "High Pass (Sharpening)"
    }
    if mode=='s':
        if sobel_state==0: return "Sobel X"                 # Sobel X
        if sobel_state==1: return "Sobel Y"                 # Sobel Y
        return "Sobel Magnitude"                             # Sobel magnitude
    return names.get(mode,"")                                # Return filter name

# -----------------------------
# Main loop
# -----------------------------
while True:
    ret, I = vc.read()                                     # Read frame from webcam
    if not ret:                                            # If failed, break
        break

    if mode=='o':
        J = I.copy()                                       # Show original
    elif mode=='g':
        J = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)           # Convert to gray
        J = cv2.cvtColor(J, cv2.COLOR_GRAY2BGR)           # Back to BGR for display
    else:
        gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)        # Gray for edge filters

        if mode=='b':
            J = cv2.GaussianBlur(I,(kernel_size,kernel_size),0)  # Gaussian blur
        elif mode=='c':
            blurred = cv2.GaussianBlur(gray,(5,5),0)             # Blur first
            sobelx = cv2.Sobel(blurred,cv2.CV_64F,1,0,ksize=3)   # Sobel X
            sobely = cv2.Sobel(blurred,cv2.CV_64F,0,1,ksize=3)   # Sobel Y
            magnitude = np.sqrt(sobelx**2 + sobely**2)            # Gradient magnitude
            magnitude = (magnitude/magnitude.max()*255).astype(np.uint8) # Normalize
            J = np.where(magnitude>50,255,0).astype(np.uint8)     # Threshold
            J = cv2.cvtColor(J, cv2.COLOR_GRAY2BGR)               # Back to BGR
        elif mode=='s':
            sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
            sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
            if sobel_state==0: J = np.abs(sobelx)                # X
            elif sobel_state==1: J = np.abs(sobely)              # Y
            else: J = cv2.magnitude(sobelx,sobely)              # Magnitude
            J = (J/J.max()*255).astype(np.uint8)
            J = cv2.cvtColor(J, cv2.COLOR_GRAY2BGR)
        elif mode=='a': J = arithmetic_mean(I,kernel_size)
        elif mode=='e': J = geometric_mean(I,kernel_size)
        elif mode=='h': J = harmonic_mean(I,kernel_size)
        elif mode=='k': J = contraharmonic_mean(I,kernel_size)
        elif mode=='m': J = cv2.medianBlur(I,kernel_size)
        elif mode=='x': J = cv2.dilate(I,np.ones((kernel_size,kernel_size)))
        elif mode=='n': J = cv2.erode(I,np.ones((kernel_size,kernel_size)))
        elif mode=='d': J = midpoint_filter(I,kernel_size)
        elif mode=='t':
            J_gray = alpha_trimmed_mean_gray(gray,kernel_size) # Apply Alpha Trimmed Mean on gray
            J = cv2.cvtColor(J_gray, cv2.COLOR_GRAY2BGR)       # Back to BGR
        elif mode=='l': J = cv2.GaussianBlur(I,(kernel_size,kernel_size),0)
        elif mode=='p':
            blur = cv2.GaussianBlur(I,(kernel_size,kernel_size),0)
            J = cv2.addWeighted(I,1.5,blur,-0.5,0)             # High pass filter

    # Overlay filter name
    text = get_mode_name(mode,sobel_state,kernel_size)
    cv2.putText(J,text,(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)

    cv2.imshow("My stream", J)                              # Display frame

    key = chr(cv2.waitKey(1)&0xFF)                          # Read keyboard input
    if key in ['o','g','b','c','s','a','e','h','k','m','x','n','d','t','l','p','q']:
        mode = key                                          # Change filter mode
    if key=='s':
        sobel_state = (sobel_state+1)%3                     # Cycle Sobel state
    if key=='+':
        kernel_size += 2                                    # Increase kernel size
        print("Kernel size:",kernel_size)
    if key=='-':
        kernel_size = max(3,kernel_size-2)                 # Decrease kernel size
        print("Kernel size:",kernel_size)
    if key=='q':
        break                                               # Quit loop

vc.release()                                               # Release webcam
cv2.destroyAllWindows()                                   # Close all windows
