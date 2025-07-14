import cv2 # For all camera and image processing tasks
import requests # To fetch images from your IP webcam
import numpy as np # For numerical operations, especially with image data
import imutils # A collection of convenience functions for OpenCV (though not heavily used here)
import kociemba # The powerful library that solves the Rubik's Cube
import time # For time-related functions, like delays
import sys # For system-specific parameters and functions, like exiting the script
import tensorflow as tf # The core library for your AI model
import h5py # Used for fixing potential issues with the .h5 model file

# --- Configuration ---
# Set the URL for your IP Webcam. Make sure this is correct for your camera!
url = "http://10.177.30.192:8080/shot.jpg" 

# Define the order in which you'll scan the cube faces (Up, Right, Front, Down, Left, Back)
order=['U','R','F','D','L','B']

# 'vals' will store the detected colors for each face as you scan them.
# Example: {'U': ['W', 'W', 'R', ...]}
vals={}

# --- Model Fix for H5PY (Important for some Teachable Machine models) ---
# This section attempts to fix a common issue where Teachable Machine's Keras models
# might have a "groups" attribute that can cause loading errors in newer TensorFlow versions.
# It safely removes this attribute if it exists.
try:
    with h5py.File("keras_model.h5", mode="r+") as f:
        model_config_string = f.attrs.get("model_config")
        if model_config_string and '"groups": 1,' in model_config_string:
            model_config_string = model_config_string.replace('"groups": 1,', '')
            f.attrs.modify('model_config', model_config_string)
            f.flush()
            print("Successfully patched 'keras_model.h5' to remove 'groups' attribute.")
except Exception as e:
    print(f"Warning: Could not patch 'keras_model.h5'. This might be fine, or cause issues later: {e}")

# 'vals_original' will store the cube state in the format Kociemba expects (UURRFFDDLLBB).
# It's initially empty lists for each face.
vals_original={
    'U':[], 'R':[], 'F':[], 'L':[], 'B':[], 'D':[]
}

# This dictionary maps the color of a **center sticker** to its corresponding
# Kociemba face notation. For example, if 'W' (white) is your Up face's center,
# then Kociemba uses 'U' for that face.
covert_values={
    'W':'U', # If White is the Up face
    'R':'R', # If Red is the Right face
    'G':'F', # If Green is the Front face
    'Y':'D', # If Yellow is the Down face
    'O':'L', # If Orange is the Left face
    'B':'B'  # If Blue is the Back face
}

# --- Teachable Machine AI Model Setup ---
try:
    # Load your pre-trained AI model from Teachable Machine (keras_model.h5)
    color_prediction_model = tf.keras.models.load_model('keras_model.h5')
    print("AI Color Prediction Model loaded successfully!")
except Exception as e:
    print(f"Error loading AI model: {e}")
    print("Make sure 'keras_model.h5' is in the same directory as this script.")
    # If the model fails to load, the script cannot proceed, so we exit.
    sys.exit("AI model failed to load. Exiting.")

try:
    # Load the labels (e.g., 'W', 'O', 'R') that your Teachable Machine model was trained with.
    # This reads from 'labels.txt', typically provided by Teachable Machine.
    with open('labels.txt', 'r') as f:
        # This line expects labels like "0 W", "1 O" and extracts just "W", "O".
        AI_COLOR_LABELS = [line.strip().split(' ', 1)[1] for line in f if line.strip()]
        # If your 'labels.txt' only contains "W", "O", "R" (no numbers), use this instead:
        # AI_COLOR_LABELS = [line.strip() for line in f if line.strip()]
    print(f"Model will predict these classes: {AI_COLOR_LABELS}")
except FileNotFoundError:
    print("Error: labels.txt not found. Make sure it's in the same directory.")
    sys.exit("Labels file not found. Exiting.")
except Exception as e:
    print(f"Error reading labels.txt: {e}")
    sys.exit("Error processing labels file. Exiting.")

# **IMPORTANT:** This is the exact image size your Teachable Machine model expects as input.
# For most Teachable Machine image models, this is 224x224 pixels.
MODEL_INPUT_IMAGE_SIZE = (224, 224) 

# This defines the size of the square "patch" of pixels we extract around each sticker
# from the live camera feed. This patch is then fed to your AI model.
# Adjust this value based on how large the Rubik's Cube stickers appear in your camera view.
# A value of 100x100 pixels is a good starting point.
AI_EXTRACT_PATCH_SIZE = 100 

## AI Color Prediction Function

#This function takes a small image patch (a portion of the camera feed) and uses your loaded AI model to predict what color is in that patch.

def predict_color_with_ai(image_patch_bgr):
    # Resize the extracted image patch to the specific size your AI model expects (e.g., 224x224).
    # INTER_AREA is good for shrinking images; it helps prevent aliasing.
    processed_patch = cv2.resize(image_patch_bgr, MODEL_INPUT_IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    
    # Convert the image from BGR (Blue-Green-Red, OpenCV's default) to RGB (Red-Green-Blue),
    # which is what most AI models (including Teachable Machine's) are trained on.
    processed_patch = cv2.cvtColor(processed_patch, cv2.COLOR_BGR2RGB)

    # Normalize pixel values. AI models usually expect pixel values to be between 0 and 1,
    # rather than 0 and 255.
    processed_patch = processed_patch / 255.0  

    # Add an extra dimension to the image. Your AI model expects input in batches,
    # even if you're only processing one image at a time. So, a (224, 224, 3) image
    # becomes (1, 224, 224, 3).
    processed_patch = np.expand_dims(processed_patch, axis=0)

    # Make a prediction using the AI model. 'verbose=0' keeps the output clean.
    predictions = color_prediction_model.predict(processed_patch, verbose=0)
    
    # Get the index of the class with the highest probability (the model's best guess).
    predicted_class_index = np.argmax(predictions)
    
    # Return the actual color label (e.g., 'W', 'O', 'R') corresponding to the predicted index.
    return AI_COLOR_LABELS[predicted_class_index]
def hsv_mask_detector(h,s,v):
    # This function is for older HSV-based color detection. It's included but currently
    # NOT used for color detection if your AI model loads successfully.
    # It would be used as a fallback if the AI model fails.
    # The HSV ranges here are examples; you'd need to calibrate them for your lighting.
    if (s >= 0 and s <= 50) and (v >= 200 and v <= 255): return "W" # White
    elif ((h >= 0 and h <= 8) or (h >= 170 and h <= 179)) and s >= 100 and s <= 255 and v >= 100 and v <= 255: return "R" # Red
    elif (h >= 9 and h <= 25) and s >= 100 and s <= 255 and v >= 100 and v <= 255: return "O" # Orange
    elif (h >= 25 and h <= 45) and s >= 100 and s <= 255 and v >= 100 and v <= 255: return "Y" # Yellow
    elif (h >= 45 and h <= 80) and s >= 100 and s <= 255 and v >= 100 and v <= 255: return "G" # Green
    elif (h >= 80 and h <= 140) and s >= 100 and s <= 255 and v >= 100 and v <= 255: return "B" # Blue
    return "X" # Unknown color

def triangle(img,p1,p2,p3):
    # Helper function to draw a simple triangle on the image.
    cv2.line(img, p1, p2, (0, 0, 0), 1)
    cv2.line(img, p2, p3, (0, 0, 0), 1)
    cv2.line(img, p1, p3, (0, 0, 0), 1)

def moves(soln,frame, current_key_press):
    # This function is intended to visualize the cube solution steps.
    # It currently contains a detailed set of lines and triangles to show moves,
    # but it needs to be integrated with the 'current_move_index' logic in the main loop
    # to display one move at a time accurately.
    # As it stands, it would try to draw all solution steps at once if called directly.
    for i in soln:
            if i =='R':
                  cv2.line(frame,(515,510),(515,230),(0,0,0),1)
                  triangle(frame,(515,220),(505,230),(525,230))
                  if key==32:
                        continue
            if i =='R2':
                  cv2.line(frame,(515,510),(515,230),(0,0,0),1)
                  triangle(frame,(515,220),(505,230),(525,230))
                  if key==32:
                        cv2.line(frame,(515,510),(515,230),(0,0,0),1)
                        triangle(frame,(515,220),(505,230),(525,230))
                        if key==32:
                              continue
            if i =="R'":
                  cv2.line(frame,(515,510),(515,230),(0,0,0),1)
                  triangle(frame,(515,520),(505,510),(525,510))
                  if key==32:
                        continue
            elif i=='L':
                  cv2.line(frame,(235,230),(235,510),(0,0,0),1)
                  triangle(frame,(235,520),(225,510),(245,510))
                  if key==32:
                        continue
            elif i=='L2':
                  cv2.line(frame,(235,230),(235,510),(0,0,0),1)
                  triangle(frame,(235,520),(225,510),(245,510))
                  if key==32:
                        cv2.line(frame,(235,230),(235,510),(0,0,0),1)
                        triangle(frame,(235,520),(225,510),(245,510))
                        if key==32:
                              continue
            elif i=="L'":
                  cv2.line(frame,(235,230),(235,510),(0,0,0),1)
                  triangle(frame,(235,220),(225,230),(245,230))
                  if key==32:
                        continue
            elif i=='U':
                  cv2.line(frame,(515,230),(238,230),(0,0,0),1)
                  triangle(frame,(245,230),(235,220),(235,240))
                  if key==32:
                        continue
            elif i=='U2':
                  cv2.line(frame,(515,230),(238,230),(0,0,0),1)
                  triangle(frame,(245,230),(235,220),(235,240))
                  if key==32:
                        cv2.line(frame,(515,230),(238,230),(0,0,0),1)
                        triangle(frame,(245,230),(235,220),(235,240))
                        if key==32:
                              continue
            elif i=="U'":
                  cv2.line(frame,(515,230),(238,230),(0,0,0),1)
                  triangle(frame,(525,230),(515,220),(515,240))
                  if key==32:
                        continue
            elif i=='F':
                  cv2.line(frame,(515,230),(238,230),(0,0,0),1)
                  triangle(frame,(245,230),(235,220),(235,240))
                  cv2.line(frame,(515,370),(238,370),(0,0,0),1)
                  triangle(frame,(245,370),(235,360),(235,380))
                  cv2.line(frame,(515,510),(238,510),(0,0,0),1)
                  triangle(frame,(245,510),(235,500),(235,520))
                  if key==32:
                        cv2.line(frame,(515,510),(515,230),(0,0,0),1)
                        triangle(frame,(515,220),(505,230),(525,230))
                        if key==32:
                              continue
            elif i=='F2':
                  cv2.line(frame,(515,230),(238,230),(0,0,0),1)
                  triangle(frame,(245,230),(235,220),(235,240))
                  cv2.line(frame,(515,370),(238,370),(0,0,0),1)
                  triangle(frame,(245,370),(235,360),(235,380))
                  cv2.line(frame,(515,510),(238,510),(0,0,0),1)
                  triangle(frame,(245,510),(235,500),(235,520))
                  if key==32:
                        cv2.line(frame,(515,510),(515,230),(0,0,0),1)
                        triangle(frame,(515,220),(505,230),(525,230))
                        if key==32:
                              cv2.line(frame,(515,510),(515,230),(0,0,0),1)
                              triangle(frame,(515,220),(505,230),(525,230))
                              if key==32:
                                    continue
            elif i=="F'":
                  cv2.line(frame,(515,230),(238,230),(0,0,0),1)
                  triangle(frame,(245,230),(235,220),(235,240))
                  cv2.line(frame,(515,370),(238,370),(0,0,0),1)
                  triangle(frame,(245,370),(235,360),(235,380))
                  cv2.line(frame,(515,510),(238,510),(0,0,0),1)
                  triangle(frame,(245,510),(235,500),(235,520))
                  if key==32:
                        cv2.line(frame,(515,510),(515,230),(0,0,0),1)
                        triangle(frame,(515,520),(505,510),(525,510))
                        if key==32:
                              continue
            elif i=='B':
                  cv2.line(frame,(515,230),(238,230),(0,0,0),1)
                  triangle(frame,(245,230),(235,220),(235,240))
                  cv2.line(frame,(515,370),(238,370),(0,0,0),1)
                  triangle(frame,(245,370),(235,360),(235,380))
                  cv2.line(frame,(515,510),(238,510),(0,0,0),1)
                  triangle(frame,(245,510),(235,500),(235,520))
                  if key==32:
                        cv2.line(frame,(235,230),(235,510),(0,0,0),1)
                        triangle(frame,(235,520),(225,510),(245,510))
                        if key==32:
                              continue
            elif i=='B2':
                  cv2.line(frame,(515,230),(238,230),(0,0,0),1)
                  triangle(frame,(245,230),(235,220),(235,240))
                  cv2.line(frame,(515,370),(238,370),(0,0,0),1)
                  triangle(frame,(245,370),(235,360),(235,380))
                  cv2.line(frame,(515,510),(238,510),(0,0,0),1)
                  triangle(frame,(245,510),(235,500),(235,520))
                  if key==32:
                        cv2.line(frame,(235,230),(235,510),(0,0,0),1)
                        triangle(frame,(235,520),(225,510),(245,510))
                        if key==32:
                              cv2.line(frame,(235,230),(235,510),(0,0,0),1)
                              triangle(frame,(235,520),(225,510),(245,510))
                              if key==32:
                                    continue
            elif i=="B'":
                  cv2.line(frame,(515,230),(238,230),(0,0,0),1)
                  triangle(frame,(245,230),(235,220),(235,240))
                  cv2.line(frame,(515,370),(238,370),(0,0,0),1)
                  triangle(frame,(245,370),(235,360),(235,380))
                  cv2.line(frame,(515,510),(238,510),(0,0,0),1)
                  triangle(frame,(245,510),(235,500),(235,520))
                  if key==32:
                        cv2.line(frame,(235,230),(235,510),(0,0,0),1)
                        triangle(frame,(235,220),(225,230),(245,230))
                        if key==32:
                              continue
            elif i=='D':
                  cv2.line(frame,(235,510),(515,510),(0,0,0),1)
                  triangle(frame,(525,510),(515,500),(515,520))
                  if key==32:
                        continue
            elif i=='D2':
                  cv2.line(frame,(235,510),(515,510),(0,0,0),1)
                  triangle(frame,(525,510),(515,500),(515,520))
                  if key==32:
                        cv2.line(frame,(235,510),(515,510),(0,0,0),1)
                        triangle(frame,(525,510),(515,500),(515,520))
                        if key==32:
                              continue
            elif i=="D'":
                  cv2.line(frame,(235,510),(515,510),(0,0,0),1)
                  triangle(frame,(225,510),(235,500),(235,520))
                  if key==32:
                        continue

def conversion_of_symbols(dictionary_of_values):
    # This function converts the detected colors (like 'W', 'O', 'R') into the
    # specific 54-character string format that the Kociemba solver expects.
    # It essentially organizes the scanned face data into the UURRFFDDLLBB order.
    for key in vals_original:
        vals_original[key] = [] # Clear previous values for a fresh scan

    for face_key_name, detected_color_list in dictionary_of_values.items():
        for detected_color_symbol in detected_color_list:
            if detected_color_symbol in covert_values:
                # If the AI predicts a known cube color, use it directly.
                vals_original[face_key_name].append(detected_color_symbol)
            elif detected_color_symbol == 'Background' or detected_color_symbol == 'X':
                # If the AI detects 'Background' or 'X' (unknown/unclassified),
                # print a warning and use 'X' as a placeholder. Kociemba will fail if 'X' is present.
                print(f"Warning: Detected '{detected_color_symbol}' for face '{face_key_name}'. Kociemba will likely fail if this isn't corrected.")
                vals_original[face_key_name].append('X')
            else:
                # Catch any other unexpected labels from your AI model.
                print(f"Warning: Unexpected label '{detected_color_symbol}' from AI model for face '{face_key_name}'. Mapping to 'X'.")
                vals_original[face_key_name].append('X')

                
# Initialize variables for the solution display
solution = None # Stores the calculated solution steps
solution_displayed = False # Flag to know if a solution is currently being shown
current_move_index = 0 # Tracks which step of the solution is currently displayed

while True:
    # --- Camera Feed Acquisition ---
    try:
        # Request an image from your IP webcam.
        img_resp = requests.get(url, timeout=5)
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to {url}. Retrying...")
        time.sleep(2) # Wait a bit before retrying
        continue # Skip to the next loop iteration
    except requests.exceptions.Timeout:
        print(f"Error: Request to {url} timed out. Retrying...")
        time.sleep(1)
        continue
    except Exception as e:
        print(f"An unexpected error occurred during request: {e}. Retrying...")
        time.sleep(1)
        continue

    # Check if the image was successfully retrieved (status code 200 means OK)
    if img_resp.status_code != 200:
        print(f"Error: Received status code {img_resp.status_code} from {url}. Retrying...")
        time.sleep(2)
        continue

    # Convert the raw image data into a NumPy array, then decode it into an OpenCV image.
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    frame = cv2.imdecode(img_arr, -1)

    # Check if the image decoding was successful
    if frame is None:
        print("Error: Could not decode image. 'frame' is None. Retrying...")
        time.sleep(1)
        continue

    # Resize the live camera feed for better display and processing.
    frame = cv2.resize(frame,(750,640))
    height,width=frame.shape[:2] # Get current frame dimensions
    centre_x,centre_y=width//2,height//2 # Calculate the center of the frame
    
    # Convert the frame to HSV color space. While not directly used by the AI model,
    # it might be useful for debugging or future HSV-based features.
    hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 

    # --- Sticker Detection Grid Setup ---
    grid_size=3 # We're looking for 3x3 stickers per face
    spacing=140 # Distance between the centers of each sticker
    dot_radius=5 # Radius for the green dots marking sticker centers
    color_order=[] # List to store the detected colors for the current face

    # Loop through a 3x3 grid to find and classify each sticker
    for i in range(grid_size):
        for j in range(grid_size):
            # Calculate the (x, y) coordinates for the center of each sticker.
            # The '+ 50' offsets the grid slightly downwards.
            x = centre_x + (j - 1) * spacing
            y = centre_y + (i - 1) * spacing + 50
            
            # --- Extracting the Sticker Patch for AI ---
            # Calculate the top-left (y_start, x_start) and bottom-right (y_end, x_end)
            # coordinates of the square patch around the sticker's center.
            # `max(0, ...)` ensures coordinates don't go off the top/left edge.
            y_start = max(0, y - AI_EXTRACT_PATCH_SIZE // 2)
            x_start = max(0, x - AI_EXTRACT_PATCH_SIZE // 2)
            
            # `min(frame.shape[0], ...)` ensures coordinates don't go off the bottom/right edge.
            y_end = min(frame.shape[0], y_start + AI_EXTRACT_PATCH_SIZE) 
            x_end = min(frame.shape[1], x_start + AI_EXTRACT_PATCH_SIZE)

            # Extract the actual image patch from the frame using NumPy slicing.
            sticker_patch_bgr = frame[y_start:y_end, x_start:x_end]

            # --- Check if the Extracted Patch is Valid ---
            # If the patch is empty (e.g., if the calculated coordinates were entirely off-screen)
            # or has zero width/height, it's invalid.
            if sticker_patch_bgr.size == 0 or sticker_patch_bgr.shape[0] == 0 or sticker_patch_bgr.shape[1] == 0:
                color = "X" # Assign 'X' (unknown) as the color
                print(f"Warning: Patch at ({x},{y}) is empty or has zero dimensions. Assigning 'X'.")
            else:
                # If the patch is valid, send it to your AI model for color prediction.
                color = predict_color_with_ai(sticker_patch_bgr) 
            
            # Add the detected color to our list for the current face.
            color_order.append(color)
            
            # --- Visual Feedback on Screen ---
            # Draw a green dot at the center of each detected sticker position.
            cv2.circle(frame,(x,y),dot_radius,(0,255,0),-1)
            # Display the predicted color text next to the dot.
            cv2.putText(frame,color,(x-10,y-10),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)

    # Show the processed camera feed with dots and detected colors.
    cv2.imshow("yo",frame)
    
    # Capture a key press (waits 1ms).
    key=cv2.waitKey(1) & 0xFF

    # --- User Input Handling ---
    if key==27: # If the ESC key (ASCII 27) is pressed, exit the loop.
        break
    elif chr(key).upper() in order: # If a valid face key (U, R, F, D, L, B) is pressed:
        face_key=chr(key).upper()
        
        # Basic check to ensure 9 colors were detected for the face.
        if len(color_order) != 9:
            print(f"Error: Scanned face {face_key} has {len(color_order)} colors, expected 9. Rescan.")
            continue # Ask user to rescan the face.

        # Store the detected colors for this face.
        vals[face_key]=color_order.copy()
        print(f"\nCaptured face {face_key}:")
        # Print the 3x3 grid of detected colors for user verification.
        for i in range(0,9,3):
            print(color_order[i],color_order[i+1],color_order[i+2])
        
        # If all 6 faces have been scanned:
        if len(vals) == 6:
            print("\nAll 6 faces scanned! Attempting to solve...")
            # Convert the 'vals' dictionary into the Kociemba-compatible string.
            conversion_of_symbols(vals)

            order_original="" # This will be the 54-character Kociemba input string.
            all_faces_valid = True
            # Build the Kociemba string and check for any 'X' (unknown) colors.
            for face_char in order:
                face_val_list = vals_original.get(face_char)
                if face_val_list is None or len(face_val_list) != 9:
                    all_faces_valid = False
                    print(f"Error: Face '{face_char}' not scanned or incomplete ({len(face_val_list) if face_val_list else 0}/9 colors).")
                    break
                
                if "X" in face_val_list:
                    all_faces_valid = False
                    print(f"Error: Face '{face_char}' contains unclassified 'X' colors. Please improve detection or rescan.")
                    break
                
                for j in face_val_list:
                    order_original+=j
            
            # If all faces are valid and we have 54 colors, try to solve.
            if all_faces_valid and len(order_original) == 54:
                print(f"Scanned Cube State for Kociemba: {order_original}")
                try:
                    # Use Kociemba to solve the cube!
                    solution=kociemba.solve(order_original)
                    solution=solution.split(" ") # Split the solution string into individual moves.
                    print(f"Solution: {' '.join(solution)}")
                    solution_displayed = True
                    current_move_index = 0 # Reset move index for new solution.
                except Exception as e:
                    # If Kociemba fails, it means the input cube state is likely invalid.
                    print(f"Error solving cube with Kociemba: {e}")
                    print("This usually means the scanned state is invalid (e.g., wrong number of colors, impossible configuration).")
                    print("Please rescan the cube carefully or debug color detection.")
                    # Clear scanned data to allow rescanning.
                    vals.clear()
                    for key in vals_original:
                        vals_original[key] = []
                    solution = None
                    solution_displayed = False
            else:
                print("Cannot solve: Not all faces scanned correctly or contain unknown/background colors.")
        else:
            print(f"Scanned {len(vals)}/{len(order)} faces. Keep going!")
            
    # --- Displaying the Solution ---
    if solution_displayed and solution:
        if current_move_index < len(solution):
            # Display the current move from the solution.
            current_move = solution[current_move_index]
            cv2.putText(frame, f"Move: {current_move}", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            if key == 32: # If Spacebar is pressed, advance to the next move.
                current_move_index += 1
                time.sleep(0.2) # Small delay to prevent multiple advances on one press.
        else:
            # Once all moves are displayed.
            cv2.putText(frame, "SOLUTION COMPLETE!", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, "Press ESC to exit.", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
            if key == ord('r'): # Press 'R' to reset and start a new scan.
                vals.clear()
                for key_vo in vals_original: vals_original[key_vo] = []
                solution = None
                solution_displayed = False
                current_move_index = 0
                print("\nResetting for new scan.")

# --- Cleanup ---
cv2.destroyAllWindows() # Close all OpenCV windows when the loop ends.
print("Application closed.")
