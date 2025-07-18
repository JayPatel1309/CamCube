# CamCube
An AI-powered Rubik's Cube solver built with Python and OpenCV, featuring live color detection from a mobile camera using a Teachable Machine model and generating solutions with Kociemba's algorithm

# 🧩 AI-Powered Rubik's Cube Solver (Live Camera)

This project provides a real-time Rubik's Cube solver that leverages your mobile phone's camera (via IP Webcam) and a custom-trained Artificial Intelligence (AI) model to detect cube colors and generate a solution using Kociemba's algorithm.

# ✨ Features

Live Camera Input: Uses your mobile phone as an IP webcam to capture live video frames.
AI-Powered Color Detection: Employs a TensorFlow/Keras model (trained with Google's Teachable Machine) to accurately identify the colors of individual Rubik's Cube stickers.
Interactive Scanning: Guides you through scanning each face of the cube by pressing corresponding keys.
Kociemba's Algorithm Integration: Once all faces are scanned, it uses the powerful Kociemba algorithm to calculate the shortest solution.
Solution Display: Prints the solution steps to the console. (Future enhancement could include on-screen visual guidance for each move).
Robust Error Handling: Includes checks for camera connection issues, image decoding failures, and invalid cube states.

# 🚀 Getting Started

Follow these steps to set up and run the Rubik's Cube solver on your machine.
Prerequisites
Before you begin, ensure you have the following installed:
Python 3.x
pip (Python package installer)
An Android/iOS phone with an IP Webcam app installed. (e.g., "IP Webcam" for Android, various options for iOS).
Internet connection (for requests to fetch images and for TensorFlow if it needs to download anything).

# 📦 Installation
```
Clone this repository:
git clone https://github.com/JayPatel1309/CamCube.git
cd CamCube
```

Install Python dependencies:
```
pip install opencv-python numpy requests imutils kociemba tensorflow h5py
```

# 🤖 AI Model Setup (Teachable Machine)

This project relies on a custom AI model for color detection. You need to train and export this model from Google's Teachable Machine.
Train Your Model on Teachable Machine:
Go to teachablemachine.withgoogle.com.
Create an Image Project.
Create 6 classes for your cube's colors: White, Red, Green, Yellow, Orange, Blue.
Highly Recommended: Add a 7th class named Background or Unknown. This helps the model differentiate actual stickers from other parts of the image.
Gather Images: For each class, use your webcam (or upload images) to capture many diverse examples of each color.
Crucial for accuracy: Capture images of individual stickers under different lighting conditions, angles, and backgrounds.
For Red and Orange, make sure you have a good variety to help the model distinguish them.
For Background/Unknown, capture images of your hand, desk, or anything that might appear in the camera view but isn't a sticker.
Train Model: Click "Train Model" and wait for the training to complete.
Preview: Test your model in the preview section to ensure it's performing well.
Export Your Model:
Click the "Export Model" button.
Select the "TensorFlow" tab.
Choose "Keras" as the "Model conversion type."
Click "Download my model".
This will download a .zip file (e.g., converted_keras.zip).
Extract Model Files:
Unzip the downloaded file. You will find two files:
keras_model.h5 (Your trained AI model)
labels.txt (A text file listing your class names in the order the model predicts them, e.g., 0 White, 1 Orange, etc.)
Place Model Files:
Copy both keras_model.h5 and labels.txt into the root directory of this project (where rubik_solver.py is located).

# 📱 IP Webcam Setup

Install IP Webcam App: Download and install an IP Webcam app on your smartphone (e.g., "IP Webcam" for Android, search for similar apps on iOS).
Start Server: Open the app and start the video server.
Get URL: The app will display an IP address and port (e.g., 192.168.1.100:8080). The URL you need for this script is typically http://<IP_ADDRESS>:<PORT>/shot.jpg for single frames.
Update url in rubik_solver.py: Open rubik_solver.py and update the url variable at the top of the script with your phone's IP camera URL.
```
url = "http://YOUR_PHONE_IP:8080/shot.jpg" # Example: "http://192.168.1.100:8080/shot.jpg"
```

Important: Your phone's IP address might change if it reconnects to Wi-Fi. Always verify the URL in the app before running the script.

# 🏃‍♀️ Usage

Run the script:
```
python main.py
```


Position the Cube: Hold your Rubik's Cube in front of your phone's camera. Ensure one face is clearly visible and centered within the green dots on the screen.
Scan Faces:
The script will display a live feed with green dots indicating the sampling points for each sticker.
Press the corresponding key for the face you are currently showing to the camera:
U for Up face ( White center)
R for Right face ( Red center)
F for Front face ( Green center)
D for Down face ( Yellow center)
L for Left face ( Orange center)
B for Back face ( Blue center)
After each key press, the detected colors for that face will be printed in your console.
Get Solution: Once you have scanned all 6 faces (U, R, F, D, L, B), the script will automatically attempt to solve the cube and print the solution steps (e.g., R U R' U') to your console.
Reset: Press R (lowercase) to reset the scanned cube state and start a new scan.
Exit: Press ESC to close the application.

# 💡 Troubleshooting

Error: Could not connect to ...:
Double-check the url in rubik_solver.py. Is the IP address correct? Is the port correct?
Is your IP Webcam app running on your phone and is its server "started"?
Are both your computer and phone connected to the same Wi-Fi network?
Check your computer's firewall; it might be blocking the connection.
Error: Could not decode image. 'frame' is None.:
This means the data received from the URL wasn't a valid image.
Verify the URL in a web browser on your PC. If you don't see an image, the URL is wrong or the app isn't serving images correctly.
Error loading AI model: ...:
Ensure keras_model.h5 and labels.txt are in the same directory as your Python script.
Verify you have tensorflow and h5py installed (pip install tensorflow h5py).
Sometimes, Teachable Machine models trained on older TensorFlow versions can cause issues with newer ones. The h5py patching code at the top of the script tries to mitigate this.
Warning: Patch at (...) is empty or has zero dimensions.:
This means the script is trying to sample a sticker position that is too close to the edge of the camera frame, or even outside it.
Adjust the spacing and + 50 offset in the x and y calculations within the for i in range(grid_size): for j in range(grid_size): loop. You might need to move the cube closer/further, or adjust the frame = cv2.resize(frame,(750,640)) dimensions.
Error solving cube with Kociemba: ... (e.g., "Invalid cube state"):
This is the most common error if your color detection is not perfect.
Check your scanned colors: Look at the console output after scanning each face. Do the detected colors (W, R, O, G, B, Y) accurately reflect the actual colors on your cube?
Verify labels.txt: Ensure the labels.txt file matches the colors you trained in Teachable Machine and that AI_COLOR_LABELS in the script is correctly populated.
Improve AI Model Training: If colors are consistently misidentified (e.g., red vs. orange), go back to Teachable Machine and add more training examples for those confusing colors, especially under varying lighting. Include Background images!
Scanning Order: Make sure you're scanning the faces in the correct Kociemba order (U, R, F, D, L, B) and maintaining the cube's orientation between scans.

# 🤝 Contributing

Feel free to fork this repository, make improvements, and submit pull requests!

# 📄 License

This project is open-source and available under the MIT License.
