# FacePass - Advanced Face Recognition System

## Overview
FacePass is an advanced face recognition system built using Flask and MediaPipe. This project allows users to capture faces, train a recognition model, and perform real-time face recognition through a web interface.

## Project Structure
```
face-recognition
├── app.py                # Main application file
├── static
│   ├── css
│   │   └── style.css     # CSS styles for the web application
│   └── js
│       └── script.js     # JavaScript functionality for user interactions
├── templates
│   ├── index.html        # Home page template
│   ├── capture.html      # Face capture page template
│   ├── train.html        # Training page template
│   └── recognize.html     # Recognition page template
├── known_faces
│   └── .gitkeep          # Placeholder for Git tracking
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## Installation
1. Clone the repository:
   ```
   git clone <repository-url>
   cd face-recognition
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
1. Start the Flask application:
   ```
   python app.py
   ```

2. Open your web browser and navigate to `http://localhost:5000`.

3. Use the navigation links to:
   - Capture faces for training.
   - Train the face recognition model.
   - Start the face recognition process.

## Dependencies
- Flask
- OpenCV
- NumPy
- MediaPipe

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License.