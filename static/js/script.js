/**
 * FacePass Configuration File
 * Central configuration for face recognition parameters
 */

// Recognition threshold - lower value means stricter matching
// Values < 0.4 = excellent match
// Values 0.4-0.6 = good match
// Values > 0.6 = poor match/unknown
const RECOGNITION_THRESHOLD = 0.45;

// Face detection settings
const DETECTION_CONFIG = {
    minConfidence: 0.5,        // Minimum confidence for face detection
    detectionInterval: 100,    // Detection interval in milliseconds
    stabilityThreshold: 3      // Number of consecutive frames needed to stabilize label
};

// Visual settings
const VISUAL_CONFIG = {
    boxLineWidth: 4,
    labelHeight: 35,
    knownFaceColor: '#4dbb6f',     // Green for recognized faces
    unknownFaceColor: '#cc3366',    // Red for unknown faces
    detectedFaceColor: '#3399ff'    // Blue for detected but not matched faces
};

// Debug settings
const DEBUG_CONFIG = {
    logRecognitionAttempts: true,   // Log all recognition attempts
    logToServer: true,              // Send logs to server
    showDebugInfo: true,            // Show debug info in UI
    detailedConsoleLog: true        // Detailed console logging
};

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        RECOGNITION_THRESHOLD,
        DETECTION_CONFIG,
        VISUAL_CONFIG,
        DEBUG_CONFIG
    };
}

console.log('🔧 FacePass Config Loaded - Recognition Threshold:', RECOGNITION_THRESHOLD);

// Predefined face data for recognition testing
const PREDEFINED_FACES = [
    {
        "image": "data:image/jpeg;base64,/9j/4AAQ...", // BASE64 IMAGE
        "name": "shasvant"
    }
];
