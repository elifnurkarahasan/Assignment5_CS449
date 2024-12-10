// Copyright 2023 The MediaPipe Authors.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//      http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
import {
    GestureRecognizer,
    FilesetResolver,
    DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

const demosSection = document.getElementById("demos");
let gestureRecognizer: GestureRecognizer;
let runningMode = "IMAGE";
let enableWebcamButton: HTMLButtonElement;
let webcamRunning: Boolean = false;
const videoHeight = "360px";
const videoWidth = "480px";

// Before we can use HandLandmarker class we must wait for it to finish
// loading. Machine Learning models can be large and take a moment to
// get everything needed to run.
const createGestureRecognizer = async () => {
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
    );
    gestureRecognizer = await GestureRecognizer.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath:
                "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
            delegate: "GPU"
        },
        runningMode: runningMode
    });
    demosSection.classList.remove("invisible");
};
createGestureRecognizer();

/********************************************************************
 // Demo 1: Detect hand gestures in images
 ********************************************************************/

const imageContainers = document.getElementsByClassName("detectOnClick");

for (let i = 0; i < imageContainers.length; i++) {
    imageContainers[i].children[0].addEventListener("click", handleClick);
}

async function handleClick(event) {
    if (!gestureRecognizer) {
        alert("Please wait for gestureRecognizer to load");
        return;
    }

    if (runningMode === "VIDEO") {
        runningMode = "IMAGE";
        await gestureRecognizer.setOptions({ runningMode: "IMAGE" });
    }
    // Remove all previous landmarks
    const allCanvas = event.target.parentNode.getElementsByClassName("canvas");
    for (var i = allCanvas.length - 1; i >= 0; i--) {
        const n = allCanvas[i];
        n.parentNode.removeChild(n);
    }

    const results = gestureRecognizer.recognize(event.target);

    // View results in the console to see their format
    console.log(results);
    if (results.gestures.length > 0) {
        const p = event.target.parentNode.childNodes[3];
        p.setAttribute("class", "info");

        const categoryName = results.gestures[0][0].categoryName;
        const categoryScore = parseFloat(
            results.gestures[0][0].score * 100
        ).toFixed(2);
        const handedness = results.handednesses[0][0].displayName;

        p.innerText = `GestureRecognizer: ${categoryName}\n Confidence: ${categoryScore}%\n Handedness: ${handedness}`;
        p.style =
            "left: 0px;" +
            "top: " +
            event.target.height +
            "px; " +
            "width: " +
            (event.target.width - 10) +
            "px;";

        const canvas = document.createElement("canvas");
        canvas.setAttribute("class", "canvas");
        canvas.setAttribute("width", event.target.naturalWidth + "px");
        canvas.setAttribute("height", event.target.naturalHeight + "px");
        canvas.style =
            "left: 0px;" +
            "top: 0px;" +
            "width: " +
            event.target.width +
            "px;" +
            "height: " +
            event.target.height +
            "px;";

        event.target.parentNode.appendChild(canvas);
        const canvasCtx = canvas.getContext("2d");
        const drawingUtils = new DrawingUtils(canvasCtx);
        for (const landmarks of results.landmarks) {
            drawingUtils.drawConnectors(
                landmarks,
                GestureRecognizer.HAND_CONNECTIONS,
                {
                    color: "#00FF00",
                    lineWidth: 5
                }
            );
            drawingUtils.drawLandmarks(landmarks, {
                color: "#FF0000",
                lineWidth: 1
            });
        }
    }
}

/********************************************************************
 // Demo 2: Continuously grab image from webcam stream and detect it.
 ********************************************************************/

const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const gestureOutput = document.getElementById("gesture_output");

// Check if webcam access is supported.
function hasGetUserMedia() {
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

// If webcam supported, add event listener to button for when user
// wants to activate it.
if (hasGetUserMedia()) {
    enableWebcamButton = document.getElementById("webcamButton");
    enableWebcamButton.addEventListener("click", enableCam);
} else {
    console.warn("getUserMedia() is not supported by your browser");
}



// Enable the live webcam view and start detection.
function enableCam(event) {
    if (!gestureRecognizer) {
        alert("Please wait for gestureRecognizer to load");
        return;
    }

    if (webcamRunning === true) {
        webcamRunning = false;
        enableWebcamButton.innerText = "ENABLE PREDICTIONS";
    } else {
        webcamRunning = true;
        enableWebcamButton.innerText = "DISABLE PREDICTIONS";
    }

    // getUsermedia parameters.
    const constraints = {
        video: true
    };

    // Activate the webcam stream.
    navigator.mediaDevices.getUserMedia(constraints).then(function (stream) {
        video.srcObject = stream;
        video.addEventListener("loadeddata", predictWebcam);
    });
}

// Update the position of the virtual cursor
function moveVirtualCursor(indexTip) {
    const cursor = document.getElementById("virtualCursor");
    const cursorX = (1 - indexTip.x) * window.innerWidth;
    const cursorY = indexTip.y * window.innerHeight;

    cursor.style.left = `${cursorX}px`;
    cursor.style.top = `${cursorY}px`;
}


// Simulate a click event
function simulateClick(x, y) {
    const element = document.elementFromPoint(x, y);
    if (element) {
        const event = new MouseEvent("click", {
            view: window,
            bubbles: true,
            cancelable: true,
            clientX: x,
            clientY: y,
        });
        element.dispatchEvent(event);
    }
}

let lastVideoTime = -1;
let results = undefined;
let lastIndexX = null; // Store x-coordinate of the index finger tip from the last frame
let lastMiddleX = null; // Store x-coordinate of the middle finger tip from the last frame
let lastIndexY = null; // Store y-coordinate of the index finger tip from the last frame
let lastMiddleY = null; // Store y-coordinate of the middle finger tip from the last frame
let gestureStableFrames = 0; // Count stable frames for the two-finger gesture
const stabilityThreshold = 1; // Minimum stable frames required


async function predictWebcam() {
    const webcamElement = document.getElementById("webcam");
    // Now let's start detecting the stream.
    if (runningMode === "IMAGE") {
        runningMode = "VIDEO";
        await gestureRecognizer.setOptions({ runningMode: "VIDEO" });
    }
    let nowInMs = Date.now();
    if (video.currentTime !== lastVideoTime) {
        lastVideoTime = video.currentTime;
        results = gestureRecognizer.recognizeForVideo(video, nowInMs);
    }

    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    const drawingUtils = new DrawingUtils(canvasCtx);

    canvasElement.style.height = videoHeight;
    webcamElement.style.height = videoHeight;
    canvasElement.style.width = videoWidth;
    webcamElement.style.width = videoWidth;

    if (results.landmarks) {
        for (const landmarks of results.landmarks) {
            drawingUtils.drawConnectors(
                landmarks,
                GestureRecognizer.HAND_CONNECTIONS,
                {
                    color: "#00FF00",
                    lineWidth: 5
                }
            );
            drawingUtils.drawLandmarks(landmarks, {
                color: "#FF0000",
                lineWidth: 2
            });
            const thumbTip = landmarks[4];
            const indexTip = landmarks[8];
            const middleTip = landmarks[12];
            const ringTip = landmarks[16];
            const pinkyTip = landmarks[20];
            const indexBase = landmarks[5];
            const middleBase = landmarks[9];
            const ringBase = landmarks[13];
            const pinkyBase = landmarks[17];
            const thumbBase = landmarks[2];


            const isIndexRaised = indexTip.y < indexBase.y;
            const isMiddleRaised = middleTip.y < middleBase.y;
            const isRingDown = ringTip.y >= ringBase.y;
            const isPinkyDown = pinkyTip.y >= pinkyBase.y;
            const isThumbDown = thumbTip.y >= thumbBase.y;
            const isThumbUp = thumbTip.y < thumbBase.y;
            const isMiddleDown = middleTip.y >= middleBase.y;



            function calculateDistance(landmark1, landmark2) {
            return Math.sqrt(
                Math.pow(landmark1.x - landmark2.x, 2) +
                Math.pow(landmark1.y - landmark2.y, 2)
              );
            }

            // Detect Cursor Gesture (thumb and index up, others down)
            if (isThumbUp && isIndexRaised && isMiddleDown && isRingDown && isPinkyDown) {
                results.gestures[0][0].categoryName = "cursor";
                moveVirtualCursor(indexTip);
            }


            // Detect Pinch Gesture for Clicking
            const thumbIndexDistance = calculateDistance(thumbTip, indexTip);
            if (thumbIndexDistance  < 0.04 && isThumbUp && isIndexRaised && isMiddleDown && isRingDown && isPinkyDown) {
                results.gestures[0][0].categoryName = "click";
                // Use the virtual cursor's position for the click
                const cursorElement = document.getElementById("virtualCursor");
                const cursorRect = cursorElement.getBoundingClientRect();
                const clickTarget = document.elementFromPoint(cursorRect.left + cursorRect.width / 2, cursorRect.top + cursorRect.height / 2);

                if (clickTarget) {
                    clickTarget.click();
                    console.log(`Clicked on element:`, clickTarget);
                }

            }


            if (calculateDistance(indexTip, middleTip) < 0.05 && isRingDown && isPinkyDown) {


                if (isIndexRaised && isMiddleRaised){
                gestureStableFrames += 1;
                // Ensure gesture is stable for a few frames
                if (gestureStableFrames >= stabilityThreshold) {
                    if (lastIndexX !== null && lastMiddleX !== null) {
                        const currentIndexX = indexTip.x;
                        const currentMiddleX = middleTip.x;
                        const currentIndexY = indexTip.y;
                        const currentMiddleY = middleTip.y;

                        // Calculate average horizontal movement of the two fingers
                        const movementX = ((currentIndexX - lastIndexX) + (currentMiddleX - lastMiddleX)) / 2;
                        const movementY = ((currentIndexY - lastIndexY) + (currentMiddleY - lastMiddleY)) / 2;


                        // Trigger horizontal scroll if movement exceeds threshold
                        if (Math.abs(movementX) > Math.abs(movementY) && Math.abs(movementX) > 0.01) {
                            // Horizontal scrolling
                            const scrollAmount = movementX * window.innerWidth * 0.8; // Scale movement and reduce speed
                            window.scrollBy(scrollAmount, 0); // Scroll horizontally
                            console.log(`Scrolling Horizontally: ${scrollAmount > 0 ? "Right" : "Left"}`);
                            // Update category
                            results.gestures[0][0].categoryName = "horizontal scroll";
                        }
                        else if (Math.abs(movementY) > Math.abs(movementX) && Math.abs(movementY) > 0.01) {
                            // Vertical scrolling
                            const scrollAmount = movementY * window.innerHeight * 0.8; // Scale movement and reduce speed
                            window.scrollBy(0, scrollAmount); // Scroll vertically
                            console.log(`Scrolling Vertically: ${scrollAmount > 0 ? "Down" : "Up"}`);
                            // Update category
                            results.gestures[0][0].categoryName = "vertical scroll";
                        }

                    }

                }
                // Update last positions
                lastIndexX = indexTip.x;
                lastMiddleX = middleTip.x;
                lastIndexY = indexTip.y;
                lastMiddleY = middleTip.y;
            }
            else {
                // Reset if the two-finger gesture is not detected
                gestureStableFrames = 0;
                lastIndexX = null;
                lastMiddleX = null;
                lastIndexY = null;
                lastMiddleY = null;
            }

            }

        }
    }
    canvasCtx.restore();
    if (results.gestures.length > 0) {
        gestureOutput.style.display = "block";
        gestureOutput.style.width = videoWidth;
        const categoryName = results.gestures[0][0].categoryName;
        const categoryScore = parseFloat(
            results.gestures[0][0].score * 100
        ).toFixed(2);
        const handedness = results.handednesses[0][0].displayName;
        gestureOutput.innerText = `GestureRecognizer: ${categoryName}\n Confidence: ${categoryScore} %\n Handedness: ${handedness}`;
    } else {
        gestureOutput.style.display = "none";
    }
    // Call this function again to keep predicting when the browser is ready.
    if (webcamRunning === true) {
        window.requestAnimationFrame(predictWebcam);
    }
}


document.getElementById("customButton").addEventListener("click", () => {
    document.body.style.backgroundColor = "pink";

});
