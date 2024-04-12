let isRecording = false; // Variable to track recording state
let mediaRecorder; // MediaRecorder object
let socket; // WebSocket object

const recordButton = document.getElementById('recordButton');
const stopButton = document.getElementById('stopButton');
const videoElement = document.getElementById('videoElement');
const millisecondsBatch = 500
const API_URL = 'http://localhost:8000'
const PROCESS_RATE = 4; // Example processing rate, TODO: user adjusts in the UI

// Upload file functionality
document.getElementById('uploadForm').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent default form submission

    var formData = new FormData();
    var fileInput = document.getElementById('file_upload');
    formData.append('file_upload', fileInput.files[0]); // Append file to FormData
    
    var submitButton = document.querySelector('button[type="submit"]');
    submitButton.disabled = true;

    fetch(`${API_URL}/uploadfile/`, {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // Display result in the table
        var resultTable = document.getElementById('resultTable');
        var newRow = resultTable.insertRow();
        newRow.insertCell(0).appendChild(document.createTextNode(fileInput.files[0].name));
        newRow.insertCell(1).appendChild(document.createTextNode(data.predicted_word));
        newRow.insertCell(2).appendChild(document.createTextNode(data.confidence_level));

        // Enable the submit button
        submitButton.disabled = false;
    })
    .catch(error => console.error('Error:', error));
});



let recordedChunks = [];
let lastWord = ''

recordButton.addEventListener('click', async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    videoElement.srcObject = stream;

    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.start(millisecondsBatch);
    mediaRecorder.ondataavailable = async(event) => {
        recordedChunks.push(event.data);

        const blob = new Blob(recordedChunks, { type: 'video/webm' });
        const formData = new FormData();
        formData.append('video', blob);
        const rate = PROCESS_RATE;
        formData.append('rate', rate);

        const response = await fetch(`${API_URL}/live/stream`, {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();
        
        // Check if data is an array
        if (Array.isArray(data.words)) {
            // Iterate through the 2D array of words and confidence values
            data.words.forEach(([word, confidence]) => {
                const textDiv = document.getElementById('output');
                // Create a new <span> element for each word
                const wordSpan = document.createElement('span');
                wordSpan.textContent = word + ' ';
                if (lastWord === word){
                    return
                }
                lastWord = word; // Update lastWord with the current word
                // Set the color of the <span> based on the confidence level
                if (confidence >= 0.8) {
                    wordSpan.style.color = 'green'; // High confidence, green color
                } else if (confidence >= 0.5) {
                    wordSpan.style.color = 'orange'; // Medium confidence, orange color
                } else {
                    wordSpan.style.color = 'red'; // Low confidence, red color
                }
                // Append the <span> element to the container <div>
                textDiv.appendChild(wordSpan);
            });
        } else {
            console.error('Data is not an array:', data);
        }


    };
});

stopButton.addEventListener('click', async() => {
    const response = await fetch(`${API_URL}/live/stop`, {
        method: 'POST',
        body: '',
    });

    // Stop the video stream in the UI
    const stream = videoElement.srcObject;
    const tracks = stream.getTracks();
    tracks.forEach(track => track.stop());
    videoElement.srcObject = null;

    mediaRecorder.stop();
});
