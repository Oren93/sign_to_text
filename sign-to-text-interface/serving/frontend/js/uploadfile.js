let isRecording = false; // Variable to track recording state
let mediaRecorder; // MediaRecorder object
let socket; // WebSocket object

const recordButton = document.getElementById('recordButton');
const stopButton = document.getElementById('stopButton');
const videoElement = document.getElementById('videoElement');
const sendToBackendButton = document.getElementById('sendToBackendButton');
const millisecondsBatch = 5000
const API_URL = 'http://localhost:8000'

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
recordButton.addEventListener('click', async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    videoElement.srcObject = stream;

    console.log(stream) //////////////////////////////////////
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.start(millisecondsBatch)
    mediaRecorder.ondataavailable = async(event) => {
        console.log('ondataavailable')//////////////////////////////////////
        recordedChunks.push(event.data);

        const blob = new Blob(recordedChunks, { type: 'video/webm' });
        const formData = new FormData();
        formData.append('video', blob);
        console.log("blob")//////////////////////////////////////
        console.log(blob)//////////////////////////////////////
        console.log("formData")//////////////////////////////////////
        console.log(formData)//////////////////////////////////////

        const response = await fetch(`${API_URL}/live/stream`, {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();
        console.log(data);//////////////////////////////////////
    };
});

stopButton.addEventListener('click', () => {
    console.log('Stop record')//////////////////////////////////////
    mediaRecorder.stop();
});

sendToBackendButton.addEventListener('click', async () => {
    const blob = new Blob(recordedChunks, { type: 'video/webm' });
    const formData = new FormData();
    formData.append('video', blob);
    console.log("blob")//////////////////////////////////////
    console.log(blob)//////////////////////////////////////
    console.log("formData")//////////////////////////////////////
    console.log(formData)//////////////////////////////////////

    const response = await fetch(`${API_URL}/live/stream`, {
        method: 'POST',
        body: formData,
    });

    const data = await response.json();
    console.log(data);//////////////////////////////////////
});

