# Sign Language Recognition with Machine Learning

Welcome to the Sign Speech Recognition project! This repository contains the complete work of a semester-long project developed for the "AI Lifecycle" course at the University of Iceland. The project focuses on building and deploying a machine learning model for sign language recognition, covering the entire lifecycle from data exploration to serving the model via a user-friendly interface.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Data Source](#data-source)
- [Project Structure](#project-structure)
- [Assignments and Presentations](#assignments-and-presentations)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)
- [Further Development](#further-development)

## Overview

This project aims to create a machine learning model for recognizing sign language and serving it through a web-based interface. The focus is not only on the model itself but on the entire data science lifecycle, including data exploration, model training, explainability deployment, and serving.

## Features

- **Sign Language Recognition Model**: A trained machine learning model capable of recognizing sign language.
- **Dockerized Deployment**: The entire application can be easily run using Docker.
- **User Interface**: A web-based UI for interacting with the model.
- **Comprehensive Documentation**: Includes weekly assignment descriptions and powerpoint presentations detailing the project's progress and methodology.

## Installation and Setup

To run the project locally, ensure you have Docker and Docker Compose installed. Follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/Oren93/sign_to_text.git
   cd sign_to_text
   ```

2. Build and run the Docker containers:

   ```bash
   docker compose up -d --build
   ```

3. Open your browser and go to `http://localhost:8080`.

## Usage

Once the application is running, you can interact with the sign speech recognition model through the web interface. Upload sign language videos or use the webcam feature to get real-time predictions.

_NOTE: This is a prototype only, due to limited computing power we used a small subset of the dataset and trained only a few epochs, therefore the model can only detect 16-32 words_

## Data Source

The data used for training and evaluating the sign language recognition model was collected by members of The Australian National University. You can read more about their work and the dataset in their [article](https://arxiv.org/pdf/1910.11006).

## Project Structure

```markdown
sign-to-text/
├── README.md
├── assignments
│   ├── assignment_description # PDF of the weekly assignment
│   ├── assignment_presentations # Slides used for our weekly presentations
│   ├── data # Metadata of the videos
│   └── videos # Not included on Github, need to apply for it
├── docker-compose.yml
└── serving
├── Dockerfile
├── cnn # CNN model, not used eventually
├── frontend
├── landmark_nn # Mediapipe landmarks model, used for live prediction
├── lstm # LSTM model, used for single word video upload
└── uploads # Stores the uploaded videos
```

## Assignments and Presentations

In the `assignment_description` and `assignment_presentations` folders, you will find detailed documentation and presentations used throughout the course. These documents provide insights into our methodology, challenges faced, and the evolution of the project.

## Contributing

We welcome contributions from the community. If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Submit a pull request.

## Acknowledgements

This project was developed as part of the "AI Lifecycle" course at the University of Iceland. We would like to thank our professor, Hafsteinn Einarsson, for guidance and support throughout the semester.

## Further Development

During the semester, we applied to the Icelandic Innovation Fund (Rannís) and received a summer grant to develop a web interface for a data collection platform. This platform enables members of the Icelandic deaf community (hopefully will scale it to other countries) to record themselves signing words, contributing to the creation of a large, comprehensive dataset.

You can find the repository for that project [here](https://github.com/Oren93/sign_lang_data).
