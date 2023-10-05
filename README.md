# Fast API Pytorch endpoint 

A basic example of FastAPI endpoint that works with Pytorch to identify what is depicted on the image and runs in Docker

Install:

    docker compose up --build

Usage:

    http://0.0.0.0:8000/docs

POST /image_upload
Upload any image with a question to 'text' field, like 'What is depicted on the image?'
