# Use the official Python image as the base image
FROM python:3.9

# Set the working directory to /app
WORKDIR /app

# Copy the inference folder to /app/inference
COPY /inference /app/inference

# Update pip and install requirements
RUN apt-get update && apt-get install -y git gcc python3-dev build-essential libgl1-mesa-dev libglib2.0-0 libsm6 libxrender1 libxext6
# RUN apt-get install -y socat
RUN python -m pip install pip==21.1.1
RUN pip install numpy
RUN pip install --upgrade --force-reinstall "git+https://github.com/ytdl-org/youtube-dl.git"
RUN pip install lap==0.4.0
RUN pip install --upgrade pip
RUN pip install  -r inference/requirements.txt
RUN pip install uvicorn

WORKDIR /app/inference

CMD ["uvicorn", "fast:app", "--host", "::", "--port", "80"]
