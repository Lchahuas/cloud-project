# Dockerfile
FROM python:3.8-slim

WORKDIR /app

# Copy requirements.txt from two directories above
COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

# Copy the rest of the application code from the current directory
COPY . .

CMD ["python", "train_model1.py"]
