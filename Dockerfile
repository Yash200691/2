# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements (create this file if you don't have it)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose port (Railway uses 5000 by default for Flask)
EXPOSE 5000

# Set environment variable for Flask
ENV FLASK_APP=app.py

# Run the Flask app
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]