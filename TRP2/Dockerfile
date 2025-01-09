# Base image
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Copy the project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose necessary ports (if applicable, adjust based on your project)
EXPOSE 8080

# Define the entry point for the application
ENTRYPOINT ["python"]
CMD ["main.py"]
