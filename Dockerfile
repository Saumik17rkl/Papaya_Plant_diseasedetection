# Use a lightweight Python image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the local code to the container
COPY . /app

# Install Python dependencies
RUN pip install -r requirements.txt

# Expose the port the app will run on
EXPOSE 8000

# Command to run the FastAPI app using uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
