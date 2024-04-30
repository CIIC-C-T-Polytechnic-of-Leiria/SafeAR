# Description: Dockerfile for SafeAR
FROM python:3.10

# Set the working directory
WORKDIR /SafeAR

# Copy the current directory contents into the container at /SafeAR
ADD . /SafeAR

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Run Flask app
CMD ["python3", "src/flask_server.py"]
