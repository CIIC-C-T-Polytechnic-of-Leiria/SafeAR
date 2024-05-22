# Description: Dockerfile for SafeAR
FROM python:3.10

# Set the working directory
WORKDIR /SafeAR

# Copy the current directory contents into the container at /SafeAR
ADD . /SafeAR

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Make port 6666 available to the world outside this container
EXPOSE 6666

# Run Flask app
CMD ["python", "server_http.py"]
