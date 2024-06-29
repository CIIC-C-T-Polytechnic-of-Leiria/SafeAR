"""
Dummy client to send an image to the Flask server for obfuscation.

This script is part of a larger project and is used to send an image file to a Flask server for obfuscation.
The image is read from the specified file path, encoded in base64, and then sent as a POST request to the server.
The server's response is then printed to the console.

Author: Tiago F. R. Ribeiro
License: MIT
"""

import argparse
import base64
import requests

def send_image_to_server(image_path):
    """
    Function to send an image to the Flask server for obfuscation.

    Args:
    image_path (str): The path to the image file to send to the server.
    """
    with open(image_path, "rb") as image_file:
        img_bytes = image_file.read()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

    # Send a POST request to the Flask server
    response = requests.post(
        "http://localhost:5000/obfuscate", json={"image": img_base64}
    )

    # Print the response status code and data
    print(f"Response status code: {response.status_code}")
    print(f"Obfuscated image data: {response.json()['image']}")

if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(
        description="Send an image to the Flask server for obfuscation."
    )

    # Add an argument for the image path
    parser.add_argument(
        "image_path", help="The path to the image file to send to the server."
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the send_image_to_server function with the image path argument
    send_image_to_server(args.image_path)
