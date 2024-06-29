"""
server_tester.py

Description: This script sends an image to a remote server and displays the returned image and latency.
Author: Tiago F. R. Ribeiro
Last Modified: 2024-06-29

TODO:
 - Check why we are getting Error: 400 Client Error: BAD REQUEST for url: http://172.22.21.43:8081/video
 - Test image and metrics display
"""
import argparse
import base64
import time
from typing import List, Deque
from collections import deque
import random
import threading
import sys
import requests
from requests.exceptions import RequestException
import os
from datetime import timedelta
import matplotlib.pyplot as plt
from PIL import Image
import io
import numpy as np

latencies: List[float] = []
processed_frames: int = 0
failed_requests: int = 0
start_time: float = time.time()
running: bool = True
error_messages: Deque[str] = deque(maxlen=2)
exit_event = threading.Event()


def display_image(image_data: bytes):
    img = Image.open(io.BytesIO(image_data))
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.title("Returned Image")
    plt.show(block=False)
    plt.pause(0.1)


def plot_latency(latencies: List[float]):
    plt.figure(figsize=(10, 4))
    plt.plot(latencies)
    plt.title("Latency over time")
    plt.xlabel("Request number")
    plt.ylabel("Latency (seconds)")
    plt.show(block=False)
    plt.pause(0.1)


def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')


def update_metrics(url: str, port: int, image_file: str):
    global processed_frames, failed_requests, latencies, start_time
    image_base_name = os.path.basename(image_file)

    while not exit_event.is_set():
        current_time = time.time()
        elapsed_time = current_time - start_time
        fps = processed_frames / elapsed_time if elapsed_time > 0 else 0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        total_requests = processed_frames + failed_requests

        clear_console()
        print("\n┌──────────────────────────────────────────────────────────┐")
        print("│                    Performance Metrics                   │")
        print("├───────────────────────────────┬──────────────────────────┤")
        print(f"│ Server URL                    │ {url}:{port:<11} │")
        print(f"│ Image File                    │ {image_base_name:<24} │")
        print("├───────────────────────────────┼──────────────────────────┤")
        print(f"│ Frames Per Second             │ {fps:24.2f} │")
        print(f"│ Average Latency               │ {avg_latency * 1000:21.2f} ms │")
        print(f"│ Processed Frames              │ {processed_frames:24d} │")
        print(f"│ Failed Requests               │ {failed_requests:24d} │")
        print(f"│ Total Requests                │ {total_requests:24d} │")
        print(f"│ Elapsed Time                  │ {str(timedelta(seconds=int(elapsed_time))):>24s} │")
        print("└───────────────────────────────┴──────────────────────────┘")

        print("\n┌─────────────────────────────────────────────────────────┐")
        print("│                    Error Messages                       │")
        print("└─────────────────────────────────────────────────────────┘")
        for msg in error_messages:
            print(f"-  {msg:<53} ")
        for _ in range(2 - len(error_messages)):
            print("                                                         ")
        print("└─────────────────────────────────────────────────────────┘")

        print("\nPress 'q' then Enter to quit.")

        exit_event.wait(1)  # Update metrics every second


def send_request(url: str, image_path: str, timeout: int = 1000) -> tuple:
    global processed_frames, failed_requests

    # Verifica se a extensão do arquivo é .jpg ou .png
    if not image_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        error_messages.append("Error: The image file must be a JPEG or PNG file.")
        return 0, None

    with open(image_path, "rb") as image_file:
        img_base64 = base64.b64encode(image_file.read()).decode("utf-8")

    start_time: float = time.time()
    try:
        files = {'imageData': img_base64}
        # Create a file to log the image data sent and save it to the current directory
        with open('src/debug/files_data.txt', 'w') as f:
            f.write(str(files))

        # print(f"url: {url}")
        response = requests.post(
            url,
            files=files,
            headers={"Content-Type": "multipart/form-data"}
        )

        # response = requests.post(
        #     url,
        #     files=files,
        #     headers={"Content-Type": "multipart/form-data"},
        #     timeout=timeout
        # )
        # print(f"files: {files}")
        response.raise_for_status()
        processed_frames += 1
        latency = time.time() - start_time
        latencies.append(latency)
        return latency, response.content
    except RequestException as e:
        failed_requests += 1
        error_messages.append(f"Error: {str(e)}")
        return 0, None


def process_image(url: str, image_path: str):
    max_retries = 5
    while not exit_event.is_set():
        for attempt in range(max_retries):
            if exit_event.is_set():
                return
            try:
                latency, image_data = send_request(url, image_path)
                if latency > 0:
                    display_image(image_data)
                    plot_latency(latencies)
                    break
            except Exception as e:
                error_messages.append(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1 and not exit_event.is_set():
                    wait_time = (2 ** attempt) + (random.randint(0, 1000) / 1000)
                    exit_event.wait(wait_time)
                else:
                    error_messages.append("Max retries reached or stopping. Moving to next iteration.")
                    break


def main(args: argparse.Namespace):
    base_url = f"http://{args.ip}:{args.port}"
    test_url = f"{base_url}/test"
    video_url = f"{base_url}/video"

    print(f"Verifying if the service is running {test_url}")
    try:
        response = requests.get(test_url)
        response.raise_for_status()
        print("Server is running.")
    except RequestException as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

    print(f"Sending image to {video_url}")

    plt.ion()  # Turn on interactive mode for matplotlib

    metrics_thread = threading.Thread(target=update_metrics, args=(args.ip, args.port, args.image_file))
    metrics_thread.daemon = True
    metrics_thread.start()

    processing_thread = threading.Thread(target=process_image, args=(video_url, args.image_file))
    processing_thread.daemon = True
    processing_thread.start()

    try:
        while not exit_event.is_set():
            if input().lower() == 'q':
                print("Stopping threads and exiting...")
                exit_event.set()
                break
    except KeyboardInterrupt:
        print("\nInterrupted by user. Stopping threads and exiting...")
        exit_event.set()

    # Wait for threads to finish
    metrics_thread.join(timeout=2)
    processing_thread.join(timeout=2)

    print("Client service stopped.")
    plt.close('all')  # Close all matplotlib windows
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processes an image using a remote server.")
    parser.add_argument("--ip", default="172.22.21.43", help="IP address of the server")
    parser.add_argument("--port", type=int, default=8081, help="Server port")
    parser.add_argument("--image_file", required=True, help="Image file path to process")

    args = parser.parse_args()
    main(args)

# 172.22.21.43:8081/video POST
