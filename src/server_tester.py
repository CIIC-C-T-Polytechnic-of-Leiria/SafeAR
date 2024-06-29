import argparse
import base64
import os
import threading
import time
from typing import List

import matplotlib.pyplot as plt
import requests
from colorama import Fore, Back, Style, init

# Initialize colorama
init(autoreset=True)

# Global variables for metrics
latencies: List[float] = []
processed_frames: int = 0
start_time: float = time.time()


def print_colorful_usage():
    """
    Print a colorful usage guide for the script.
    """
    print(f"{Fore.CYAN}{Style.BRIGHT}==================================")
    print(f"{Fore.YELLOW}Image Processing and Metrics Script")
    print(f"{Fore.CYAN}{Style.BRIGHT}=================================={Style.RESET_ALL}")
    print(f"\n{Fore.GREEN}This script processes images and displays metrics for server requests.")
    print(f"\n{Fore.MAGENTA}Usage:")
    print(f"  {Fore.WHITE}python script_name.py [options]")
    print(f"\n{Fore.MAGENTA}Options:")
    print(f"  {Fore.YELLOW}--ip IP{Fore.WHITE}        Server IP address (default: 172.22.21.43)")
    print(f"  {Fore.YELLOW}--port PORT{Fore.WHITE}    Server port (default: 8081)")
    print(f"  {Fore.YELLOW}--image_folder PATH{Fore.WHITE}    Path to the folder containing images")
    print(f"  {Fore.YELLOW}--fps FPS{Fore.WHITE}      Frames per second to simulate (default: 30)")
    print(f"\n{Fore.MAGENTA}Example:")
    print(f"  {Fore.WHITE}python script_name.py --ip 192.168.1.100 --port 8080 --image_folder /path/to/images --fps 25")
    print(f"\n{Fore.CYAN}{Style.BRIGHT}=================================={Style.RESET_ALL}")
    print(f"\n")


def send_request(url: str, image_path: str) -> float:
    """
    Send a request to the server with the encoded image.

    Args:
        url (str): The URL of the server.
        image_path (str): The path to the image file.

    Returns:
        float: The latency of the request.
    """
    global processed_frames

    # Load and encode the image
    with open(image_path, "rb") as image_file:
        img_base64 = base64.b64encode(image_file.read()).decode("utf-8")

    # Send request to the server
    start_time = time.time()
    response = requests.post(
        url,
        headers={"Content-Type": "multipart/form-data"},
        data={"imageData": img_base64},
    )
    end_time = time.time()

    # Calculate and store latency
    latency = end_time - start_time
    latencies.append(latency)

    # Process response
    if response.status_code == 200:
        processed_frames += 1
        print(f"Frame {processed_frames} processed successfully.")
    else:
        print(f"Error processing frame: {response.status_code}")

    return latency


def process_images(url: str, image_folder: str, frame_delay: float):
    """
    Process images from the specified folder.

    Args:
        url (str): The URL of the server.
        image_folder (str): The path to the folder containing images.
        frame_delay (float): The delay between processing frames.
    """
    image_files = [
        f for f in os.listdir(image_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        latency = send_request(url, image_path)
        print(f"Image: {image_file}, Latency: {latency:.4f} seconds")
        time.sleep(frame_delay)


def show_metrics():
    """
    Continuously display current metrics.
    """
    while True:
        time.sleep(5)  # Update every 5 seconds
        current_time = time.time()
        elapsed_time = current_time - start_time
        fps = processed_frames / elapsed_time if elapsed_time > 0 else 0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0

        print(f"\nCurrent metrics:")
        print(f"FPS: {fps:.2f}")
        print(f"Average latency: {avg_latency:.4f} seconds")
        print(f"Processed frames: {processed_frames}")
        print("------------------------")


def plot_latency():
    """
    Plot the latency over time.
    """
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot(latencies)
    ax.set_xlabel("Frames")
    ax.set_ylabel("Latency (s)")
    ax.set_title("Latency over time")

    while True:
        line.set_ydata(latencies)
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(1)


def main(args: argparse.Namespace):
    """
    Main function to run the image processing and metrics.

    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    url = f"http://{args.ip}:{args.port}/video"
    frame_delay = 1 / args.fps

    # Start threads
    metrics_thread = threading.Thread(target=show_metrics)
    plot_thread = threading.Thread(target=plot_latency)

    metrics_thread.start()
    plot_thread.start()

    # Start image processing
    process_images(url, args.image_folder, frame_delay)

    # Wait for threads to finish (which in this case will never happen)
    metrics_thread.join()
    plot_thread.join()


if __name__ == "__main__":
    print_colorful_usage()
    print(f"{Fore.WHITE}For more details, use the -h or --help option.\n")

    parser = argparse.ArgumentParser(description="Process images and display metrics.")
    parser.add_argument("--ip", default="172.22.21.43", help="Server IP address")
    parser.add_argument("--port", type=int, default=8081, help="Server port")
    parser.add_argument("--image_folder", required=True, help="Path to the folder containing images")
    parser.add_argument("--fps", type=float, default=30, help="Frames per second to simulate")

    args = parser.parse_args()
    main(args)
