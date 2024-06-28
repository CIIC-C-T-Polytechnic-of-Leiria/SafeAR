import base64
import threading
import time

import matplotlib.pyplot as plt
import requests

# Configurações
IP = "172.22.21.43"
PORT = 8081
URL = f"http://{IP}:{PORT}/video"

# Variáveis globais para métricas
latencies = []
processed_frames = 0
start_time = time.time()


def send_request(image_path):
    global processed_frames

    # Carregar e codificar a imagem
    with open(image_path, "rb") as image_file:
        img_base64 = base64.b64encode(image_file.read()).decode("utf-8")

    # Enviar pedido ao servidor
    start_time = time.time()
    response = requests.post(
        URL,
        headers={"Content-Type": "multipart/form-data"},
        data={"imageData": img_base64},
    )
    end_time = time.time()

    # Calcular e armazenar latência
    latency = end_time - start_time
    latencies.append(latency)

    # Processar resposta
    if response.status_code == 200:
        processed_frames += 1
        print(f"Frame {processed_frames} processado com sucesso.")
    else:
        print(f"Erro no processamento do frame: {response.status_code}")

    return latency


def process_images(image_folder):
    import os

    image_files = [
        f for f in os.listdir(image_folder) if f.endswith((".png", ".jpg", ".jpeg"))
    ]

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        latency = send_request(image_path)
        print(f"Imagem: {image_file}, Latência: {latency:.4f} segundos")
        time.sleep(0.033)  # Simula 30 FPS


def show_metrics():
    while True:
        time.sleep(5)  # Atualizar a cada 5 segundos
        current_time = time.time()
        elapsed_time = current_time - start_time
        fps = processed_frames / elapsed_time if elapsed_time > 0 else 0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0

        print(f"\nMétricas atuais:")
        print(f"FPS: {fps:.2f}")
        print(f"Latência média: {avg_latency:.4f} segundos")
        print(f"Frames processados: {processed_frames}")
        print("------------------------")


def plot_latency():
    plt.ion()
    fig, ax = plt.subplots()
    (line,) = ax.plot(latencies)
    ax.set_xlabel("Frames")
    ax.set_ylabel("Latência (s)")
    ax.set_title("Latência ao longo do tempo")

    while True:
        line.set_ydata(latencies)
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(1)


if __name__ == "__main__":
    image_folder = "caminho/para/sua/pasta/de/imagens"  # Atualize este caminho

    # Iniciar threads
    metrics_thread = threading.Thread(target=show_metrics)
    plot_thread = threading.Thread(target=plot_latency)

    metrics_thread.start()
    plot_thread.start()

    # Iniciar o processamento de imagens
    process_images(image_folder)

    # Aguardar as threads terminarem (que neste caso nunca acontecerá)
    metrics_thread.join()
    plot_thread.join()
