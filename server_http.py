import argparse
import os
from flask import Flask, render_template, request, Response
import base64
import subprocess
import PIL
from src.safear_service import SafeARService


###################################### COMENTARIOS  #################################################
#     meter a função de ofuscaçaõ na imagem que recebo
#   
#   
#   
#   
######################################################################################################

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('src/templates/index.html')

@app.route('/video', methods=['GET', 'POST'])
def camera_stream():
    
    # Receber os dados da imagem em base 64
    image_data = request.form['imageData']
    image_data += "=" * ((4 - len(image_data) % 4) % 4)
    # Processar os dados da imagem como desejar-mos
    # dar decode à base64 temos assim uma string 
    #processed_image_data = base64.b64decode(image_data)
    #print(processed_image_data)
    
    # Initialize the SafeARService
    safe_ar_service = SafeARService()

    # Configure the SafeARService with the desired model number and obfuscation policies
    safe_ar_service.configure(model_number=0, obfuscation_policies={0: "blurring", 1: "blurring"})

    # Auxiliary function to read the base64 image from a file
    #image_base64 = safe_ar_service.read_base64_image("test_samples/images/image.txt")

    # Image Obfuscation using the SafeARService
    processed_frame_bytes = safe_ar_service.process_frame(image_data)

    # Auxiliary function to save the processed frame to a file
    safe_ar_service.save_processed_frame(processed_frame_bytes, "outputs/img_out.png")
    #para ver se os dados vêm em base 64 ou não, pois se vierem não vale a pena estar a codificá-los novamente
    print(processed_frame_bytes)

    # codificar os dados da imagem processada como base64
    processed_image_base64 = base64.b64encode(processed_frame_bytes)
    
    # provavelmente passarei "processed_image_base64" mas se sair do modulo ja em base64 ent passarei processed_frame_bytes
    return processed_image_base64

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app")
    parser.add_argument("--port", default=8080, type=int, help="port number") 
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port) 