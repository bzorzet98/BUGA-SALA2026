import os
import r2_download as hd
from dotenv import load_dotenv

# 1. Cargar las credenciales del archivo enviado por Rudy
# Asegúrate de que el archivo 'participant-download.env' esté en esta misma carpeta
load_dotenv("participant-download.env")

def descargar_porcion_bruv():
    try:
        # 2. Configurar el cliente S3 para Cloudflare R2
        print("Conectando con Cloudflare R2...")
        client = hd.get_s3_client()
        
        # 3. Cargar el manifiesto de datos
        bucket_name = os.environ.get("R2_BUCKET", "sala-2026-hackathon-data")
        manifest = hd.load_manifest(
            bucket=bucket_name, 
            s3_client=client, 
            cache_path="manifest.json"
        )
        
        # 4. Descargar solo una porción (un sub-video de ~4GB)
        # Usamos el tag 'vid2-sub02' para no llenar el disco duro
        dataset_target = "bruv-videos"
        sub_video_tag = ["vid2-sub02"]
        
        print(f"Iniciando descarga de {dataset_target} (Fragmento: {sub_video_tag})...")
        stats = hd.download_dataset(
            manifest, 
            dataset_name=dataset_target, 
            tags=sub_video_tag
        )
        
        print("\n" + "="*30)
        print("¡Descarga completada con éxito!")
        print(f"Los datos están en: {os.path.join(os.getcwd(), 'hackathon_data', dataset_target)}")
        print("="*30)

    except Exception as e:
        print(f"\nERROR: {e}")
        print("Verifica que 'participant-download.env' tenga las llaves correctas y que r2_download.py esté presente.")

if __name__ == "__main__":
    descargar_porcion_bruv()