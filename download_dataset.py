import os
import r2_download as hd
from dotenv import load_dotenv

# Especifica el nombre exacto del archivo que descargaste
load_dotenv("participant-download.env") 

# Ahora sí, esto ya encontrará las variables R2_ENDPOINT, etc.
client = hd.get_s3_client()

manifest = hd.load_manifest(
    bucket=os.environ["R2_BUCKET"], 
    s3_client=client, 
    cache_path="manifest.json"
)

# Descargar los datos de clima
hd.download_dataset(manifest, dataset_name="precipitation-nowcasting")