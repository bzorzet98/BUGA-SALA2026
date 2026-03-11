import json
import os
import argparse
import matplotlib.pyplot as plt
from src.utils import configure_path_to_save
from global_config import main_root

################################ CONFIGURATION OF SCRIPTS #########################
parser = argparse.ArgumentParser()
parser.add_argument('--script_config', type=str, default='ResNet18_TrainingConfig_v0')
args = parser.parse_args()

############################### EXTRACT THE CONFIGURATIONS ########################
script_config = args.script_config

evaluation_label = script_config['general_parameters']["label"]
database_path =  Path(main_root) / Path(*script_config['general_parameters']["database_path"])
results_path = Path(main_root) / Path(*script_config['general_parameters']["path_to_save_results"])

############################### INITIALIZE THE TRAINING DATASET ####################
dataset_config = script_config["training_dataset"]
training_dataset = import_class(dataset_config['class_name'], 
                                dataset_config['module_name'])(**dataset_config['params'])
training_dataset.load_data(database_path)

############################## INITIALIZE THE MODEL TO FIT ##########################
model_config = script_config["model"]
model =  import_class(model_config['class_name'], 
                                model_config['module_name'])(**model_config['params'])

############################# FIT THE MODEL #########################################
model.fit_model(traning_dataset)


############################ SAVE MODELS PARAMETERS #################################
model.save_model_parameters()

def save_fish_crops(crops_tensor, metadata, output_folder="output_samples"):
    """
    Guarda los recortes de peces procesados en una carpeta local.
    crops_tensor: [N, 3, 224, 224] (Tensores de PyTorch)
    metadata: Lista de diccionarios con info de cada pez
    """
    # 1. Crear la carpeta si no existe
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Carpeta creada: {output_folder}")

    n_peces = len(crops_tensor)
    if n_peces == 0:
        print("No hay peces para guardar.")
        return

    for i in range(n_peces):
        # 2. Desnormalización y conversión a numpy (H, W, C)
        img_viz = crops_tensor[i].permute(1, 2, 0).cpu().numpy()
        
        # Revertir la normalización de ImageNet para que los colores se vean bien
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_viz = (std * img_viz + mean).clip(0, 1)

        # 3. Generar un nombre de archivo único
        # Ejemplo: LGH020002_frame_1981_fish_0.png
        video_name = metadata[i]['video']
        frame_id = metadata[i]['frame']
        file_name = f"{video_name}_frame_{frame_id}_fish_{i}.png"
        save_path = os.path.join(output_folder, file_name)

        # 4. Guardar usando plt.imsave (más directo para arrays 0-1)
        plt.imsave(save_path, img_viz)
    
    print(f"Se han guardado {n_peces} recortes en '{output_folder}'.")
    



root_path = os.getcwd()
# Try the new class
crops_tensor, meta = ds[0] 

print(f"Video: {meta[0]['video']} - Frame: {meta[0]['frame']}")
print(f"Peces encontrados: {len(crops_tensor)}")

# 2. Visualizamos los primeros 5 peces encontrados en ese frame
save_fish_crops(crops_tensor, meta)

# --- EJEMPLO DE USO ---
# 1. Obtenemos los datos de un frame
crops, meta = ds[0] # Usando la instancia 'ds' de tu FishDataset



root_path = os.getcwd()





# 2. Función de Inferencia Individual por Archivo
def run_inference_per_file(dataset, model, output_dir="Results"):
    """
    Procesa cada frame y guarda un JSON individual con los embeddings.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directorio de salida creado: {output_dir}")

    print(f"Iniciando extracción para {len(dataset)} archivos...")

    with torch.no_grad():
        for i in range(len(dataset)):
            # Obtenemos los datos del Dataset
            crops_tensor, metadata = dataset[i]
            
            # Si el frame no tiene peces, igual creamos un archivo vacío o saltamos
            if crops_tensor.shape[0] == 0:
                continue 

            # Recuperar el nombre base del archivo original (ej: LGH020002_frame_1981)
            # Accedemos a la lista de nombres que definimos en el __init__ del Dataset
            original_filename = dataset.filenames[i] 
            
            # Inferencia del lote de peces en el frame
            crops_tensor = crops_tensor.to(device)
            embeddings = model(crops_tensor).cpu().numpy()

            # Estructura de salida para este archivo específico
            file_results = {
                "filename": f"{original_filename}.jpg",
                "video_id": metadata[0]['video'],
                "frame_id": metadata[0]['frame'],
                "total_objects": len(embeddings),
                "objects": []
            }

            for idx, emb in enumerate(embeddings):
                file_results["objects"].append({
                    "object_index": idx,
                    "class_id": metadata[idx]['class_id'],
                    "bbox_yolo": metadata[idx]['bbox_yolo'], # [cx, cy, w, h]
                    "embedding": emb.tolist()
                })

            # Guardar con el mismo nombre que la imagen pero extensión .json
            save_path = os.path.join(output_dir, f"{original_filename}.json")
            with open(save_path, 'w') as f:
                json.dump(file_results, f, indent=4)

            if i % 10 == 0:
                print(f"Procesado: {original_filename}.json")

    print(f"--- Proceso completado. Archivos guardados en {output_dir} ---")
    
    
# Try the new class
ds = FishDataset(root_dir=os.path.join(root_path,"Dataset" ), transform=minimal_transform)

run_inference_per_file(ds, model, output_dir="Datasets/embeddings")