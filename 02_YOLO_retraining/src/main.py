from pathlib import Path

from config_loader import load_config
from dataset_utils import (
    get_images,
    check_missing_labels,
    build_image_class_mapping,
    count_instances,
)
from split_utils import (
    make_dataset_folders,
    select_train_val_test_split,
    copy_split,
)
from yaml_utils import create_data_yaml
from train_utils import (
    create_model,
    train_model,
    validate_model,
)


def main(config_path: str = "../Configs/config.json"):
    config = load_config(config_path)

    paths_cfg = config["paths"]
    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    train_cfg = config["train"]
    val_cfg = config["val"]

    images_dir = Path(paths_cfg["images_dir"])
    labels_dir = Path(paths_cfg["labels_dir"])
    dataset_root = Path(paths_cfg["dataset_root"])
    project_dir = paths_cfg["project_dir"]

    images = get_images(images_dir, dataset_cfg["image_extensions"])
    print("Número de imágenes encontradas:", len(images))

    missing = check_missing_labels(images, labels_dir)
    for name in missing:
        print("FALTA LABEL:", name)

    img_to_classes, class_presence = build_image_class_mapping(images, labels_dir)

    print("\nPresencia por clase (número de imágenes donde aparece):")
    print(dict(class_presence))

    make_dataset_folders(dataset_root)
    print("\nCarpetas creadas en:", dataset_root)

    train_images, val_images, test_images, split_info = select_train_val_test_split(
        images=images,
        img_to_classes=img_to_classes,
        class_presence=class_presence,
        random_seed=dataset_cfg["random_seed"],
        min_images=dataset_cfg["min_images"],
    )

    print("\nImágenes con >=2 clases:")
    for p in split_info["multi_class_images"]:
        print(" ", p.name, "->", sorted(set(img_to_classes[p.name])))

    print("\nImágenes con 1 clase:")
    for p in split_info["single_class_images"]:
        print(" ", p.name, "->", sorted(set(img_to_classes[p.name])))

    print("\nVALIDACIÓN:")
    for p in val_images:
        print(" ", p.name, "-> clases:", sorted(set(img_to_classes[p.name])))

    print("\nTRAIN:")
    for p in train_images:
        print(" ", p.name, "-> clases:", sorted(set(img_to_classes[p.name])))

    print("\nTotal train:", len(train_images))
    print("Total val:", len(val_images))

    copy_split(train_images, val_images,test_images, labels_dir, dataset_root)
    print("\nSplit copiado correctamente.")

    train_counts = count_instances(dataset_root / "labels/train")
    val_counts = count_instances(dataset_root / "labels/val")

    print("\nInstancias en TRAIN:")
    print(dict(train_counts))

    print("\nInstancias en VAL:")
    print(dict(val_counts))

    class_names = dataset_cfg["class_names"]
    print("\nNúmero de clases:", len(class_names))

    yaml_path = create_data_yaml(dataset_root, class_names)
    print("\ndata.yaml creado en:", yaml_path)
    print(yaml_path.read_text())

    model = create_model(model_cfg["weights"])

    print("\nIniciando entrenamiento...")
    train_results = train_model(
        model=model,
        yaml_path=yaml_path,
        train_cfg=train_cfg,
        project_dir=project_dir
    )

    print("\nEntrenamiento finalizado.")
    print(train_results)

    print("\nIniciando validación...")
    metrics = validate_model(model, yaml_path, val_cfg)

    print("\nResultados de validación:")
    print("mAP50-95:", metrics.box.map)
    print("mAP50:", metrics.box.map50)
    print("mAP75:", metrics.box.map75)


if __name__ == "__main__":
    main()