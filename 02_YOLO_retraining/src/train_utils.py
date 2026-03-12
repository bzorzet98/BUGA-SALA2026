from ultralytics import YOLO


def create_model(weights_path: str):
    return YOLO(weights_path)


def train_model(model, yaml_path, train_cfg: dict, project_dir: str):
    results = model.train(
        data=str(yaml_path),
        epochs=train_cfg["epochs"],
        imgsz=train_cfg["imgsz"],
        batch=train_cfg["batch"],
        device=train_cfg["device"],
        pretrained=train_cfg["pretrained"],
        project=project_dir,
        name=train_cfg["run_name"],
        cache=train_cfg["cache"],
        patience=train_cfg["patience"],
        degrees=train_cfg["degrees"],
        translate=train_cfg["translate"],
        scale=train_cfg["scale"],
        fliplr=train_cfg["fliplr"],
        mosaic=train_cfg["mosaic"],
        mixup=train_cfg["mixup"],
        copy_paste=train_cfg["copy_paste"]
    )

    return results


def validate_model(model, yaml_path, val_cfg: dict):
    metrics = model.val(
        data=str(yaml_path),
        split=val_cfg["split"],
        imgsz=val_cfg["imgsz"],
        device=val_cfg["device"]
    )

    return metrics