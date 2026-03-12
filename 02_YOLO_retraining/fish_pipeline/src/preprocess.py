import cv2


def preprocess_clahe_rgb(image_rgb):
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l2 = clahe.apply(l)

    lab2 = cv2.merge([l2, a, b])
    out_bgr = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)

    return out_rgb


def preprocess_clahe_bilateral(image_rgb):
    x = preprocess_clahe_rgb(image_rgb)

    x_bgr = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)

    x_bgr = cv2.bilateralFilter(
        x_bgr,
        d=7,
        sigmaColor=50,
        sigmaSpace=50
    )

    return cv2.cvtColor(x_bgr, cv2.COLOR_BGR2RGB)


def apply_preprocessing(frame_bgr, preprocess_cfg):
    method = preprocess_cfg.get("method", "none")

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    if method == "clahe_rgb":
        processed = preprocess_clahe_rgb(frame_rgb)
    elif method == "clahe_bilateral":
        processed = preprocess_clahe_bilateral(frame_rgb)
    elif method == "none":
        processed = frame_rgb
    else:
        raise ValueError(f"Unknown preprocessing method: {method}")

    return cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)