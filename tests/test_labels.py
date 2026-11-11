import json

from plantdx.labels import INFO_FILE, disease_info, is_healthy, pretty_label

PLANTVILLAGE_CLASSES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight", "Potato___Late_blight",
    "Potato___healthy", "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy", "Tomato___Bacterial_spot", "Tomato___Early_blight",
    "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy",
]


def test_pretty_labels():
    assert pretty_label("Corn_(maize)___Common_rust_") == "Corn (maize) – Common rust"
    assert pretty_label("Pepper,_bell___Bacterial_spot") == "Pepper, bell – Bacterial spot"
    assert pretty_label("Tomato___healthy") == "Tomato – healthy"


def test_every_plantvillage_class_has_info():
    assert len(PLANTVILLAGE_CLASSES) == 38
    known = json.loads(INFO_FILE.read_text(encoding="utf-8"))
    for cls in PLANTVILLAGE_CLASSES:
        assert is_healthy(cls) or cls in known, cls
        info = disease_info(cls)
        assert info["type"] and info["about"] and len(info["actions"]) >= 2


def test_unknown_class_falls_back():
    info = disease_info("Mango___Anthracnose")
    assert "Anthracnose" in info["about"] and info["actions"]
