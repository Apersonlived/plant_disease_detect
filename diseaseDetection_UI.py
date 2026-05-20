import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms, models
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Tomato Disease Detector",
    page_icon="🍅",
    layout="centered"
)

# Constants
CLASS_NAMES = [
    "Early Blight",
    "Late Blight",
    "Healthy",
    "Yellow Leaf Curl Virus",
]

# Treatment plans shown after classification
TREATMENT_INFO = {
    "Early Blight": {
        "description": (
            "Early Blight is caused by the fungus Alternaria solani. "
            "It appears as dark brown concentric rings (target-like spots) "
            "on lower, older leaves first, then spreads upward."
        ),
        "treatment": [
            "Remove and destroy infected leaves immediately.",
            "Apply copper-based fungicides (e.g. Copper Oxychloride) every 7–10 days.",
            "Avoid overhead watering — water at the base of the plant.",
            "Ensure adequate plant spacing for air circulation.",
            "Rotate crops: do not plant tomatoes in the same soil for a long time.",
        ],
        "severity": "Moderate",
        "color": "orange",
    },
    "Late Blight": {
        "description": (
            "Late Blight is caused by Phytophthora infestans — the same pathogen "
            "responsible for the Irish Potato Famine. It spreads rapidly in cool, "
            "wet conditions and can destroy an entire crop within days."
        ),
        "treatment": [
            "Act immediately — Late Blight spreads very rapidly.",
            "Remove and bag all infected plant material — do not compost.",
            "Apply systemic fungicides containing Metalaxyl or Mancozeb.",
            "Avoid working with plants when wet to prevent spread.",
            "In severe cases, remove entire affected plants to protect neighbours.",
            "Monitor remaining plants daily.",
        ],
        "severity": "Severe — Act Immediately",
        "color": "red",
    },
    "Healthy": {
        "description": (
            "The leaf appears healthy with no visible signs of disease, "
            "fungal infection, or viral symptoms."
        ),
        "treatment": [
            "Continue regular watering and fertilisation schedule.",
            "Monitor plants weekly for early signs of disease.",
            "Maintain good air circulation between plants.",
            "Apply preventative copper fungicide spray monthly during wet seasons.",
        ],
        "severity": "None",
        "color": "green",
    },
    "Yellow Leaf Curl Virus": {
        "description": (
            "Tomato Yellow Leaf Curl Virus (TYLCV) is transmitted by whiteflies "
            "(Bemisia tabaci). Infected leaves curl upward and turn yellow. "
            "There is no cure once a plant is infected."
        ),
        "treatment": [
            "There is no chemical cure: infected plants cannot be saved.",
            "Remove and destroy infected plants immediately to prevent spread.",
            "Control whitefly populations with insecticidal soap or neem oil.",
            "Use reflective mulches to deter whiteflies.",
            "Plant resistant tomato varieties in future seasons.",
            "Install insect-proof netting over seedlings.",
        ],
        "severity": "Severe — No Cure",
        "color": "red",
    },
}

# Model file paths — place .pth files in the same folder as app.py
MODEL_PATHS = {
    "Baseline CNN"        : "./baseline_cnn_best.pth",
    "Improved CNN"        : "./improved_cnn_best.pth",
    "Fine-Tuned ResNet50" : "./finetuned_resnet50_best.pth",
}

NUM_CLASSES = len(CLASS_NAMES)
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Model definitions
class BaselineCNN(nn.Module):
    """4-block CNN trained from scratch. Input: 224x224."""
    def __init__(self, num_classes=4):
        super(BaselineCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x


class ImprovedCNN(nn.Module):
    """5-block CNN with augmentation and regularisation. Input: 128x128."""
    def __init__(self, num_classes=4):
        super(ImprovedCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # double conv
            nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 2 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x


def build_resnet50(num_classes=4):
    """ImageNet pretrained ResNet50 fine-tuned for tomato disease. Input: 224x224."""
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# Preprocessing transforms
TRANSFORMS = {
    "Baseline CNN": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    "Improved CNN": transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    "Fine-Tuned ResNet50": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}


# Model loader
@st.cache_resource
def load_model(model_name):
    path = MODEL_PATHS[model_name]

    if not os.path.exists(path):
        return None, (
            f"Model file not found: `{path}`\n\n"
            f"Make sure `{os.path.basename(path)}` is in the same "
            f"folder as `app.py`."
        )

    if model_name == "Baseline CNN":
        model = BaselineCNN(num_classes=NUM_CLASSES)
    elif model_name == "Improved CNN":
        model = ImprovedCNN(num_classes=NUM_CLASSES)
    else:
        model = build_resnet50(num_classes=NUM_CLASSES)

    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model, None


# Inference
def predict(image: Image.Image, model_name: str):
    """
    Returns:
        top3        : list of (class_name, confidence_%) sorted by confidence
        error       : error string or None
    """
    model, error = load_model(model_name)
    if error:
        return None, error

    transform     = TRANSFORMS[model_name]
    img_tensor    = transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs       = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()

    # Build top-3 sorted by confidence descending
    top3_idx = np.argsort(probabilities)[::-1][:3]
    top3 = [
        (CLASS_NAMES[i], float(probabilities[i]) * 100)
        for i in top3_idx
    ]

    return top3, None


# UI
st.title("🍅 Tomato Disease Detector")
st.markdown(
    "Upload a tomato leaf image, select a model, and get an instant "
    "disease classification with treatment recommendations."
)
st.markdown("---")

# Model selector
model_name = st.selectbox(
    "Select Model",
    options=list(MODEL_PATHS.keys()),
)

descriptions = {
    "Baseline CNN": (
        "🔹 **Baseline CNN** — 4-block CNN trained from scratch "
    ),
    "Improved CNN": (
        "🔸 **Improved CNN** — 5-block CNN with augmentation, class-weighted "
    ),
    "Fine-Tuned ResNet50": (
        "⭐ **Fine-Tuned ResNet50** — ImageNet pretrained, two-phase fine-tuning "
        "Best Preformance"
    ),
}
st.info(descriptions[model_name])
st.markdown("---")

# Image uploader 
uploaded_file = st.file_uploader(
    "Upload a tomato leaf image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, width=300)

    with col2:
        st.subheader("Classification Result")

        with st.spinner(f"Running {model_name}..."):
            top3, error = predict(image, model_name)

        if error:
            st.error(f"❌ {error}")

        else:
            predicted_class, confidence = top3[0]
            info = TREATMENT_INFO[predicted_class]

            # Primary prediction
            if predicted_class == "Healthy":
                st.success(f"✅  **{predicted_class}**")
            elif info["severity"].startswith("Severe"):
                st.error(f"🚨  **{predicted_class}**")
            else:
                st.warning(f"⚠️  **{predicted_class}**")

            st.metric(label="Confidence", value=f"{confidence:.2f}%")
            st.progress(int(confidence))

            if confidence < 60.0:
                st.warning(
                    "Low confidence — the model is uncertain. "
                    "Try a clearer, well-lit close-up of the leaf, "
                    "or switch to Fine-Tuned ResNet50."
                )

    st.markdown("---")

    # Top 3 predictions
    st.subheader("Top 3 Predictions")

    for rank, (cls_name, conf) in enumerate(top3):
        col_label, col_bar, col_pct = st.columns([2, 4, 1])

        with col_label:
            if rank == 0:
                st.markdown(f"**{cls_name}** 🥇")
            else:
                st.markdown(f"{cls_name}")

        with col_bar:
            st.progress(int(conf))

        with col_pct:
            st.markdown(f"**{conf:.1f}%**")

    st.markdown("---")

    # Disease info and treatment plan
    predicted_class, _ = top3[0]
    info = TREATMENT_INFO[predicted_class]

    st.subheader("Disease Information")
    st.markdown(f"**Severity:** `{info['severity']}`")
    st.markdown(info["description"])

    st.subheader("Recommended Treatment")
    for step, tip in enumerate(info["treatment"], start=1):
        st.markdown(f"{step}. {tip}")

    st.markdown("---")

    # Footer caption
    st.caption(
        f"Tomato Disease Prediction · "
        f"Model: **{model_name}** · "
        f"Device: **{str(DEVICE).upper()}** · "
        f"Classes: {NUM_CLASSES}"
    )

# Page footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray; font-size:12px;'>"
    "Tomato Disease Detection · Deep Learning Project · "
    "Using PlantVillage & PlantDoc Dataset"
    "</div>",
    unsafe_allow_html=True
)