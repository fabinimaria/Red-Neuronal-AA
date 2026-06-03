import os
from pathlib import Path

import streamlit as st
import torch
from streamlit_drawable_canvas import st_canvas

from model import MNISTNet
from train import get_data_loaders, get_device, load_saved_model, train_model
from utils import predict_digit, preprocess_canvas


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "saved_model.pth"
DATA_DIR = BASE_DIR / "data"
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".matplotlib"))

import matplotlib.pyplot as plt


st.set_page_config(
    page_title="MNIST Visual NN",
    layout="wide",
)


st.markdown(
    """
    <style>
    :root {
        --ink: #111318;
        --muted: #667085;
        --line: #d9dee8;
        --blue: #2f5bea;
        --green: #17a56b;
        --red: #e5484d;
        --amber: #f0a202;
        --panel: #ffffff;
        --page: #f6f8fb;
    }

    .stApp {
        background:
            radial-gradient(circle at 12% 0%, rgba(47, 91, 234, 0.11), transparent 26rem),
            radial-gradient(circle at 92% 8%, rgba(23, 165, 107, 0.10), transparent 22rem),
            linear-gradient(180deg, #ffffff 0, var(--page) 23rem),
            var(--page);
    }

    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2rem;
        max-width: 1220px;
    }

    h1, h2, h3 {
        color: var(--ink);
        letter-spacing: 0;
    }

    h3 {
        margin-top: 1.7rem;
    }

    .nn-topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        border: 1px solid var(--line);
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.86);
        padding: 0.8rem 1rem;
        margin-bottom: 1.1rem;
        box-shadow: 0 8px 24px rgba(17, 19, 24, 0.05);
    }

    .nn-brand {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        font-weight: 800;
        color: var(--ink);
    }

    .nn-logo {
        width: 2rem;
        height: 2rem;
        border-radius: 8px;
        display: grid;
        place-items: center;
        color: #ffffff;
        background: var(--ink);
        font-size: 0.95rem;
        font-weight: 800;
    }

    .nn-nav {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        justify-content: flex-end;
    }

    .nn-nav span,
    .step-pill,
    .legend-chip {
        border: 1px solid var(--line);
        border-radius: 999px;
        background: #ffffff;
        color: var(--muted);
        font-size: 0.82rem;
        font-weight: 650;
        padding: 0.35rem 0.62rem;
    }

    .hero {
        border: 1px solid var(--line);
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.92);
        padding: 1.3rem 1.35rem;
        margin-bottom: 1rem;
        box-shadow: 0 12px 38px rgba(17, 19, 24, 0.06);
    }

    .eyebrow {
        color: var(--blue);
        font-size: 0.78rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.25rem;
    }

    .hero h1 {
        font-size: clamp(2rem, 4vw, 3.3rem);
        line-height: 1.02;
        margin: 0;
    }

    .hero p {
        color: var(--muted);
        font-size: 1.02rem;
        max-width: 58rem;
        margin: 0.7rem 0 0;
    }

    .step-strip,
    .legend-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 0.95rem;
    }

    .step-pill strong {
        color: var(--ink);
        margin-right: 0.3rem;
    }

    .section-label {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        color: var(--ink);
        font-size: 1.08rem;
        font-weight: 800;
        margin: 1.4rem 0 0.6rem;
    }

    .section-number {
        width: 1.55rem;
        height: 1.55rem;
        border-radius: 8px;
        display: inline-grid;
        place-items: center;
        color: #ffffff;
        background: var(--ink);
        font-size: 0.82rem;
    }

    .legend-dot {
        display: inline-block;
        width: 0.62rem;
        height: 0.62rem;
        border-radius: 999px;
        margin-right: 0.35rem;
        vertical-align: -0.02rem;
    }

    [data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.85rem 1rem;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--line);
        border-radius: 8px;
        box-shadow: 0 10px 30px rgba(17, 19, 24, 0.05);
    }

    .stButton > button {
        border-radius: 8px;
        font-weight: 650;
    }

    .stDataFrame {
        border: 1px solid var(--line);
        border-radius: 8px;
        overflow: hidden;
    }

    canvas {
        border-radius: 8px !important;
        box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.14);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def initialize_model():
    device = get_device()
    model = MNISTNet()
    model_loaded = load_saved_model(model, str(MODEL_PATH), device)
    model.to(device)
    model.eval()
    return model, device, model_loaded


@st.cache_resource
def load_data(batch_size: int):
    return get_data_loaders(str(DATA_DIR), batch_size=batch_size)


def reset_model():
    model = MNISTNet()
    model.to(st.session_state.device)
    model.eval()
    st.session_state.model = model
    st.session_state.model_loaded = False
    st.session_state.history = []
    st.session_state.prediction = None
    if MODEL_PATH.exists():
        MODEL_PATH.unlink()


def draw_architecture():
    layers = [
        ("Input\n784", 0, 9, "#111318"),
        ("Hidden\n128", 1, 6, "#2f5bea"),
        ("Hidden\n64", 2, 5, "#17a56b"),
        ("Output\n10", 3, 4, "#e5484d"),
    ]

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.set_facecolor("#fbfcff")
    fig.patch.set_facecolor("#fbfcff")

    for index in range(len(layers) - 1):
        _, x1, count1, _ = layers[index]
        _, x2, count2, _ = layers[index + 1]
        offset1 = (9 - count1) / 2
        offset2 = (9 - count2) / 2
        for y1 in range(count1):
            for y2 in range(count2):
                ax.plot(
                    [x1 + 0.13, x2 - 0.13],
                    [y1 + offset1, y2 + offset2],
                    color="#c8d1de",
                    linewidth=0.55,
                    alpha=0.75,
                    zorder=1,
                )

    for _, x, count, color in layers:
        offset = (9 - count) / 2
        for y in range(count):
            circle = plt.Circle(
                (x, y + offset),
                0.145,
                color=color,
                ec="#ffffff",
                linewidth=1.4,
                zorder=2,
            )
            ax.add_patch(circle)

    for label, x, _, color in layers:
        ax.text(
            x,
            -0.65,
            label,
            ha="center",
            va="top",
            fontsize=11,
            color=color,
            fontweight="bold",
        )

    ax.set_xlim(-0.45, 3.45)
    ax.set_ylim(-1.2, 9.5)
    ax.axis("off")
    return fig


def plot_training_history(history):
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    train_accuracy = [row["train_accuracy"] for row in history]
    test_accuracy = [row["test_accuracy"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7))
    fig.patch.set_facecolor("#fbfcff")

    axes[0].plot(epochs, train_loss, marker="o", color="#e5484d", linewidth=2)
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.22)

    axes[1].plot(epochs, train_accuracy, marker="o", label="Train", color="#2f5bea", linewidth=2)
    axes[1].plot(epochs, test_accuracy, marker="o", label="Test", color="#17a56b", linewidth=2)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.22)

    for ax in axes:
        ax.set_facecolor("#fbfcff")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


def plot_probabilities(probabilities):
    fig, ax = plt.subplots(figsize=(8.4, 3.4))
    fig.patch.set_facecolor("#fbfcff")
    digits = list(range(10))
    colors = ["#dbe4ff"] * 10
    colors[int(probabilities.argmax())] = "#2f5bea"

    ax.bar(digits, probabilities, color=colors, edgecolor="#2444a7", linewidth=0.6)
    ax.set_xticks(digits)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Dígito")
    ax.set_ylabel("Probabilidad")
    ax.set_title("Probabilidades softmax")
    ax.grid(axis="y", alpha=0.22)
    ax.set_facecolor("#fbfcff")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


if "model" not in st.session_state:
    model, device, model_loaded = initialize_model()
    st.session_state.model = model
    st.session_state.device = device
    st.session_state.model_loaded = model_loaded

if "history" not in st.session_state:
    st.session_state.history = []

if "prediction" not in st.session_state:
    st.session_state.prediction = None

model = st.session_state.model
device = st.session_state.device

st.markdown(
    """
    <div class="nn-topbar">
        <div class="nn-brand">
            <div class="nn-logo">NN</div>
            <span>MNIST Visual NN</span>
        </div>
        <div class="nn-nav">
            <span>Dataset: MNIST</span>
            <span>MLP 784 -> 128 -> 64 -> 10</span>
            <span>PyTorch + Streamlit</span>
        </div>
    </div>
    <section class="hero">
        <div class="eyebrow">Interactive neural network lab</div>
        <h1>Train, draw, and inspect a digit classifier.</h1>
        <p>
            Una experiencia visual inspirada en nn-visual: entrenás la red,
            probás tu propio trazo y ves cómo el modelo reparte probabilidad
            entre las diez clases.
        </p>
        <div class="step-strip">
            <span class="step-pill"><strong>1</strong> Train model</span>
            <span class="step-pill"><strong>2</strong> Draw digit</span>
            <span class="step-pill"><strong>3</strong> Predict</span>
            <span class="step-pill"><strong>4</strong> Inspect architecture</span>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

status = "saved_model.pth cargado" if st.session_state.model_loaded else "modelo nuevo"
top_metrics = st.columns(4)
top_metrics[0].metric("Estado", status)
top_metrics[1].metric("Dispositivo", str(device))
top_metrics[2].metric("Capas", "784 / 128 / 64 / 10")
top_metrics[3].metric("Epochs corridas", len(st.session_state.history))

st.markdown(
    '<div class="section-label"><span class="section-number">1</span>Entrenar la red</div>',
    unsafe_allow_html=True,
)
train_tab = st.container()
st.markdown(
    '<div class="section-label"><span class="section-number">2</span>Dibujar y predecir</div>',
    unsafe_allow_html=True,
)
predict_tab = st.container()
st.markdown(
    '<div class="section-label"><span class="section-number">3</span>Inspeccionar arquitectura</div>',
    unsafe_allow_html=True,
)
architecture_tab = st.container()

with train_tab:
    controls, results = st.columns([0.34, 0.66])

    with controls:
        with st.container(border=True):
            st.subheader("Configure network")
            epochs = st.slider("Epochs", min_value=1, max_value=25, value=5, step=1)
            batch_size = st.selectbox("Batch size", options=[64, 128, 256], index=1)
            learning_rate = st.selectbox(
                "Learning rate",
                options=[0.0005, 0.001, 0.002],
                index=1,
                format_func=lambda value: f"{value:g}",
            )
            weight_decay = st.selectbox(
                "Weight decay",
                options=[0.0, 0.0001, 0.0005],
                index=1,
                format_func=lambda value: f"{value:g}",
            )

            train_clicked = st.button("Train", type="primary", use_container_width=True)
            reset_clicked = st.button("Reset model", use_container_width=True)

        if reset_clicked:
            reset_model()
            st.success("Modelo reiniciado y archivo guardado eliminado.")
            st.rerun()

    with results:
        with st.container(border=True):
            st.subheader("Training monitor")

            if train_clicked:
                train_loader, test_loader = load_data(batch_size)
                progress_bar = st.progress(0)
                live_metrics = st.empty()

                def update_progress(epoch, total_epochs, loss, train_acc, test_acc):
                    progress_bar.progress(epoch / total_epochs)
                    live_metrics.metric(
                        label=f"Epoch {epoch}/{total_epochs}",
                        value=f"Loss {loss:.4f}",
                        delta=f"Train {train_acc:.2%} | Test {test_acc:.2%}",
                    )

                try:
                    with st.spinner("Entrenando la red neuronal..."):
                        history = train_model(
                            model=model,
                            epochs=epochs,
                            train_loader=train_loader,
                            test_loader=test_loader,
                            device=device,
                            learning_rate=learning_rate,
                            weight_decay=weight_decay,
                            progress_callback=update_progress,
                            save_path=str(MODEL_PATH),
                        )

                    st.session_state.history = history
                    st.session_state.model_loaded = True
                    st.success(f"Entrenamiento terminado. Modelo guardado en {MODEL_PATH.name}.")
                except RuntimeError as error:
                    st.error(
                        "No se pudo completar el entrenamiento en este entorno. "
                        "La app sigue funcionando para dibujar y predecir con el modelo guardado."
                    )
                    st.exception(error)

            if st.session_state.history:
                last = st.session_state.history[-1]
                metric_cols = st.columns(3)
                metric_cols[0].metric("Training loss", f"{last['train_loss']:.4f}")
                metric_cols[1].metric("Training accuracy", f"{last['train_accuracy']:.2%}")
                metric_cols[2].metric("Test accuracy", f"{last['test_accuracy']:.2%}")

                st.pyplot(plot_training_history(st.session_state.history), clear_figure=True)
                st.dataframe(
                    [
                        {
                            "epoch": row["epoch"],
                            "training loss": round(row["train_loss"], 4),
                            "training accuracy": f"{row['train_accuracy']:.2%}",
                            "test loss": round(row["test_loss"], 4),
                            "test accuracy": f"{row['test_accuracy']:.2%}",
                            "lr": f"{row['learning_rate']:.6f}",
                        }
                        for row in st.session_state.history
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("Entrená el modelo para ver loss, accuracy y la evolución por epoch.")

with predict_tab:
    canvas_col, pred_col = st.columns([0.48, 0.52])

    with canvas_col:
        with st.container(border=True):
            st.subheader("Input canvas")
            canvas_result = st_canvas(
                fill_color="rgba(0, 0, 0, 0)",
                stroke_width=18,
                stroke_color="#ffffff",
                background_color="#000000",
                height=300,
                width=300,
                drawing_mode="freedraw",
                display_toolbar=True,
                key="mnist_canvas",
            )
            predict_clicked = st.button("Predict", type="primary", use_container_width=True)

    with pred_col:
        with st.container(border=True):
            st.subheader("Model output")

            if canvas_result.image_data is not None:
                image_tensor, preview = preprocess_canvas(canvas_result.image_data)
                preview_cols = st.columns([0.35, 0.65])
                preview_cols[0].image(
                    preview.resize((150, 150)),
                    caption="28x28 procesado",
                    width=150,
                )
                preview_cols[1].code(f"Tensor shape: {tuple(image_tensor.shape)}")

                if predict_clicked:
                    predicted_digit, probabilities = predict_digit(model, image_tensor, device)
                    st.session_state.prediction = {
                        "digit": predicted_digit,
                        "probabilities": probabilities,
                    }

                if st.session_state.prediction is not None:
                    predicted_digit = st.session_state.prediction["digit"]
                    probabilities = st.session_state.prediction["probabilities"]
                    confidence = probabilities[predicted_digit]

                    pred_metrics = st.columns(2)
                    pred_metrics[0].metric("Dígito", predicted_digit)
                    pred_metrics[1].metric("Confianza", f"{confidence:.2%}")

                    st.pyplot(plot_probabilities(probabilities), clear_figure=True)
                    st.dataframe(
                        [
                            {"digit": digit, "probability": f"{probability:.2%}"}
                            for digit, probability in enumerate(probabilities)
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("Dibujá un número y apretá Predict.")
            else:
                st.info("El canvas está listo para dibujar un dígito.")

with architecture_tab:
    arch_col, detail_col = st.columns([0.62, 0.38])
    with arch_col:
        with st.container(border=True):
            st.subheader("Network graph")
            st.pyplot(draw_architecture(), clear_figure=True)
            st.markdown(
                """
                <div class="legend-row">
                    <span class="legend-chip"><span class="legend-dot" style="background:#111318"></span>Input</span>
                    <span class="legend-chip"><span class="legend-dot" style="background:#2f5bea"></span>Hidden 1</span>
                    <span class="legend-chip"><span class="legend-dot" style="background:#17a56b"></span>Hidden 2</span>
                    <span class="legend-chip"><span class="legend-dot" style="background:#e5484d"></span>Output</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with detail_col:
        with st.container(border=True):
            st.subheader("Layer stack")
            st.code(
                """MNISTNet(
  Flatten()
  Linear(784, 128) + ReLU
  Linear(128, 64) + ReLU
  Linear(64, 10)
)""",
                language="text",
            )
            st.write("El entrenamiento usa augmentations suaves para parecerse más a dígitos dibujados a mano.")
