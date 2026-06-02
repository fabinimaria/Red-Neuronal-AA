# MNIST Visual Neural Network

App web en Python con Streamlit para entrenar una red neuronal simple sobre MNIST, dibujar un número en un canvas y ver la predicción con probabilidades por dígito.

## Estructura

```text
mnist_visual_nn/
|-- app.py
|-- model.py
|-- train.py
|-- utils.py
|-- saved_model.pth
|-- .gitignore
|-- .streamlit/
|-- requirements.txt
`-- README.md
```

## Modelo

La red neuronal está definida en `model.py`:

- Input: 784 valores, correspondientes a imágenes MNIST de 28x28.
- Hidden layer 1: 128 neuronas con ReLU.
- Hidden layer 2: 64 neuronas con ReLU.
- Output: 10 clases, una por dígito de 0 a 9.

## Instalación

Desde la carpeta `mnist_visual_nn`:

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
streamlit run app.py
```

En Windows:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Uso

1. Elegí la cantidad de epochs, batch size y learning rate.
2. Presioná `Train` para entrenar el modelo.
3. La app descarga MNIST automáticamente la primera vez.
4. Al terminar, guarda los pesos en `saved_model.pth`.
5. Si `saved_model.pth` ya existe, la app lo carga al iniciar.
6. Dibujá un número blanco sobre fondo negro en el canvas.
7. Presioná `Predict` para ver la predicción y las probabilidades softmax.

Para una primera corrida razonable, usá `5` epochs, batch size `128`, learning rate `0.001` y weight decay `0.0001`. Si querés mejorar el accuracy, subí a `10` o `15` epochs.

El entrenamiento usa pequeñas transformaciones aleatorias sobre MNIST para que el modelo generalice mejor a números dibujados en el canvas.

## Preprocesamiento del canvas

`utils.py` convierte el dibujo a formato compatible con MNIST:

- Convierte el canvas RGBA a escala de grises.
- Invierte colores si detecta fondo claro.
- Recorta el dígito dibujado.
- Redimensiona y centra el dígito en 28x28.
- Normaliza con media y desvío estándar de MNIST.
- Devuelve un tensor con forma `[1, 1, 28, 28]`.

## Publicación con Streamlit Community Cloud

La forma más simple de hacer pública la app es subir este proyecto a GitHub y desplegarlo en Streamlit Community Cloud.

1. Creá un repositorio público en GitHub, por ejemplo `mnist-visual-nn`.
2. Subí los archivos de esta carpeta al repositorio.
3. Entrá a Streamlit Community Cloud: `https://share.streamlit.io/`.
4. Elegí `New app`.
5. Seleccioná tu repositorio.
6. En `Main file path`, usá:

```text
app.py
```

7. En opciones avanzadas, elegí Python `3.10` si está disponible.
8. Hacé deploy.

La app usa `requirements.txt` para instalar dependencias y `.streamlit/config.toml` para el tema visual.

Notas:

- `venv/`, `data/`, `__pycache__/` y caches locales no se suben.
- `saved_model.pth` sí puede subirse para que la app pública arranque con un modelo ya entrenado.
- Si no subís `saved_model.pth`, la app igual funciona, pero empieza con el modelo sin entrenar hasta apretar `Train`.
