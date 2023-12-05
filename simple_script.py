# Let's start exploring model training with FastAI and modal
from dataclasses import dataclass
import os

from modal import Image, Stub, method, asgi_app, Secret
from pathlib import Path
from fastapi import FastAPI

stub = Stub(name="face-shapes")
MODEL_DIR = "/face_model"
BASE_MODEL = "lagerbaer/female_face_shapes"


web_app = FastAPI()


def download_model_to_folder():
    from huggingface_hub import snapshot_download

    os.makedirs(MODEL_DIR, exist_ok=True)

    snapshot_download(
        BASE_MODEL,
        local_dir=MODEL_DIR,
        token=os.environ["HUGGINGFACE_TOKEN"],
    )


# Image packages roughly taken from the modal docs
image = (
    Image.debian_slim()
    .pip_install(
        "fastai",
        "gradio~=3.6",
        "httpx",
        # When using pip PyTorch is not automatically installed by fastai.
        "torch",
        "torchvision",
        "hf-transfer~=0.1",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        download_model_to_folder,
        secret=Secret.from_name("my-huggingface-secret"),
    )
)

USE_GPU = os.environ.get("MODAL_GPU")


@stub.cls(image=image, gpu=USE_GPU)
class FaceClassifier:
    def __enter__(self):
        from fastai.learner import load_learner

        print("Loading the learner...")
        self.learner = load_learner(Path(MODEL_DIR, "learner.pkl"))
        print("Done")

    @method()
    def predict(self, image) -> str:
        print("Predicting label")
        prediction = self.learner.predict(image)
        classification = prediction[0]
        return classification


@stub.function(image=image, gpu=USE_GPU)
def classify_url(image_url: str) -> None:
    """Utility function for command-line classification runs."""
    import httpx

    r = httpx.get(image_url)
    if r.status_code != 200:
        raise RuntimeError(f"Could not download '{image_url}'")

    classifier = FaceClassifier()
    label = classifier.predict.remote(image=r.content)
    print(f"Classification: {label}")


@stub.function(image=image)
@asgi_app()
def fastapi_app():
    import gradio as gr
    from gradio.routes import mount_gradio_app

    classifier = FaceClassifier()
    interface = gr.Interface(
        fn=classifier.predict.remote,
        inputs=gr.Image(shape=(224, 224)),
        outputs="label",
    )
    return mount_gradio_app(
        app=web_app,
        blocks=interface,
        path="/",
    )
