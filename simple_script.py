# Let's start exploring model training with FastAI and modal
from dataclasses import dataclass
import os

from modal import Image, NetworkFileSystem, Stub
from pathlib import Path

stub = Stub(name="face-shapes")

# Image packages roughly taken from the modal docs
image = (
    Image.debian_slim()
    .pip_install(
        "fastai",
        "gradio",
        "httpx",
        # When using pip PyTorch is not automatically installed by fastai.
        "torch",
        "torchvision",
    )
    .run_commands("mkdir /faces")
    .copy_local_dir("faces", "/faces")
)

# Now we need a volume to store the model artifact
volume = NetworkFileSystem.persisted("faceshape-training-vol")

# Some constants for the environment
FASTAI_HOME = "/fastai_home"
MODEL_CACHE = Path(FASTAI_HOME, "models")
USE_GPU = os.environ.get("MODAL_GPU")
MODEL_EXPORT_PATH = Path(MODEL_CACHE, "model-exports", "inference.pkl")

# Apparently we need this so the fastai library behaves nicely, i.e., saves data into the path of our
# persistent volume
os.environ["FASTAI_HOME"] = FASTAI_HOME


@dataclass
class Config:
    epochs: int = 10
    dim: int = 256
    gpu: str = USE_GPU


@stub.function(
    image=image,
    gpu=USE_GPU,
    network_file_systems={str(MODEL_CACHE): volume},
    timeout=2700,
)
def train():
    from fastai.data.transforms import parent_label
    from fastai.metrics import accuracy
    from fastai.vision.all import Resize, models, vision_learner, RatioResize
    from fastai.vision.data import (
        CategoryBlock,
        DataBlock,
        ImageBlock,
        TensorCategory,
        get_image_files,
        RandomSplitter,
    )
    from fastai.vision.augment import CropPad

    config: Config = Config()

    dataset_path = Path("/", "faces")

    # Let's see if we need to tweak the resize method!
    dblock = DataBlock(
        blocks=[ImageBlock(), CategoryBlock()],
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=[RatioResize(config.dim), CropPad(config.dim)],
    )

    dls = dblock.dataloaders(dataset_path, bs=64)

    learn = vision_learner(dls, models.resnet18, metrics=accuracy)
    learn.fine_tune(config.epochs, freeze_epochs=3)
    learn.save("faces_model")

    MODEL_EXPORT_PATH.parent.mkdir(exist_ok=True)
    learn.export(MODEL_EXPORT_PATH)
