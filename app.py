from arkitekt_next import register
from mikro_next.api.schema import (
    from_array_like,
    Image,
    File,
    get_image,
)
from kraph.api.schema import LinkedExpression, list_paired_entities, EntityFilter, create_model, Model
import xarray as xr
import numpy as np
from typing import Optional, List
from csbdeep.io import load_training_data
from csbdeep.models import Config, CARE
import numpy as np
import uuid
import shutil
from csbdeep.data import RawData, create_patches
from csbdeep.data import no_background_patches, norm_percentiles, sample_percentiles
import numpy as np
import matplotlib.pyplot as plt

from tifffile import imread
from csbdeep.utils import axes_dict, plot_some, plot_history
from csbdeep.utils.tf import limit_gpu_memory
from csbdeep.io import load_training_data
from csbdeep.models import Config, CARE
from arkitekt_next.tqdm import tqdm

@register()
def gpu_is_available() -> str:
    """Check GPU

    Check if the gpu is available

    """
    from tensorflow.python.client import device_lib
    return str(device_lib.list_local_devices())




@register()
def train_care_model(
    expression: LinkedExpression,
    epochs: int = 100,
    patches_per_image: int = 1024,
    trainsteps_per_epoch: int = 400,
    validation_split: float = 0.1,
) -> Model:
    """Train Care Model

    Trains a care model according on a specific context.

    Args:
        context (ContextFragment): The context
        epochs (int, optional): Number of epochs. Defaults to 10.
        patches_per_image (int, optional): Number of patches per image. Defaults to 1024.
        trainsteps_per_epoch (int, optional): Number of trainsteps per epoch. Defaults to 10.
        validation_split (float, optional): Validation split. Defaults to 0.1.


    Returns:
        ModelFragment: The Model
    """

    training_data_id = str(uuid.uuid4())
    
    entities = list_paired_entities(graph=expression.graph.id, left_filter=EntityFilter(identifier="@mikro/image"), right_filter=EntityFilter(identifier="@mikro/image"))

    X = [get_image(e.left.object).data.sel(t=0, c=0).compute() for e in entities]
    Y = [get_image(e.right.object).data.sel(t=0, c=0).compute() for e in entities]


    print(X, Y)


    raw_data = RawData.from_arrays(X, Y, axes="ZYX")
    print(raw_data)

    X, Y, XY_axes = create_patches(
        raw_data=raw_data,
        patch_size=(16, 64, 64),
        n_patches_per_image=patches_per_image,
        save_file=f"data/{training_data_id}.npz",
    )

    (X, Y), (X_val, Y_val), axes = load_training_data(
        f"data/{training_data_id}.npz",
        validation_split=validation_split,
        verbose=True,
    )
    config = Config(axes, train_steps_per_epoch=trainsteps_per_epoch)

    model = CARE(config, training_data_id, basedir=".trainedmodels")


    for i in tqdm(range(epochs)):
        model.train(X, Y, validation_data=(X_val, Y_val), epochs=1)

    archive = shutil.make_archive(
        "active_model", "zip", f".trainedmodels/{training_data_id}"
    )

    model = create_model(
        "active_model.zip",
        name=f"Care Model",
    )


    shutil.rmtree(f"data")
    return model


@register()
def predict(
    representation: Image, model: Model
) -> Image:
    """Predict Care

    Use a care model and some images to generate images

    Args:
        model (ImageToImageModelFragment): The model
        representations (List[RepresentationFragment]): The images

    Returns:
        List[RepresentationFragment]: The predicted images
    """

    random_dir = str(uuid.uuid4())
    generated = []


    f  = model.store.download("model.zip")

    shutil.unpack_archive(f, f".modelcache/{random_dir}")

    image_data = representation.data.sel(c=0, t=0).data.compute()
    care_model = CARE(config=None, name=random_dir, basedir=".modelcache")
    restored = care_model.predict(
        image_data, "ZXY"
    )

    t = restored.dtype
    if   'float' in t.name:
        t_new = np.float32
    elif 'uint'  in t.name: 
        t_new = np.uint16 if t.itemsize >= 2 else np.uint8
    elif 'int'   in t.name:
        t_new = np.int16
    else:                  
        t_new = t
    
    img = restored.astype(t_new, copy=False)

    generated = from_array_like(
        img,
        name=f"Care denoised of {representation.name}",
        tags=["denoised"],
        origins=[representation],
    )

    shutil.rmtree(f".modelcache/{random_dir}")
    return generated




