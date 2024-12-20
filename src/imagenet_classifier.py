import os
from dataclasses import dataclass

import cv2 as cv
import numpy as np
import onnx
import onnxruntime as ort
import torch
from cv2.typing import MatLike
from PIL import Image
from timm.models import create_model
from torchvision import models, transforms

from configuration import configure_logger, current_timestamp
from imagenet_classes import CAT_CLASSES, MAX_CLASS_RANK
from motion_detection import ClusterBoundingBox

torch.backends.quantized.engine = "qnnpack"

imagenet_model = None
logger = configure_logger()


def classify_fn(_: np.ndarray) -> np.ndarray:
    raise Exception("No classification function set. Please load a classifier")


@dataclass(order=True)
class ClassificationResult:
    frame_idx: int
    cat_rankings: dict[int, int]  # Ranking of cat-specific classes
    top_20_classes: list[int]  # Ranking of top-20 classes
    class_scores: np.ndarray  # Raw score for each class


def load_mobilenet_v3():
    global imagenet_model
    global classify_fn

    imagenet_model = models.quantization.mobilenet_v3_large(
        pretrained=True, quantize=True, weights=models.quantization.MobileNet_V3_Large_QuantizedWeights.DEFAULT
    )

    preprocess = models.quantization.MobileNet_V3_Large_QuantizedWeights.DEFAULT.transforms()

    imagenet_model = torch.compile(imagenet_model)
    # We pass through a random input as torch compilation is lazy
    # and will only trigger on first input.
    imagenet_model(torch.randn(1, 3, 224, 224))

    def run_mobilenet_v3(frame_batch: np.ndarray) -> np.ndarray:
        t = torch.tensor(frame_batch).permute(0, 3, 1, 2)  # Rearrange to B, C, H, W format
        t = preprocess(t)
        return imagenet_model(t).numpy()

    classify_fn = run_mobilenet_v3


def load_mobilenet_v4():
    global imagenet_model
    global classify_fn

    # Load PyTorch model
    model_name = "mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k	"
    onnx_file_path = f"model/{model_name}.onnx"

    # Export to ONNX if not already done so
    if not os.path.exists(onnx_file_path):
        model = create_model(model_name, pretrained=True)
        model.eval()  # Set model to evaluation mode

        # Dummy input to define input shape
        dummy_input = torch.randn(1, 3, 224, 224)

        # Export the model to ONNX format
        torch.onnx.export(
            model,  # PyTorch model
            dummy_input,  # Example input tensor
            onnx_file_path,  # Save path
            export_params=True,  # Store trained weights
            opset_version=20,  # ONNX version (13 is widely supported)
            do_constant_folding=True,  # Optimize constant folding
            input_names=["input"],  # Input tensor name
            output_names=["output"],  # Output tensor name
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},  # Support dynamic batch size
        )

    onnx_model = onnx.load(onnx_file_path)
    onnx.checker.check_model(onnx_model)
    logger.info(f"ONNX {model_name} is valid!")

    # Start an inference session
    imagenet_model = ort.InferenceSession(onnx_file_path)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Resize image to 224x224
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize image
        ]
    )
    input_name = imagenet_model.get_inputs()[0].name
    output_name = imagenet_model.get_outputs()[0].name

    def run_mobilenet_v4(frame_batch: np.ndarray) -> np.ndarray:
        batch_tensor = torch.tensor(frame_batch, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0  # Scale to [0, 1]
        transformed_batch = torch.stack([transform(img) for img in batch_tensor])
        raw_output = imagenet_model.run([output_name], {input_name: transformed_batch.numpy()})
        return np.array(raw_output[0])

    classify_fn = run_mobilenet_v4


def load_faster_vit():
    global imagenet_model
    global classify_fn

    # FIXME this is a monkeypatch due to a bug in fastervit. See https://github.com/NVlabs/FasterViT/issues/141
    from timm.models import _builder

    _builder._update_default_kwargs = _builder._update_default_model_kwargs
    from fastervit import create_model as create_fast_model

    # Load PyTorch model
    model_name = "faster_vit_0_224"
    onnx_file_path = f"model/{model_name}.onnx"

    # Export to ONNX if not already done so
    if not os.path.exists(onnx_file_path):
        model = create_fast_model(model_name, pretrained=True, model_path="/tmp/faster_vit_0.pth.tar")
        model.eval()  # Set model to evaluation mode

        # Dummy input to define input shape
        dummy_input = torch.randn(1, 3, 224, 224)

        # Export the model to ONNX format
        torch.onnx.export(
            model,  # PyTorch model
            dummy_input,  # Example input tensor
            onnx_file_path,  # Save path
            export_params=True,  # Store trained weights
            opset_version=20,  # ONNX version (13 is widely supported)
            do_constant_folding=True,  # Optimize constant folding
            input_names=["input"],  # Input tensor name
            output_names=["output"],  # Output tensor name
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},  # Support dynamic batch size
        )

    onnx_model = onnx.load(onnx_file_path)
    onnx.checker.check_model(onnx_model)
    logger.info(f"ONNX {model_name} is valid!")

    # Start an inference session
    imagenet_model = ort.InferenceSession(onnx_file_path)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Resize image to 224x224
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize image
        ]
    )
    input_name = imagenet_model.get_inputs()[0].name
    output_name = imagenet_model.get_outputs()[0].name

    def run_faster_vit(frame_batch: np.ndarray) -> np.ndarray:
        batch_tensor = torch.tensor(frame_batch, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0  # Scale to [0, 1]
        transformed_batch = torch.stack([transform(img) for img in batch_tensor])
        raw_output = imagenet_model.run([output_name], {input_name: transformed_batch.numpy()})
        return np.array(raw_output[0])

    classify_fn = run_faster_vit


MODELS = {"mobilenet_v3": load_mobilenet_v3, "mobilenet_v4": load_mobilenet_v4, "faster_vit_0": load_faster_vit}


def load_imagenet_model(model_name: str):
    if not imagenet_model:
        loader_fn = MODELS.get(model_name)
        assert loader_fn, f"Could not find model named {model_name}. Known models: {list(MODELS.keys())}"
        loader_fn()


def get_best_frame(data: list[tuple[int, MatLike, list[ClusterBoundingBox]]]) -> tuple[int, MatLike]:
    best_data_idx = 0
    best_box_idx = 0
    best_size = data[best_data_idx][2][best_box_idx].size
    for i, (_, _, boxes) in enumerate(data):
        for j, b in enumerate(boxes):
            if b.size > best_size:
                best_data_idx = i
                best_box_idx = j
                best_size = b.size

    chosen_idx, chosen_frame, boxes = data[best_data_idx]
    chosen_box = boxes[best_box_idx]
    boxed_frame = chosen_frame[chosen_box.y_min : chosen_box.y_max, chosen_box.x_min : chosen_box.x_max]
    boxed_frame = boxed_frame[:, :, [2, 1, 0]]  # Swap colour channels from BGR -> RGB
    return chosen_idx, boxed_frame


def split_list(lst, k):
    n = len(lst)
    return [lst[i * n // k : (i + 1) * n // k] for i in range(k)]


def classify_cat_multiclass(
    data: list[tuple[int, MatLike, list[ClusterBoundingBox]]],
    num_samples: int,
    save_classifier_frames: bool,
    classifier_frame_save_folder=os.path.join("data", "log"),
    classifier_frame_prefix: str | None = None,
) -> list[ClassificationResult]:
    """Selects the most promising frame and bounding box in the input
    and then preprocesses it for a torch image classification model.

    Args:
        data (list[tuple[MatLike, list[ClusterBoundingBox]]]): frames and bounding boxes to use
         classifier_indices: list[int]: frame index corresponding to frames with bounding boxes
        save_classifier_frames (bool): whether to save the chosen frame and bounding box
        classifier_frame_save_folder (_type_, optional): _description_. Defaults to os.path.join("data", "log").
        classifier_frame_prefix (str | None, optional): _description_. Defaults to None.

    Returns:
        ClassificationResult: properties of classification
    """
    # Pick frame with the most motion to classify

    chosen_frames = [get_best_frame(d) for d in split_list(data, num_samples)]

    if save_classifier_frames:
        for f_idx, frame in chosen_frames:
            im = Image.fromarray(frame)
            file_prefix = classifier_frame_prefix or current_timestamp()
            suffix = "classifier_frame.png"
            path_str = os.path.join(classifier_frame_save_folder, f"{file_prefix}_{f_idx}_{suffix}")
            im.save(path_str)

    # Classify frames
    model_input = np.stack([cv.resize(frame, (224, 224), interpolation=cv.INTER_LINEAR) for _, frame in chosen_frames])
    out = classify_fn(model_input)

    results = []
    for i in range(out.shape[0]):
        # Extract stats
        # We multiply by -1 here as argsort is ascending only, but we want to make it descending
        ranked_classes = np.argsort(out[i] * -1)
        ranked_classes = [int(r) for r in ranked_classes]
        top_20_classes = ranked_classes[:20]
        cat_rankings = {cls: i for i, cls in enumerate(ranked_classes) if cls in CAT_CLASSES}
        rank_strs = ", ".join([f"{cls}:{i}/{MAX_CLASS_RANK}" for cls, i in cat_rankings.items()])
        logger.debug(f"Cat Class rankings: {rank_strs}")
        results.append(ClassificationResult(chosen_frames[i][0], cat_rankings, top_20_classes, out[i]))
    return results


if __name__ == "__main__":
    pass
    # print(ort.get_available_providers())
    # image_path = "data/analysis/day_cat/2024_11_17-16_43_39_0_classifier_frame.png"
    # image = Image.open(image_path).convert("RGB")
    # # t = np.array(image)
    # t = read_image(image_path)
    # t = t.unsqueeze(0)

    # load_start_t = time.time()
    # # load_imagenet_model("mobilenet_v3")
    # # load_imagenet_model("mobilenet_v4")
    # load_imagenet_model("fastvit_sa36")
    # print(f"Load time: {(time.time() - load_start_t):.2f}s")

    # classify_start_t = time.time()
    # out = classify_imagenet(image)
    # print(f"Classify time: {(time.time() - classify_start_t):.2f}s")
    # print(out.cat_rankings)
