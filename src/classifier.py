import os
from datetime import datetime

import torch
from cv2.typing import MatLike
from PIL import Image
from torchvision import models
from torchvision.io import read_image

from configuration import configure_logger, to_timestamp
from motion_detection import ClusterBoundingBox

torch.backends.quantized.engine = "qnnpack"

model = None
preprocess = None
logger = configure_logger()


def classify_cat(
    data: list[tuple[MatLike, list[ClusterBoundingBox]]],
    save_classifier_frames: bool,
    classifier_frame_save_folder=os.path.join("data", "log"),
) -> bool:
    # Pick frame with the most motion to classify
    best_frame_idx = 0
    best_box_idx = 0
    best_size = data[best_frame_idx][1][best_box_idx].size
    for i, (_, boxes) in enumerate(data):
        for j, b in enumerate(boxes):
            if b.size > best_size:
                best_frame_idx = i
                best_box_idx = j
                best_size = b.size

    chosen_frame, boxes = data[best_frame_idx]
    chosen_box = boxes[best_box_idx]

    boxed_frame = chosen_frame[chosen_box.y_min : chosen_box.y_max, chosen_box.x_min : chosen_box.x_max]
    s = boxed_frame.shape

    # TODO make controllable
    if save_classifier_frames:
        current_time = datetime.now()
        im = Image.fromarray(boxed_frame)
        path_str = os.path.join(classifier_frame_save_folder, f"{to_timestamp(current_time)}_classifier_frame.png")
        im.save(path_str)

    t = torch.tensor(boxed_frame).permute(2, 0, 1)  # Rearrange to C, H, W format
    t = t.unsqueeze(0)  # Add batch dimension, changing to B, C, H, W format
    # TODO return something
    classify(t)


# TODO doc
def classify(frames: torch.tensor) -> list[tuple[int, int]]:
    global model
    global preprocess

    if not model:
        model = models.quantization.mobilenet_v3_large(
            pretrained=True, quantize=True, weights=models.quantization.MobileNet_V3_Large_QuantizedWeights.DEFAULT
        )
        preprocess = models.quantization.MobileNet_V3_Large_QuantizedWeights.DEFAULT.transforms()

    model.eval()
    torch.jit.script(model)

    cat_classes = {281, 282, 283, 284, 285}

    with torch.no_grad():
        model_input = preprocess(frames)
        out = model(model_input).flatten()  # TODO we kind assume a batch size of 1

        ranked_classes = torch.argsort(out, descending=True)
        ranked_classes = [int(r) for r in ranked_classes]

        cat_rankings = [(i, cls) for i, cls in enumerate(ranked_classes) if cls in cat_classes]
        max_rank = len(ranked_classes)
        rank_strs = ", ".join([f"{cls}:{i}/{max_rank}" for i, cls in cat_rankings])
        logger.debug(f"Cat Class rankings: {rank_strs}")


if __name__ == "__main__":
    # Our image
    # image_path = "data/image/our_cats/simple_enhanced_image.png"

    # Our other image
    # image_path = "data/image/our_cats/2024_11_26-20_26_32_classifier_frame.png"

    # Our other other image
    # image_path = "data/image/our_cats/2024_11_26-20_27_06_classifier_frame.png"

    # Our debug image
    # Frame performance:
    # Cat Class rankings: 284:86/1000, 283:196/1000, 285:349/1000, 281:633/1000, 282:678/1000
    image_path = "data/log/2024_11_29-19_18_15_classifier_frame.png"
    tensor_path = "data/image/our_cats/2024_11_26-21_38_28_tensor.pt"
    # image_tensor_path = "data/image/our_cats/2024_11_26-21_38_28_resaved.png"

    # Our 2nd debug image
    # Frame performance:
    # Cat Class rankings: 284:141/1000, 283:342/1000, 285:449/1000, 281:781/1000, 282:789/1000
    # image_path = "data/image/our_cats/2024_11_26-21_38_31_classifier_frame.png

    # Training image
    # image_path = "data/image/training_cats/285_egyptian.jpeg"

    # Online image
    # image_path = "data/image/other_cats/web_cat_2.jpg"

    # debug_tensor = torch.load(tensor_path)
    t = read_image(image_path)
    # debug_arr = np.transpose(np.asarray(Image.open(image_path)), (2,0,1))
    t = t.unsqueeze(0)

    # debug_arr = simulate_pillow_save_load(np.transpose(debug_tensor.squeeze().numpy(),(1,2,0)))

    # im = to_pil_image(debug_tensor.squeeze(0))  # Convert array to tensor, then to PIL
    # debug_arr = to_tensor(im).mul(255).byte().numpy()

    # debug_numpy = debug_tensor.squeeze().permute(1,2,0).numpy()
    # im = Image.fromarray(debug_numpy)
    # debug_arr = np.asarray(im)
    classify(t)
