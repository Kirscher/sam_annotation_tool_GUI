import cv2
import os
import json
import time
import torch
import numpy as np

from pycocotools import mask as mask_utils
from segment_anything import sam_model_registry, SamPredictor

# Constants
INPUT_DIR = "input"
OUTPUT_DIR = "output"
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg"]
POINT_RADIUS = 3


def get_image_files(directory):
    """Return a list of image filenames in the given directory."""
    return [
        f for f in os.listdir(directory) if f.lower().endswith(tuple(IMAGE_EXTENSIONS))
    ]


# Create the output directory if it does not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get the list of image files in the input directory
image_files = get_image_files(INPUT_DIR)

# Load the model
print("Loading model...")
sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")

# Determine the device to run the model on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}")

# Load the model to the device
sam = sam.to(device=device)
predictor = SamPredictor(sam)


def mouse_click(event, x, y, flags, param):
    """
    Callback function for mouse click events.

    Args:
        event: The type of mouse event (e.g., cv2.EVENT_LBUTTONDOWN).
        x: The x-coordinate of the mouse click.
        y: The y-coordinate of the mouse click.
        flags: Any flags passed to the callback.
        param: Any extra parameters passed to the callback.

    Returns:
        None
    """
    global input_point, input_label, input_stop
    if not input_stop:
        if event == cv2.EVENT_LBUTTONDOWN:
            input_point.append([x, y])
            input_label.append(1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            input_point.append([x, y])
            input_label.append(0)
    else:
        if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
            print('Cannot add points. Press "w" to exit mask selection mode.')


def apply_mask(image, mask, alpha_channel=True):
    """
    Applies a mask to an image.

    Args:
        image (numpy.ndarray): The input image.
        mask (numpy.ndarray): The mask to be applied.
        alpha_channel (bool, optional): Whether to include an alpha channel in the output image.
            Defaults to True.

    Returns:
        numpy.ndarray: The masked image.
    """
    if alpha_channel:
        alpha = np.zeros_like(image[..., 0])
        alpha[mask == 1] = 255
        image = cv2.merge((image[..., 0], image[..., 1], image[..., 2], alpha))
    else:
        image = np.where(mask[..., None] == 1, image, 0)
    return image


def apply_color_mask(image, mask, color, color_dark=0.5):
    """
    Applies a color mask to an image.

    Args:
        image (numpy.ndarray): The input image.
        mask (numpy.ndarray): The mask indicating the regions to apply the color to.
        color (tuple): The RGB color to apply.
        color_dark (float, optional): The darkness factor for the color. Defaults to 0.5.

    Returns:
        numpy.ndarray: The image with the color mask applied.
    """
    for c in range(3):
        image[:, :, c] = np.where(
            mask == 1,
            image[:, :, c] * (1 - color_dark) + color_dark * color[c],
            image[:, :, c],
        )
    return image


def get_next_filename(base_path: str, filename: str) -> str:
    """
    Generate a new filename by appending a number to the original filename.

    Parameters:
    base_path (str): The directory where to check for the filename.
    filename (str): The original filename.

    Returns:
    str: The new filename if an available one is found, otherwise None.
    """
    name, ext = os.path.splitext(filename)
    i = 1
    while True:
        new_name = f"{name}_{i}{ext}"
        if not os.path.exists(os.path.join(base_path, new_name)):
            return new_name
        i += 1


def save_mask(mask, output_dir, filename):
    """
    Save the mask in COCO RLE format as a JSON file in the specified output directory.

    Args:
        image (numpy.ndarray): The original image.
        mask (numpy.ndarray): The mask to be applied to the image.
        output_dir (str): The directory where the masked image will be saved.
        filename (str): The original filename of the image.

    Returns:
        None
    """
    # Encode the mask in COCO RLE format
    rle = mask_utils.encode(np.asfortranarray(mask))
    # Convert the RLE mask to a string
    rle["counts"] = rle["counts"].decode("utf-8")

    # Save the RLE mask to a JSON file
    with open(os.path.join(output_dir, filename.rsplit(".", 1)[0] + ".json"), "w") as f:
        json.dump(rle, f)

    print(f"Saved as {filename.rsplit('.', 1)[0]}.json")


current_index = 0

cv2.namedWindow("SAM - Annotation Tool")
cv2.setMouseCallback("SAM - Annotation Tool", mouse_click)
input_point = []
input_label = []
input_stop = False
while True:
    filename = image_files[current_index]
    image_orign = cv2.imread(os.path.join(INPUT_DIR, filename))
    image = cv2.cvtColor(image_orign.copy(), cv2.COLOR_BGR2RGB)
    selected_mask = None
    logit_input = None
    while True:
        # print(input_point)
        input_stop = False
        image_display = image_orign.copy()
        display_info = f"{filename} | s: save | w: predict | d: next image | a: prev. image | space: clear | q: remove last point"
        cv2.putText(
            image_display,
            display_info,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        for point, label in zip(input_point, input_label):
            color = (0, 255, 0) if label == 1 else (0, 0, 255)
            cv2.circle(image_display, tuple(point), POINT_RADIUS, color, -1)
        if selected_mask is not None:
            color = tuple(np.random.randint(0, 256, 3).tolist())
            selected_image = apply_color_mask(image_display, selected_mask, color)

        cv2.imshow("SAM - Annotation Tool", image_display)
        key = cv2.waitKey(1)

        if key == ord(" "):
            input_point = []
            input_label = []
            selected_mask = None
            logit_input = None
        elif key == ord("w"):
            print("Predicting...")
            input_stop = True
            if len(input_point) > 0 and len(input_label) > 0:

                start_time = time.time()
                predictor.set_image(image)
                input_point_np = np.array(input_point)
                input_label_np = np.array(input_label)

                masks, scores, logits = predictor.predict(
                    point_coords=input_point_np,
                    point_labels=input_label_np,
                    mask_input=(
                        logit_input[None, :, :] if logit_input is not None else None
                    ),
                    multimask_output=True,
                )
                end_time = time.time()
                print(f"Prediction duration: {end_time - start_time} seconds")

                mask_idx = 0
                num_masks = len(masks)
                while 1:
                    color = tuple(np.random.randint(0, 256, 3).tolist())
                    image_select = image_orign.copy()
                    selected_mask = masks[mask_idx]
                    selected_image = apply_color_mask(
                        image_select, selected_mask, color
                    )
                    mask_info = f"Total: {num_masks} | Current: {mask_idx} | Score: {scores[mask_idx]:.2f} | w: confirm | d: next mask | a: prev. mask | q: rm last point | s: save"
                    cv2.putText(
                        selected_image,
                        mask_info,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                    cv2.imshow("SAM - Annotation Tool", selected_image)

                    key = cv2.waitKey(10)
                    if key == ord("q") and len(input_point) > 0:  # remove last point
                        input_point.pop(-1)
                        input_label.pop(-1)
                    elif key == ord("s"):  # save mask
                        save_mask(
                            selected_mask,
                            OUTPUT_DIR,
                            filename,
                        )
                    elif key == ord("a"):  # prev mask
                        if mask_idx > 0:
                            mask_idx -= 1
                        else:
                            mask_idx = num_masks - 1
                    elif key == ord("d"):  # next mask
                        if mask_idx < num_masks - 1:
                            mask_idx += 1
                        else:
                            mask_idx = 0
                    elif key == ord("w"):  # confirm
                        break
                    elif key == ord(" "):  # clear
                        input_point = []
                        input_label = []
                        selected_mask = None
                        logit_input = None
                        break
                logit_input = logits[mask_idx, :, :]
                print("max score:", np.argmax(scores), " select:", mask_idx)

        elif key == ord("a"):  # prev image
            current_index = max(0, current_index - 1)
            input_point = []
            input_label = []
            break
        elif key == ord("d"):  # next image
            current_index = min(len(image_files) - 1, current_index + 1)
            input_point = []
            input_label = []
            break
        elif key == 27:  # ESC
            break
        elif key == ord("q") and len(input_point) > 0:  # remove last point
            input_point.pop(-1)
            input_label.pop(-1)
        elif key == ord("s") and selected_mask is not None:  # save mask
            save_mask(selected_mask, OUTPUT_DIR, filename)

    if key == 27:  # ESC
        break
