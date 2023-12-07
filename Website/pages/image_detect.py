import streamlit as st
from PIL import Image
import numpy as np
from numpy import array, float32
import os
import requests
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches
import json

def resize_image(image):
    image = Image.open(image)
    image = image.resize((192, 192))
    image_array = np.array(image)
    return np.expand_dims(image_array, axis=0)

def save_image_as_jpeg(image_data, filename):
    # Create the directory if it doesn't exist
    if not os.path.exists('../uploaded_images'):
        os.makedirs('../uploaded_images')

    # Remove extension and add .jpg
    filename = os.path.splitext(filename)[0] + '.jpg'

    # Save the image
    image = Image.open(image_data)
    image.save(os.path.join('uploaded_images', filename), 'JPEG')

def return_pose(model_output):

    pose_num = eval(model_output["predict"])[0]

    if pose_num == 0:

        pose = "Lunge Left"
    elif pose_num == 1:

        pose = "Lunge Right"
    elif pose_num == 2:

        pose = "Pushup"
    elif pose_num == 3:
        pose = "Squat"

    return pose

def _keypoints_and_edges_for_display(keypoints_with_scores,
                                     height,
                                     width,
                                     keypoint_threshold=0.11):
    """Returns high confidence keypoints and edges for visualization.

    Args:
    keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
        the keypoint coordinates and scores returned from the MoveNet model.
    height: height of the image in pixels.
    width: width of the image in pixels.
    keypoint_threshold: minimum confidence score for a keypoint to be
        visualized.

    Returns:
    A (keypoints_xy, edges_xy, edge_colors) containing:
        * the coordinates of all keypoints of all detected entities;
        * the coordinates of all skeleton edges of all detected entities;
        * the colors in which the edges should be plotted.
    """

    KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): 'g',
    (0, 2): 'g',
    (1, 3): 'g',
    (2, 4): 'g',
    (0, 5): 'g',
    (0, 6): 'g',
    (5, 7): 'g',
    (7, 9): 'g',
    (6, 8): 'g',
    (8, 10): 'g',
    (5, 6): 'g',
    (5, 11): 'g',
    (6, 12): 'g',
    (11, 12): 'g',
    (11, 13): 'g',
    (13, 15): 'g',
    (12, 14): 'g',
    (14, 16): 'g'
    }

    keypoints_all = []
    keypoint_edges_all = []
    edge_colors = []
    num_instances, _, _, _ = keypoints_with_scores.shape
    for idx in range(num_instances):
        kpts_x = keypoints_with_scores[0, idx, :, 1]
        kpts_y = keypoints_with_scores[0, idx, :, 0]
        kpts_scores = keypoints_with_scores[0, idx, :, 2]
        kpts_absolute_xy = np.stack(
            [width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1)
        kpts_above_thresh_absolute = kpts_absolute_xy[
            kpts_scores > keypoint_threshold, :]
        keypoints_all.append(kpts_above_thresh_absolute)

    for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
        if (kpts_scores[edge_pair[0]] > keypoint_threshold and
            kpts_scores[edge_pair[1]] > keypoint_threshold):

            x_start = kpts_absolute_xy[edge_pair[0], 0]
            y_start = kpts_absolute_xy[edge_pair[0], 1]
            x_end = kpts_absolute_xy[edge_pair[1], 0]
            y_end = kpts_absolute_xy[edge_pair[1], 1]
            line_seg = np.array([[x_start, y_start], [x_end, y_end]])
            keypoint_edges_all.append(line_seg)
            edge_colors.append(color)

    if keypoints_all:
        keypoints_xy = np.concatenate(keypoints_all, axis=0)

    else:
        keypoints_xy = np.zeros((0, 17, 2))

    if keypoint_edges_all:
        edges_xy = np.stack(keypoint_edges_all, axis=0)
    else:
        edges_xy = np.zeros((0, 2, 2))

    return keypoints_xy, edges_xy, edge_colors

def draw_prediction_on_image(
    image, keypoints_with_scores, crop_region=None, close_figure=False,
    output_image_height=None):
  """Draws the keypoint predictions on image.

  Args:
    image: A numpy array with shape [height, width, channel] representing the
      pixel values of the input image.
    keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
      the keypoint coordinates and scores returned from the MoveNet model.
    crop_region: A dictionary that defines the coordinates of the bounding box
      of the crop region in normalized coordinates (see the init_crop_region
      function below for more detail). If provided, this function will also
      draw the bounding box on the image.
    output_image_height: An integer indicating the height of the output image.
      Note that the image aspect ratio will be the same as the input image.

  Returns:
    A numpy array with shape [out_height, out_width, channel] representing the
    image overlaid with keypoint predictions.
  """

  height, width, channel = image.shape

  print(height, width, channel)

  aspect_ratio = float(width) / height

  fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
  # To remove the huge white borders
  fig.tight_layout(pad=0)
  ax.margins(0)
  ax.set_yticklabels([])
  ax.set_xticklabels([])
  plt.axis('off')

  im = ax.imshow(image)

  line_segments = LineCollection([], linewidths=(4), linestyle='solid')
  ax.add_collection(line_segments)
  # Turn off tick labels
  scat = ax.scatter([], [], s=60, color='#FF1493', zorder=3)

  (keypoint_locs, keypoint_edges,
   edge_colors) = _keypoints_and_edges_for_display(
       keypoints_with_scores, height, width)

  line_segments.set_segments(keypoint_edges)
  line_segments.set_color(edge_colors)
  if keypoint_edges.shape[0]:
    line_segments.set_segments(keypoint_edges)
    line_segments.set_color(edge_colors)
  if keypoint_locs.shape[0]:
    scat.set_offsets(keypoint_locs)

  if crop_region is not None:
    xmin = max(crop_region['x_min'] * width, 0.0)
    ymin = max(crop_region['y_min'] * height, 0.0)
    rec_width = min(crop_region['x_max'], 0.99) * width - xmin
    rec_height = min(crop_region['y_max'], 0.99) * height - ymin
    rect = patches.Rectangle(
        (xmin,ymin),rec_width,rec_height,
        linewidth=1,edgecolor='b',facecolor='none')
    ax.add_patch(rect)

  fig.canvas.draw()
  fig = plt.gcf()
  size = fig.get_size_inches()*fig.dpi
  print(size)
  image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
#   print(image_from_plot)

  image_from_plot = image_from_plot.reshape(1200,900,3)
#   image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#   plt.close(fig)
#   if output_image_height is not None:
#     output_image_width = int(output_image_height / height * width)
#     image_from_plot = cv2.resize(
#         image_from_plot, dsize=(output_image_width, output_image_height),
#          interpolation=cv2.INTER_CUBIC)
  return image_from_plot

def plot_skeleton_on_image(image, keypoints_with_scores):

    display_image = load_image_for_skeleton(image)

    output_overlay = draw_prediction_on_image(
        np.squeeze(display_image, axis=0), keypoints_with_scores)

    return output_overlay

def load_image_for_skeleton(image_path):
    # Open the image file
    with Image.open(image_path) as img:
        # Convert to RGB if it's a 4-channel image (e.g., RGBA)
        if img.mode == 'RGBA':
            img = img.convert('RGB')

        # Resize and pad the image to maintain aspect ratio
        width, height = img.size
        new_size = max(width, height)
        new_im = Image.new("RGB", (new_size, new_size))
        new_im.paste(img, ((new_size-width)//2, (new_size-height)//2))

        # Resize to the target dimensions
        resized_image = new_im.resize((1280, 1280))

        # Convert to NumPy array and expand dimensions
        display_image = np.expand_dims(np.array(resized_image), axis=0)

        return display_image

url = 'http://0.0.0.0:8000' # URL of the registry API

# Streamlit page layout
st.title('Image Upload and Reshape')

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:

    #st.image(uploaded_file, caption="uploaded_img", use_column_width=True)
    image = Image.open(uploaded_file)

    image_array = np.array(image)

    # save_image_as_jpeg(uploaded_file, "test_image")

    # # Path to the local image file
    # file_path = "uploaded_images/test_image.jpg"

    # Open the file in binary mode
    # with open(file_path, 'rb') as f:
    #     # Define the file as a dictionary. The key ('file' in this case)
    #     # should match the name of the parameter in your FastAPI endpoint
    #     files = {'file': (file_path, f, 'image/jpeg')}

        # Make the POST request
    # respose_skeleton = requests.post(f"{url}/skeletonizer/", files=files)

    respose_skeleton = requests.post("http://0.0.0.0:8000/skeletonizer", json=json.dumps(image_array.tolist()))

    keypoints_scores = eval(eval(eval(respose_skeleton.text)["keypoints_scores"]))
    keypoints_angles = eval(eval(respose_skeleton.text)["keypoints"])

    keypoint_angles = {key: value["0"] for key, value in keypoints_angles.items()}

    # skele_array = plot_skeleton_on_image("uploaded_images/test_image.jpg", keypoints_scores)

    skele_array = draw_prediction_on_image(image_array, keypoints_scores)

    skele_image = Image.fromarray(skele_array)

    st.image(skele_image)

    dict_var = eval(respose_skeleton.text)

    dict_var['keypoints'] = f"{keypoint_angles}"

    input_for_model = {"data":dict_var}
    response_pose = requests.post(f"{url}/automl_model/", json=input_for_model)

    pose_string = return_pose(eval(response_pose.text))

    st.markdown(f"**GOD DAMN NOW THAT'S ONE HELL OF A {pose_string.upper()}, KEEP UP THE GOOD WORK CHAMP**")
