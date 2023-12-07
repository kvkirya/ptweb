import streamlit as st
import numpy as np
from PIL import Image
from time import sleep
import cv2
import av
from streamlit_webrtc import webrtc_streamer
import requests
import os
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches
import json
from numpy import array, float32
import threading

lock = threading.Lock()
frame_container = {"frame_count": 0}


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

  image_from_plot = image_from_plot.reshape(1200,1600,3)

# #   image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
# #   plt.close(fig)
#   if output_image_height is not None:
#     output_image_width = int(output_image_height / height * width)
#     image_from_plot = cv2.resize(
#         image_from_plot, dsize=(output_image_width, output_image_height),
#          interpolation=cv2.INTER_CUBIC)
  return image_from_plot

def plot_skeleton_on_image(image, keypoints_with_scores):

    # display_image = load_image_for_skeleton(image)

    output_overlay = draw_prediction_on_image(
        image, keypoints_with_scores)

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


def webcam():

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        image_array = frame.to_ndarray(format="bgr24")

        print(image_array.shape)

        image_array = np.flip(image_array, axis=-1)

        with lock:
            if frame_container["frame_count"] % 2 == 0:
                frame_container["frame_count"] = 1
                response = requests.post("https://ptai-2smsbtwy5q-ew.a.run.app/skeletonizer", json=json.dumps(image_array.tolist()))
                respose_skeleton = response
                keypoints_scores = eval(eval(eval(respose_skeleton.text)["keypoints_scores"]))
                frame_container['keypoints_scores'] = keypoints_scores
            else:
                frame_container["frame_count"] += 1


        keypoints_scores = frame_container["keypoints_scores"]

        # keypoint_angles = {key: value["0"] for key, value in keypoints_angles.items()}

        # skele_array = plot_skeleton_on_image(image_array, keypoints_scores)

        image_array = draw_prediction_on_image(image_array, keypoints_scores)

        revserse_skele = np.flip(image_array, axis=-1)

        # if model is None:
        #     return image

        # orig_h, orig_w = image.shape[0:2]

        # # cv2.resize used in a forked thread may cause memory leaks
        # input = np.asarray(Image.fromarray(image).resize((width, int(width * orig_h / orig_w))))

        # transferred = style_transfer(input, model)

        # result = Image.fromarray((transferred * 255).astype(np.uint8))
        # image = np.asarray(result.resize((orig_w, orig_h)))
        return av.VideoFrame.from_ndarray(revserse_skele, format="bgr24")

    ctx = webrtc_streamer(
        key="neural-style-transfer",
        video_frame_callback=video_frame_callback,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                            },
        media_stream_constraints={"video": True, "audio": False},
    )


def show_loading_bars(loading_bar_dict):
    ''' This shows a loading bar for each key in a dictionary provided,
        for the number of seconds included in their keys
    '''
    # Delay
    sleep(0.5)

    # Extract data from loading_bar_dict
    text_prompts = list(loading_bar_dict.keys())
    time_delay = list(loading_bar_dict.values())

    # Create a loading bar for each dictionary key
    for i in range(len(text_prompts)):
        loading_bar(text_prompts[i], time_delay[i])

    # Print success message and delete after 2 seconds


def loading_bar(text: str, loading_time: int):
    ''' This creates a loading bar that displays the inputted text
        and lasts for the loading_time length
    '''
    progress_bar = st.progress(0)
    st.text(text)
    for i in range(loading_time * 10):
        progress = (i + 1) / (loading_time * 10)
        progress_bar.progress(progress)
        sleep(0.1)

def home():
    st.subheader('''''')

def about_the_team():
    st.title('About the Team')
    st.write("Welcome to the PT AI Team page!")
    st.write("Our team is dedicated to creating innovative solutions for physical training and health.")

def squat_tutorial():
    st.title('Squat Tutorial')
    st.markdown("![Alt Text](https://seven.app/media/images/image4.gif)")
    st.subheader('Proper Foot Placement:')
    st.write('''Stand with your feet shoulder-width apart. Your toes should be slightly turned out, around 5-20 degrees, depending on your comfort and mobility.
Keep your weight evenly distributed across your feet.''')
    st.subheader('''Posture:''')
    st.write('''Maintain a neutral spine with your chest up and shoulders back.
Engage your core by pulling your belly button toward your spine. This will help stabilize your spine during the squat.''')
    st.subheader('''Squat Descent:''')
    st.write('''Begin the squat by pushing your hips back, as if you are sitting into a chair. Imagine that your hips are moving backward and downward simultaneously.
Keep your knees in line with your toes and make sure they don't cave inward.
Lower yourself gradually and maintain control. Go as low as your mobility and comfort allow, ideally until your thighs are parallel to the ground or even lower (known as a deep squat).
Keep your weight on your heels or mid-foot, not on your toes.''')

    st.subheader('''Depth and Range of Motion:''')
    st.write('''Aim for a full range of motion if your mobility allows it. Going deeper can activate more muscle fibers.
Ensure your knees do not go past your toes when you squat.
''')

    st.subheader('''Squat Ascent:''')
    st.write('''Push through your heels to stand back up. Keep your core engaged throughout the movement.
Straighten your hips and knees simultaneously as you return to the starting position.''')

def pushup_tutorial():
    st.title('Push-Up Tutorial')
    st.markdown("![Proper Push-Up](https://example.com/your_pushup_gif.gif)")
    st.subheader('Proper Push-Up Form:')
    st.write('''1. **Starting Position:** Begin with your hands placed slightly wider than shoulder-width apart, and your palms flat on the floor.
    2. **Body Alignment:** Keep your body in a straight line from head to heels. Engage your core muscles to maintain this alignment throughout the exercise.
    3. **Elbows Tucked:** Lower your chest toward the ground by bending your elbows. Keep your elbows close to your body at a 45-degree angle.
    4. **Full Range of Motion:** Lower your body until your chest is about an inch from the floor, or as low as your mobility allows.
    5. **Push Back Up:** Push through your palms and extend your arms to return to the starting position.
    6. **Breathing:** Inhale as you lower your body, and exhale as you push back up.
    7. **Repetitions and Sets:** Aim for 3 sets of 10-15 repetitions, or adjust based on your fitness level.
    8. **Variations:** You can modify push-ups by doing them on your knees or elevating your hands on a stable surface if needed.
    9. **Rest:** Make sure to take adequate rest between sets to maintain proper form.''')

def lunge_tutorial():
    st.title('Lunge Tutorial')
    st.markdown("![Proper Lunge](https://example.com/your_lunge_gif.gif)")
    st.subheader('Proper Lunge Form:')
    st.write('''1. **Starting Position:** Stand up straight with your feet hip-width apart.
    2. **Step Forward:** Take a step forward with one leg, bending both knees to create two 90-degree angles.
    3. **Front Knee Alignment:** Ensure that your front knee is directly above your front ankle.
    4. **Back Knee:** Your back knee should hover just above the ground without touching it.
    5. **Upright Posture:** Keep your upper body straight with your chest up and shoulders back.
    6. **Core Engagement:** Engage your core muscles to maintain balance and stability.
    7. **Step Back:** Push off with your front foot to return to the starting position.
    8. **Switch Legs:** Alternate between your left and right legs for each lunge.
    9. **Breathing:** Inhale as you step forward and lower into the lunge, and exhale as you push back up.
    10. **Repetitions and Sets:** Aim for 3 sets of 10-12 lunges per leg or adjust based on your fitness level.
    11. **Variations:** You can modify lunges by doing reverse lunges, walking lunges, or adding weights for extra resistance.
    12. **Rest:** Take adequate rest between sets to maintain proper form.''')


def main():
    st.set_page_config(layout='wide')

     # Create a horizontal layout for the title and logo
    title_column, logo_column = st.columns([5, 1])


    with title_column:
        st.title('PT-AI')
        st.sidebar.title("Navigation")

    with logo_column:
        st.image("https://t4.ftcdn.net/jpg/02/49/85/41/360_F_249854185_WiRZhGX2B81qEtXcYVCcNiyBVDfeFWIb.jpg", use_column_width=True)

    webcam()

    # Add navigation buttons with custom styling
    # if st.sidebar.button("Home"):

    # if st.sidebar.button("About the Team"):
    #     about_the_team()
    # if st.sidebar.button("Squat Tutorial"):
    #     squat_tutorial()
    # if st.sidebar.button("Pushup Tutorial"):
    #     pushup_tutorial()
    # if st.sidebar.button("Lunge Tutorial"):
    #     lunge_tutorial()

if __name__ == "__main__":
    main()
