import cv2
from matplotlib import pyplot as plt
import numpy as np
from ultralytics import YOLO
import yaml
import os
import datetime

#SAMPLE_BACKGROUND_1 = os.path.join('sample_data', 'sample_background1.mp4')
#SAMPLE_BACKGROUND_2 = os.path.join('sample_data', 'sample_background2.mp4')
CONFIG_FILE = 'config.yaml'
DATA_DIRECTORY = f'data'


def main():
  # Load configuration
  with open(CONFIG_FILE, 'r') as file:
    yaml_config = yaml.safe_load(file)

  # cv2 read from webcam
  cap = cv2.VideoCapture(0)
  fps = cap.get(cv2.CAP_PROP_FPS)  # get fps from cap
  frame_number = fps / yaml_config["T1_FPS"]
  frame_counter = 0
  print(f"fps: {fps}, frame_number: {frame_number}")
  print(DATA_DIRECTORY)
  
  
  capture_and_save_frames(cap, frame_number, frame_counter)
  median_background_inference(DATA_DIRECTORY)
  cv2.destroyAllWindows()

def capture_and_save_frames(cap, frame_number, frame_counter, background_region=0.01, model = YOLO("yolov8n-seg.pt")):
    def capture_and_save_frames(cap, frame_number, frame_counter, background_region=0.01, model=YOLO("yolov8n-seg.pt")):
      """
      Captures frames from a video capture object, processes them using a segmentation model, and saves frames 
      where the background region difference exceeds a specified threshold.
      Args:
        cap (cv2.VideoCapture): Video capture object.
        frame_number (int): The interval at which frames are processed (e.g., every 60th frame).
        frame_counter (int): Counter to keep track of the number of frames processed.
        background_region (float, optional): The threshold for the background region difference ratio to save the frame. Defaults to 0.01.
        model (YOLO, optional): The YOLO segmentation model to use for processing frames. Defaults to YOLO("yolov8n-seg.pt"). This can be passed in main using T2_BACKGROUND_REGION in config.yaml.
      Returns:
        None
      """
    while cap.isOpened() and frame_counter < 1000:
      success, frame = cap.read()
      #cv2.imshow('webcam frame', frame)
      if success:
          # Increment frame counter
          frame_counter += 1

          # Process every 60th frame
          if frame_counter % frame_number == 0:
              # Run segmentation model through the frame
              results = model(frame)

              # Get segmentation mask
              if results and results[0].masks:
                  segmentation_mask = results[0].masks.data[0].cpu().numpy()

                  # Create bg_mask with the same size as the frame
                  bg_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

                  # Resize segmentation_mask to match frame's shape
                  resized_segmentation_mask = cv2.resize(segmentation_mask.astype(np.uint8), (frame.shape[1], frame.shape[0]))

                  # Create new mask
                  new_mask = np.logical_or(bg_mask, np.logical_not(resized_segmentation_mask))
                  new_mask = new_mask.astype(np.uint8)

                  # Calculate difference
                  diff_ratio = np.sum(np.abs(new_mask - bg_mask)) / (frame.shape[0] * frame.shape[1])

                  # Save frame as an image if the difference ratio is greater than 0.01
                  if diff_ratio > background_region:
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S%f")
                        cv2.imwrite(os.path.join(DATA_DIRECTORY, f"{timestamp}.jpg"), frame)
      else:
        break
    cap.release()
      
def median_background_inference(directory):
  """Reads all images in a given directory using OpenCV."""
  image_files = [f for f in os.listdir(directory) if f.endswith('.jpg')]
  images = []
  for i in range(min(20,len(image_files))):
    image_path = os.path.join(directory, image_files[i])
    img = cv2.imread(image_path)
    if img is not None:
      images.append(img)
    else:
      print(f"Could not read image: {image_path}")
  fa_background = np.median(images, axis=0).astype(np.uint8)
  cv2.imwrite('result/background_median.png', fa_background)

  plt.imshow(cv2.cvtColor(fa_background, cv2.COLOR_BGR2RGB))
  plt.show()
  return fa_background

if __name__ == "__main__":
    main()