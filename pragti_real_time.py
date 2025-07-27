import os
import cv2
import numpy as np
import pytesseract
from fuzzywuzzy import fuzz
import matplotlib.pyplot as plt
from ultralytics import YOLO
import re
from PIL import Image, ImageDraw
import sounddevice as sd  # Import the sounddevice library here
from transformers import AutoTokenizer
from transformers import VitsModel  # Replace 'some_library' with the actual library providing VitsModel
import torch
import soundfile as sf
from collections import Counter
# Define and set up the path to store Tesseract OCR data for Malayalam
tessdata_dir = os.path.expanduser('/home/hp/Desktop/pragti/tessdata')
if not os.path.exists(tessdata_dir):
    os.makedirs(tessdata_dir)

# Set Tesseract data directory environment variable
os.environ['TESSDATA_PREFIX'] = tessdata_dir


# Set up paths for image and label folders
base_path = '/home/hp/Desktop/pragti'
label_folder = os.path.join(base_path, 'labels')

def find_threshold(heights):
    threshold=sum(heights)/len(heights)
    if not heights:
        print("Warning: No bounding box heights detected.")
        return 0  # Default threshold if no heights available
    count_above_threshold = sum(h > threshold for h in heights)
    percentage = (count_above_threshold / len(heights)) * 100
    return np.percentile(heights, 100-percentage)

def visualize_and_categorize_boxes(thresh_roi):
   

    image_pil = Image.fromarray(thresh_roi)  # Convert ndarray to PIL Image
    draw = ImageDraw.Draw(image_pil)
    custom_config = '--oem 1 --psm 6 -l c1'
    ocr_data = pytesseract.image_to_boxes(thresh_roi, config=custom_config)
    image_height, image_width = thresh_roi.shape[:2]  # Get height and width

    # List to store bounding box heights for threshold calculation
    box_heights = []

    # Categories for bounding boxes
    categories = {
        "small": [],
        "large": []
    }
    characters = []

    # Strings to store characters for each category
    small_chars = ""
    large_chars = ""


    for line in ocr_data.splitlines():
        parts = line.split()
        char = parts[0]
        x1, y1, x2, y2 = map(int, parts[1:5])

        # Filter out non-Malayalam characters
        if not '\u0D00' <= char <= '\u0D7F':
            continue

        # Transform coordinates to Pillow's top-left origin
        y1 = image_height - y1
        y2 = image_height - y2

        # Calculate height of the bounding box
        box_height = abs(y1 - y2)
        box_heights.append(box_height)

        # Store the character and its bounding box details
        characters.append([char, x1, y1, x2, y2, box_height])

    # Find the threshold based on the heights
    threshold = find_threshold(box_heights)
   
    # Classify the bounding boxes into small and large
    for g in range(len(characters)):
        char, x1, y1, x2, y2, box_height = characters[g]
        aflag=0
        bflag=0
        acount=0
        bcount=0
        # Get the heights of the previous and next characters, if they exist
        bbox_height = characters[g - 1][5] if g > 0 else None
        abox_height = characters[g + 1][5] if g < len(characters) - 1 else None
        #a4inibox_height = characters[a][5] if a < len(characters) - 1 else None
        for i in range(0,10):
            if g+i < len(characters):
             if characters[g+i][5] > threshold :
                    acount+=1
            else:
                break
        if acount > 5:
            aflag=1

        for i in range(0,10):
             if g-i > 0:
                 if characters[g-i][5] > threshold :
                        bcount+=1
                 else:
                     break
        if bcount > 5:
          bflag=1
        # Categorize the bounding box
        if box_height > threshold and (
                   ((bbox_height is None or (bbox_height > threshold or aflag)) and
                   (abox_height is None or abox_height > threshold or bflag)) 
               ):
            category = "large"
            large_chars += char
        
        else:
            category = "small"
            small_chars += char
        

        # Append to the appropriate category
        categories[category].append((char, x1, y1, x2, y2, box_height))

      


    # Print the characters in each category
    return large_chars

def segment_ocr_text(ocr_text):
    # Normalize the text by removing unwanted characters
    normalized_text = re.sub(r'[^ാ-ൗക-ഹഅ-ൿ\s-]', '', ocr_text)  # Keep Malayalam characters, spaces, and dashes

    # Split the text into words using spaces, line breaks, or dashes
    words = re.split(r'\s+|-', normalized_text)

    # Remove any empty strings resulting from splitting
    words = [word for word in words if word]

    return words


def find_best_match_for_route(primary, folder_path):
    """Find the best match for the primary word in text files from the folder."""
    best_match = None
    best_score = 0

    for filename in os.listdir(folder_path):
        #if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                file_content = file.read()
                for line in file_content.splitlines():
                    similarity = fuzz.token_sort_ratio(primary, line.strip())
                    if similarity > best_score:
                        best_score = similarity
                        best_match = line.strip()

    return best_match, best_score
# Function to draw bounding boxes and matched text on the image
def draw_boxes(image, boxes, texts, best_matches):
    for box, text, match in zip(boxes, texts, best_matches):
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)  # Draw rectangle for box
        #cv2.putText(image, match, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)  # Display text
    return image
from fuzzywuzzy import fuzz

def give_me_speech(text):
    

    # Load model and tokenizer
    model = VitsModel.from_pretrained("aoxo/swaram")
    tokenizer = AutoTokenizer.from_pretrained("aoxo/swaram")

    
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt")

    # Generate audio
    with torch.no_grad():
        output = model(**inputs).waveform

    # Process the output
    print("Output shape:", output.shape)
    output = output.squeeze(0).cpu().numpy()  # Remove batch dimension and convert to NumPy
    output = output.astype('float32')         # Ensure data type is float32

    # Save the audio for playback
    sf.write("output.wav", output, samplerate=model.config.sampling_rate)
    print("Audio saved as output.wav")

    # Play the audio in real-time
    print("Playing audio...")
    sd.play(output, samplerate=model.config.sampling_rate)
    sd.wait()  # Wait until the audio is done playing

    # Return the path to the saved audio file
    return "output.wav"

        

# Function to detect text and read it using OCR in real-time from a video feed
def detect_and_read_text():
    
    model = YOLO('/home/hp/Desktop/pragti/yolo model/best.pt')  # Load YOLO model for text detection
    end_list=[]

    cap = cv2.VideoCapture("/home/hp/Desktop/pragti/test1.mp4")  # Open video feed from the default camera
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
     
  
        ret, frame = cap.read()  # Capture a frame from the video feed
        if not ret:
            print("Error: Could not read frame from camera.")
            break

        results = model(frame)  # Perform detection on the frame
        bboxes = results[0].boxes.xyxy.cpu().numpy()  # Extract bounding boxes for detected objects
        # Initialize lists to store OCR text, matches, confidence scores, and boxes
        texts, best_matches, confidences, boxes = [],[],[],[]

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox[:4]  # Get bounding box coordinates
            if results[0].boxes.conf[0] > 0.3:  # Filter by confidence threshold
                roi = frame[int(y1):int(y2), int(x1):int(x2)]  # Extract region of interest

                # Preprocess ROI for OCR
                roi = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)  # Resize
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                denoised_roi = cv2.fastNlMeansDenoising(gray_roi)  # Denoise image
                thresh_roi = cv2.threshold(denoised_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]  # Threshold
                
            

                # Save preprocessed images for debugging
                path = '/home/hp/Desktop/pragti/processed'
                cv2.imwrite(path + 'roi.jpg', roi)
                cv2.imwrite(path + 'gray_roi.jpg', gray_roi)
                cv2.imwrite(path + 'denoised_roi.jpg', denoised_roi)
                cv2.imwrite(path + 'thresh_roi.jpg', thresh_roi)

                # Perform OCR on thresholded ROI
                config = '--oem 1 --psm 6 -l  c1'  # Tesseract OCR configuration for Malayalam
                ocr_text = pytesseract.image_to_string(thresh_roi, config=config)  # Extract text from ROI

                if ocr_text.strip():  # If OCR text is not empty
                    segmented_words=segment_ocr_text(ocr_text)
                    print(segmented_words)
                    # Base folder path
                    base_folder_path = "/home/hp/Desktop/main"  
                     # Path for primary folder
                    folder_path_p = os.path.join(base_folder_path, "primary")
                    raw_match_p=visualize_and_categorize_boxes(thresh_roi)
                    #print("extracted primary: ",raw_match_p)
                    
                    ocr_match_p=None
                    best=0 
                    for p in segmented_words:
                         score = fuzz.token_sort_ratio(raw_match_p,p)
                         if score > best:
                                          
                           ocr_match_p=p
                           best=score
                    #print(ocr_match)
                    #print(best_match_p)
                    best_match_p,best_score_p =find_best_match_for_route(ocr_match_p,folder_path_p)
                    if ocr_match_p is not None:
                        segmented_words.remove(ocr_match_p)
                        secondary = ' '.join(segmented_words)
                    else:
                        print("NO MATCHES FOR PRIMARY")
                        break

                  
                    if best_match_p is None:
                        break
                    folder_path_s = os.path.join(base_folder_path, "secondary", best_match_p)
                       # Ensure the folder exists
                    

                    if not os.path.exists(folder_path_s):
                        print(f"Error: The folder '{folder_path_s}' does not exist.")
                    else:
                        best_match_s, best_score_s = find_best_match_for_route(secondary, folder_path_s)

                    texts.append(ocr_text)
                    best_matches.append(best_match_p)
                    confidences.append(best_score_p)
                    boxes.append((int(x1), int(y1), int(x2), int(y2)))  # Store bounding box coordinates

                

        if any(conf > 50 for conf in confidences):  # Only proceed if confidence is high enough
            print("Detected Texts:")
            for text, match, conf, box in zip(texts, best_matches, confidences, boxes):
                if conf > 40:

                    if best_match_s is not None and best_match_p is not None:

                                audio_out = best_match_s + " വഴി " + best_match_p + " പോകുന്ന ബസ് വന്നുകൊണ്ടിരിക്കുന്നു"
                                end_list.append(audio_out)
                                freq = Counter(end_list)
                                max_freq_element = max(freq, key=freq.get)
                                frequency = freq[max_freq_element] 
                                if frequency > 6:
                                    max_freq_element_final = max(freq, key=freq.get)
                                    print(freq)
                                    cap.release()  # Release the video capture object
                                    cv2.destroyAllWindows()  # Close all OpenCV windows
                                    give_me_speech(max_freq_element_final)
                                    detect_and_read_text()

                                else:
                                    continue
                    else:
                                print("One or both matches are None. Skipping execution.")

# Start real-time text detection and reading
detect_and_read_text()
