import os
import cv2
import numpy as np
import pytesseract
from fuzzywuzzy import fuzz
import wget
import matplotlib.pyplot as plt
from ultralytics import YOLO
from transformers import AutoTokenizer
from transformers import VitsModel # Replace 'some_library' with the actual library providing VitsModel
import torch
import soundfile as sf
import time

import time
from collections import Counter
import sys  # For exiting the application

# Define and set up the path to store Tesseract OCR data for Malayalam
tessdata_dir = os.path.expanduser("/usr/share/tesseract-ocr/4.00/tessdata/")
if not os.path.exists(tessdata_dir):
    os.makedirs(tessdata_dir)

# URL for downloading Malayalam language data for Tesseract OCR
mal_traineddata_url = 'https://github.com/tesseract-ocr/tessdata_best/raw/main/mal.traineddata'
mal_traineddata_path = os.path.join(tessdata_dir, 'mal.traineddata')

# Download the Malayalam language model if it doesn't exist locally
if not os.path.exists(mal_traineddata_path):
    print("Downloading Malayalam language data...")
    wget.download(mal_traineddata_url, mal_traineddata_path)
    print("\nDownload complete!")

# Set Tesseract data directory environment variable
os.environ['TESSDATA_PREFIX'] = tessdata_dir


# Set up paths for image and label folders
base_path = '/media/irfan/New Volume/Projects/pragathi/ksrtc_intelligent_transport_assistance/ksrtc_intelligent_transport_assistance/dataset with groundtruth for ocr/DATASET/'
label_folder = os.path.join(base_path, 'labels')




#Creating dictionary
def load_data_and_create_dict():
    malayalam_dict = set()  # Initialize an empty set for Malayalam words
    for label_file in os.listdir(label_folder):
        if label_file.endswith('.txt'):  # Process only text files
            with open(os.path.join(label_folder, label_file), 'r', encoding='utf-8') as f:
                word = f.read().strip()  # Read and clean up each word
                if word:
                    malayalam_dict.add(word)  # Add unique word to the set
    return sorted(list(malayalam_dict))  # Return a sorted list of words

#Malayalam tts
def give_me_speech(text):
    import os
    import subprocess
    from transformers import AutoTokenizer
    from transformers import VitsModel
    import torch
    import soundfile as sf

    # Load model and tokenizer
    model = VitsModel.from_pretrained("aoxo/swaram")
    tokenizer = AutoTokenizer.from_pretrained("aoxo/swaram")

    # Create the full text
    a = "അടുത്ത ബസ് "
    b = " വരെ പോകും."
    fulltext = a + text + b
    print(fulltext)

    # Tokenize the input text
    inputs = tokenizer(fulltext, return_tensors="pt")

    # Generate audio
    with torch.no_grad():
        output = model(**inputs).waveform

    # Process the output
    print("Output shape:", output.shape)
    output = output.squeeze(0).cpu().numpy()  # Remove batch dimension and convert to NumPy
    output = output.astype('float32')         # Ensure data type is float32

    # Save the audio for playback
    save_path = "/media/irfan/New Volume/Projects/pragathi/temp_for_audio//output.wav"  # Save in a temporary directory
    sf.write(save_path, output, samplerate=model.config.sampling_rate)
    print(f"Audio saved as {save_path}")

    # Play the audio automatically
    print("Playing audio...")
    try:
        subprocess.run(["aplay", save_path], check=True)  # Use `aplay` for playback
    except FileNotFoundError:
        print("Error: 'aplay' utility not found. Please install it using 'sudo apt install alsa-utils'.")

    # Return the path to the saved audio file
    return save_path

#Function to find the best matching word using fuzzy matching
def find_best_match(ocr_text, malayalam_dict):
    best_match = ""
    best_ratio = 0
    for word in malayalam_dict:
        ratio = fuzz.token_sort_ratio(ocr_text, word)  # Calculate similarity
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = word
    return best_match, best_ratio  # Return best match and its similarity score


# Function to draw bounding boxes and matched text on the image
def draw_boxes(image, boxes, texts, best_matches):
    for box, text, match in zip(boxes, texts, best_matches):
        # Only display text if 'match' is not empty or None
        if match:
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)  # Draw rectangle for box
            cv2.putText(image, match, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)  # Display text
    return image

# Function to log OCR data, best match, and confidence score to a file
def write_data_to_file(file_path, ocr_text, best_match, confidence):
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(f"OCR Text:\n{ocr_text}\n")
        file.write(f"Best Match:\n{best_match}\n")
        file.write(f"Confidence:\n{confidence}%\n")
        file.write("\n") 
# Function to detect text and read it using OCR in real-time from a video feed


def detect_and_read_text(malayalam_dict):
    model = YOLO("/media/irfan/New Volume/Projects/pragathi/yolomodels/bestchechi.pt")  # Load YOLO model for text detection

    cap = cv2.VideoCapture(0)  # Open video feed from the default camera
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Initialize a dictionary to store word frequencies
    word_frequencies = {}

    while True:
        ret, frame = cap.read()  # Capture a frame from the video feed
        if not ret:
            print("Error: Could not read frame from camera.")
            break

        cv2.imshow('Camera Feed', frame)  # Display the live video feed

        results = model(frame)  # Perform detection on the frame
        bboxes = results[0].boxes.xyxy.cpu().numpy()  # Extract bounding boxes for detected objects

        # Initialize lists to store OCR text, matches, confidence scores, and boxes
        texts, best_matches, confidences, image_with_boxes, boxes = [], [], [], frame.copy(), []

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox[:4]  # Get bounding box coordinates
            if results[0].boxes.conf[0] > 0.3:  # Filter by confidence threshold
                roi = frame[int(y1):int(y2), int(x1):int(x2)]  # Extract region of interest

                # Preprocess ROI for OCR
                roi = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)  # Resize
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                denoised_roi = cv2.fastNlMeansDenoising(gray_roi)  # Denoise image
                thresh_roi = cv2.threshold(denoised_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]  # Threshold
                
                # Display the preprocessed images
                cv2.imshow('ROI', roi)
                cv2.imshow('Gray ROI', gray_roi)
                cv2.imshow('Denoised ROI', denoised_roi)
                cv2.imshow('Thresholded ROI', thresh_roi)

                # Save preprocessed images for debugging
                path = "/media/irfan/New Volume/Projects/pragathi/debug/"
                cv2.imwrite(path + 'roi.jpg', roi)
                cv2.imwrite(path + 'gray_roi.jpg', gray_roi)
                cv2.imwrite(path + 'denoised_roi.jpg', denoised_roi)
                cv2.imwrite(path + 'thresh_roi.jpg', thresh_roi)

                # Perform OCR on thresholded ROI
                config = '--oem 1 --psm 6 -l  mal'  # Tesseract OCR configuration for Malayalam
                ocr_text = pytesseract.image_to_string(thresh_roi, config=config).strip() ##CHECK WITH AND WITHOUT .STRIP() # Extract text from ROI

                if ocr_text.strip():  # If OCR text is not empty
                    best_match, confidence = find_best_match(ocr_text, malayalam_dict)  # Find best match
                    texts.append(ocr_text)
                    best_matches.append(best_match)
                    confidences.append(confidence)
                    boxes.append((int(x1), int(y1), int(x2), int(y2)))  # Store bounding box coordinates

                  
                    # Print OCR results
                    print(f"OCR Text: {ocr_text}")
                    print(f"Best Match: {best_match}")
                    print(f"Confidence: {confidence}%")
                    print("---")
       

        if any(conf > 70 for conf in confidences):  # Only proceed if confidence is high enough
            print("Detected Texts:")
            bmlist = []
            for text, match, conf, box in zip(texts, best_matches, confidences, boxes):
                if conf > 70:
                    print(f"OCR Text: {text}")
                    print(f"Best Match: {match}")
                    print(f"Confidence: {conf}%")
                    print(f"Bounding Box: {box}")
                    bmlist.append(match)
                    output_path="/media/irfan/New Volume/Projects/pragathi/output/out.txt"
                    write_data_to_file(output_path, ocr_text , best_match ,confidence)  # Write data to file

            for match in bmlist:
                word_frequencies[match] = word_frequencies.get(match, 0) + 1

            # Trigger TTS if a word crosses the threshold frequency
            if word_frequencies:
                max_word = max(word_frequencies, key=word_frequencies.get)
                max_count = word_frequencies[max_word]
                print(f"Most frequent word: {max_word} with count: {max_count}")

                if max_count >= 3:  # Threshold for triggering TTS
                    give_me_speech(max_word)
                    time.sleep(30)  # Wait for 30 seconds
                    word_frequencies.clear()  # Reset word frequencies


                    

            frame_with_boxes = draw_boxes(frame.copy(), boxes, texts, best_matches)  # Draw boxes on frame
            cv2.imshow('Text Detection', frame_with_boxes)  # Display the frame with text boxes

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit loop if 'q' key is pressed
            break

    cap.release()  # Release the video capture object
    cv2.destroyAllWindows()  # Close all OpenCV windows
# Load Malayalam words into dictionary
malayalam_dict = load_data_and_create_dict()

# Start real-time text detection and reading                
detect_and_read_text(malayalam_dict)