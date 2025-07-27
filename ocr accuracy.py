import os
import pytesseract
from jiwer import wer, cer

# Configure Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
tessdata_dir_config = r'--tessdata-dir "C:\Program Files\Tesseract-OCR\tessdata"'  # Update the path if needed

# Directories containing the dataset
images_folder = r"C:\Users\ggthe\OneDrive\Desktop\processed\images" # Folder with test images
labels_folder = r"C:\Users\ggthe\OneDrive\Desktop\processed\labels" # Folder with ground truth text files
language = "custom1"  # Malayalam language code

# Initialize aggregated strings
all_ocr_text = ""
all_ground_truth = ""

# Process each image in the folder
for image_file in os.listdir(images_folder):
    if image_file.endswith((".png", ".jpg", ".jpeg", ".tif")):
        # Construct corresponding label file path
        base_name = os.path.splitext(image_file)[0]
        label_file = os.path.join(labels_folder, f"{base_name}.txt")

        # Check if label file exists
        if not os.path.exists(label_file):
            print(f"Label file missing for {image_file}")
            continue

        # Perform OCR on the image
        image_path = os.path.join(images_folder, image_file)
        ocr_text = pytesseract.image_to_string(image_path, lang=language, config=tessdata_dir_config)

        # Read the ground truth text
        with open(label_file, "r", encoding="utf-8") as f:
            ground_truth = f.read()

        # Aggregate OCR and ground truth texts
        all_ocr_text += ocr_text.strip() + " "
        all_ground_truth += ground_truth.strip() + " "

# Calculate accuracy metrics
if all_ground_truth.strip() and all_ocr_text.strip():
    overall_wer = wer(all_ground_truth, all_ocr_text)
    overall_cer = cer(all_ground_truth, all_ocr_text)

    # Print results
    print("\n--- Overall Dataset Performance ---")
    print(f"Overall Word Error Rate (WER): {overall_wer:.2%}")
    print(f"Overall Character Error Rate (CER): {overall_cer:.2%}")
else:
    print("No valid data to process.")
