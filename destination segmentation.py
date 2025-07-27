import pytesseract
from PIL import Image, ImageDraw
import numpy as np

def find_threshold(heights):
    # Calculate the 60th percentile as a good threshold
    threshold = sum(heights) / len(heights)
    return threshold

def visualize_and_categorize_boxes(image_path):
    # Open and preprocess the image
    image = Image.open(image_path)
    #image = image.convert("L")
    draw = ImageDraw.Draw(image)

    # Configure Tesseract for OCR
    custom_config = r'--oem 1 --psm 6 -l mal'
    ocr_data = pytesseract.image_to_boxes(image, config=custom_config)
    image_width, image_height = image.size

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
    print(f"Calculated Threshold: {threshold}")
    print(characters)

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

        # Draw the bounding box with different colors for each category
        color = {"small": "green", "large": "red"}[category]
        draw.rectangle([(x1, y2), (x2, y1)], outline=color, width=2)

        # Draw character label and height
        draw.text((x1, y2 - 20), f"{char} ({box_height})", fill=color)

    # Display the image
    print("large ",large_chars)
    image.show()
    

    # Print the categorized results
    for category, boxes in categories.items():
        print(f"{category.capitalize()} Boxes:")
        for box in boxes:
            char, x1, y1, x2, y2, box_height = box
            print(f"  Char: {char}, Height: {box_height}, Coords: ({x1}, {y1}, {x2}, {y2})")

   

   

# Example usage
image_path = " "
visualize_and_categorize_boxes(image_path)
