from ultralytics import YOLO

if __name__ == '__main__':
    # Initialize the YOLOv8 Nano model using pre-trained weights
    model = YOLO("yolov8n.pt")  # Using the YOLOv8 Nano model with pre-trained weights

    # Train the model (fine-tuning) on your custom dataset
    model.train(
        data="D:/Projects/train_with_noclass/data.yaml",        # Path to your data.yaml file
        epochs=100,              # Total number of epochs
        batch=8,                 # Batch size
        imgsz=640,               # Image size
        workers=2,               # Number of workers for data loading
        device=0                 # Use GPU; set to 'cpu' if no GPU is available
    )

    print("Training of YOLOv8 Nano with pre-trained weights completed.")
