

# Now you can import everything else
import torch
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEPETrainer

if __name__ == '__main__':
    
    # 1. Initialize a detection model from a config
    model = YOLOE("yoloe-26l.yaml")

    # 2. Load weights from a pretrained segmentation checkpoint 
    model.load("yoloe-26l-seg.pt")

    # 3. Fine-tune on your detection dataset
    results = model.train(
        data="/mnt/d/AeroDefence/augmented/data.yaml",  
        epochs=80,
        batch=50,       
        patience=10,
        trainer=YOLOEPETrainer,
        project="ASIoU_Experiment",
        name="yoloe-26l-asiou_loss",
        device=[0, 1]       
    )