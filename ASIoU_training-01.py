# Now you can import everything else
from ultralytics import YOLOE

# from ultralytics import YOLO
from ultralytics.models.yolo.yoloe import YOLOEPETrainer

if __name__ == "__main__":
    # ASIoU loss variants to train sequentially
    asiou_variants = ["lambda-eiou", "lambda-half-eiou"]

    # Base model path (used to reload fresh weights for each variant)
    # base_model_path = "/mnt/c/Users/aksha/AeroDefence/YOLOE/runs/detect/ASIoU_Experiment/yoloe-26l-asiou_loss-2.0/weights/last.pt"

    for variant in asiou_variants:
        print(f"\n{'=' * 60}")
        print(f"  Starting training with ASIoU mode: {variant}")
        print(f"{'=' * 60}\n")

        # Reload model fresh for each variant
        model = YOLOE("yoloe-26l.yaml")

        model.load("yoloe-26l-seg.pt")

        results = model.train(
            data="/mnt/e/Moe-Akshat/augmented/data.yaml",
            epochs=80,
            batch=35,
            patience=10,
            trainer=YOLOEPETrainer,
            asiou_mode=variant,
            project="ASIoU_Experiment",
            name=f"yoloe-26l-{variant}",
            device=[0, 1],
            workers=8,
        )

        print(f"\n{'=' * 60}")
        print(f"  Completed training with ASIoU mode: {variant}")
        print(f"{'=' * 60}\n")
