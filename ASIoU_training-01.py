import os

# Enable detailed error traceback from DDP subprocesses
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from ultralytics import YOLOE

if __name__ == "__main__":
    # Resume training from the last checkpoint (loads full training state: weights + optimizer + epoch)

    model = YOLOE("yoloe-26l.yaml")

    # # Load weights from a pretrained segmentation checkpoint (same scale)
    model.load("yoloe-26l-seg.pt")

    # Resume fine-tuning — resume=True restores project/name/epochs from checkpoint automatically
    results = model.train(
        data="/mnt/d/AeroDefence/augmented/data.yaml",
        # data = '/mnt/e/Roboflow-2025-10-12/augmented/data.yaml',
        epochs=50,
        batch=35,
        patience=10,
        device=[0, 1],
        project="ASIoU_Experiment",
        name="yoloe-26l-update_AS-asiou_loss",
    )
