#!/usr/bin/env python3
"""
Script to convert YOLO model to ONNX format for TypeScript deployment
"""

from ultralytics import YOLO
import torch


def convert_yolo_to_onnx(model_path, output_path, img_size=640):
    """
    Convert YOLO model to ONNX format

    Args:
        model_path: Path to your trained YOLO model (.pt file)
        output_path: Output path for ONNX model (.onnx file)
        img_size: Input image size (default 640)
    """

    print(f"Loading YOLO model from: {model_path}")
    model = YOLO(model_path)

    print(f"Converting to ONNX format...")
    print(f"Input size: {img_size}x{img_size}")

    # Export to ONNX
    success = model.export(
        format='onnx',
        imgsz=img_size,
        optimize=True,
        half=False,  # Use FP32 for better compatibility
        dynamic=False,  # Static shape for better performance
        simplify=True,  # Simplify the model
        opset=11  # ONNX opset version (11 is widely supported)
    )

    if success:
        # The exported file will be in the same directory as the original model
        # with .onnx extension
        model_dir = '/'.join(model_path.split('/')[:-1])
        model_name = model_path.split('/')[-1].replace('.pt', '.onnx')
        generated_onnx_path = f"{model_dir}/{model_name}"

        print(f"✅ Model successfully converted!")
        print(f"ONNX model saved at: {generated_onnx_path}")
        print(f"Model input shape: [1, 3, {img_size}, {img_size}]")
        print(f"Model outputs: bounding boxes with format [x, y, w, h, confidence, class_prob]")

        return generated_onnx_path
    else:
        print("❌ Failed to convert model")
        return None


if __name__ == "__main__":
    # Update this path to your trained model
    model_path = "/Users/swaminathang/yolov5/runs/detect/train19/weights/best.pt"

    # Convert model
    onnx_path = convert_yolo_to_onnx(
        model_path=model_path,
        output_path="license_plate_detector.onnx",
        img_size=640
    )

    if onnx_path:
        print(f"\n🎉 Ready for TypeScript deployment!")
        print(f"Use this ONNX model file: {onnx_path}")