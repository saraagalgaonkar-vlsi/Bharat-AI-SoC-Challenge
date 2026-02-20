# Bharat-AI-SoC-Challenge
Real-Time Object Detection using YOLO Tiny v3 on Ultra96 with NNgen Acceleration

This project was developed as part of the Bharat AI SoC Challenge, where we implemented real-time object detection using YOLO Tiny v3 on the Ultra96 board.

Instead of running the neural network purely on the ARM processor, we used NNgen to convert the trained YOLO Tiny v3 model into a hardware accelerator and deployed it onto the FPGA fabric of the Ultra96 (Zynq UltraScale+ MPSoC).

This project demonstrates efficient hardware-software co-design for edge AI applications.

🧠 Model: YOLO Tiny v3

We implemented YOLO Tiny v3, a lightweight object detection model optimized for embedded systems.

Why YOLO Tiny v3?

Fast inference

Smaller architecture compared to full YOLOv3

Suitable for FPGA acceleration

Real-time detection capability

The model performs:

Object classification

Bounding box regression

Confidence prediction

🖥 Hardware Platform

The system runs on the Ultra96, built around the Xilinx Zynq UltraScale+ MPSoC.

Key Components Used:

ARM Cortex-A53 (Processing System – PS)

FPGA Fabric (Programmable Logic – PL)

DDR Memory

AXI Interfaces

The Ultra96 enables heterogeneous computing:

Component	Role
ARM (PS)	Preprocessing, control, postprocessing
FPGA (PL)	YOLO Tiny v3 acceleration via NNgen
⚙️ Role of NNgen

We used NNgen to:

Convert the trained YOLO Tiny v3 model into a hardware description

Generate FPGA-compatible accelerator logic

Apply quantization for efficient fixed-point inference

Automatically map CNN layers to hardware modules

NNgen helped bridge the gap between deep learning models and FPGA implementation without manually designing every convolution block.

🏗 System Architecture
Data Flow:

Camera captures input frame

ARM resizes and normalizes the image

Input tensor passed to FPGA accelerator

NNgen-generated hardware performs convolution and feature extraction

Output returned to ARM

Postprocessing (NMS, bounding boxes) executed

Results displayed

The computationally intensive layers (convolution, activation, pooling) are accelerated in hardware, significantly improving performance compared to CPU-only execution.

🛠 Tools & Technologies
Hardware Design

Vivado Design Suite

Vitis

NNgen

AXI4 interfaces

Software

Python (control & preprocessing)

OpenCV

NumPy

Linux on Ultra96

🚀 Implementation Flow
1️⃣ Model Preparation

YOLO Tiny v3 trained

Model exported

Converted using NNgen

Quantized for fixed-point hardware execution

2️⃣ Hardware Generation

NNgen generated accelerator IP

Integrated into Vivado block design

Connected via AXI

Bitstream generated

3️⃣ Deployment

Bitstream programmed onto Ultra96

ARM application controls inference

Real-time object detection achieved
