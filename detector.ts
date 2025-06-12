import * as ort from "onnxruntime-node";
import * as fs from "fs";
import * as path from "path";
import { spawn } from "child_process";
import sharp from "sharp";

// Types for our detection system
interface DetectionResult {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  confidence: number;
}

interface PlateDetection {
  unique_plate_id: string;
  frame_number: number;
  timestamp_seconds: number;
  license_plate_bbox: [number, number, number, number];
  detection_score: number;
}

interface VideoProcessingResult {
  video_path: string;
  video_info: {
    total_frames: number;
    fps: number;
    duration_seconds: number;
    frame_skip: number;
  };
  detections: PlateDetection[];
  processing_summary: {
    frames_processed: number;
    total_plates_detected: number;
    total_processing_time_seconds: number;
  };
}

export class LicensePlateDetectorTS {
  private session: ort.InferenceSession | null = null;
  private plateCounter: number = 0;
  private inputWidth: number = 640;
  private inputHeight: number = 640;

  constructor(
    private modelPath: string,
    private minConfidence: number = 0.25
  ) {}

  /**
   * Initialize the ONNX model
   */
  async initialize(): Promise<void> {
    try {
      console.log(`Loading ONNX model from: ${this.modelPath}`);

      // Create session with optimization
      this.session = await ort.InferenceSession.create(this.modelPath, {
        executionProviders: ["cpu"], // Use 'cuda' if GPU is available
        graphOptimizationLevel: "all",
        enableCpuMemArena: true,
        enableMemPattern: true,
      });

      console.log("ONNX model loaded successfully");

      // Get input dimensions from model
      const inputNames = this.session.inputNames;
      if (inputNames.length > 0) {
        const inputInfo = this.session.inputNames[0];
        console.log(`Model input: ${inputInfo}`);
      }
    } catch (error) {
      throw new Error(`Failed to load ONNX model: ${error}`);
    }
  }

  /**
   * Preprocess image for YOLO inference using Sharp
   */
  private async preprocessImage(imageBuffer: Buffer): Promise<Float32Array> {
    try {
      // Resize and convert to RGB using Sharp
      const { data, info } = await sharp(imageBuffer)
        .resize(this.inputWidth, this.inputHeight)
        .raw()
        .toBuffer({ resolveWithObject: true });

      // Convert to Float32Array and normalize (0-255 -> 0-1)
      const imageData = new Float32Array(
        3 * this.inputWidth * this.inputHeight
      );

      // Convert from HWC (Height-Width-Channels) to CHW (Channels-Height-Width) format
      // and normalize pixel values
      for (let c = 0; c < 3; c++) {
        for (let h = 0; h < this.inputHeight; h++) {
          for (let w = 0; w < this.inputWidth; w++) {
            const hwcIndex = (h * this.inputWidth + w) * 3 + c;
            const chwIndex =
              c * this.inputHeight * this.inputWidth + h * this.inputWidth + w;
            imageData[chwIndex] = data[hwcIndex] / 255.0;
          }
        }
      }

      return imageData;
    } catch (error) {
      console.error("Error preprocessing image:", error);
      throw error;
    }
  }

  /**
   * Run YOLO inference on preprocessed image
   */
  private async runInference(
    imageData: Float32Array
  ): Promise<DetectionResult[]> {
    if (!this.session) {
      throw new Error("Model not initialized");
    }

    try {
      // Create input tensor
      const inputTensor = new ort.Tensor("float32", imageData, [
        1,
        3,
        this.inputHeight,
        this.inputWidth,
      ]);

      // Run inference
      const results = await this.session.run({ images: inputTensor });

      // Process output (this depends on your YOLO model output format)
      const output =
        results.output0 || results.output || Object.values(results)[0];

      if (!output) {
        return [];
      }

      return this.processYoloOutput(
        output.data as Float32Array,
        output.dims as number[]
      );
    } catch (error) {
      console.error("Inference error:", error);
      return [];
    }
  }

  /**
   * Process YOLO model output to extract detections
   */
  private processYoloOutput(
    outputData: Float32Array,
    outputDims: number[]
  ): DetectionResult[] {
    const detections: DetectionResult[] = [];

    // YOLO output format is typically [batch, anchors, 5+classes]
    // where 5 = [x, y, w, h, confidence]
    const numDetections = outputDims[1];
    const numElements = outputDims[2];

    for (let i = 0; i < numDetections; i++) {
      const offset = i * numElements;

      const x = outputData[offset];
      const y = outputData[offset + 1];
      const w = outputData[offset + 2];
      const h = outputData[offset + 3];
      const confidence = outputData[offset + 4];

      if (confidence >= this.minConfidence) {
        // Convert center coordinates to corner coordinates
        const x1 = x - w / 2;
        const y1 = y - h / 2;
        const x2 = x + w / 2;
        const y2 = y + h / 2;

        detections.push({
          x1: x1 * this.inputWidth,
          y1: y1 * this.inputHeight,
          x2: x2 * this.inputWidth,
          y2: y2 * this.inputHeight,
          confidence,
        });
      }
    }

    return this.applyNMS(detections);
  }

  /**
   * Apply Non-Maximum Suppression to remove overlapping detections
   */
  private applyNMS(
    detections: DetectionResult[],
    iouThreshold: number = 0.5
  ): DetectionResult[] {
    if (detections.length === 0) return [];

    // Sort by confidence
    detections.sort((a, b) => b.confidence - a.confidence);

    const keep: DetectionResult[] = [];
    const suppress = new Set<number>();

    for (let i = 0; i < detections.length; i++) {
      if (suppress.has(i)) continue;

      keep.push(detections[i]);

      for (let j = i + 1; j < detections.length; j++) {
        if (suppress.has(j)) continue;

        const iou = this.calculateIoU(detections[i], detections[j]);
        if (iou > iouThreshold) {
          suppress.add(j);
        }
      }
    }

    return keep;
  }

  /**
   * Calculate Intersection over Union (IoU) between two bounding boxes
   */
  private calculateIoU(box1: DetectionResult, box2: DetectionResult): number {
    const x1 = Math.max(box1.x1, box2.x1);
    const y1 = Math.max(box1.y1, box2.y1);
    const x2 = Math.min(box1.x2, box2.x2);
    const y2 = Math.min(box1.y2, box2.y2);

    if (x2 <= x1 || y2 <= y1) return 0;

    const intersection = (x2 - x1) * (y2 - y1);
    const area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
    const area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);
    const union = area1 + area2 - intersection;

    return intersection / union;
  }

  /**
   * Extract frames from video using FFmpeg
   */
  private async extractFrames(
    videoPath: string,
    frameSkip: number = 3
  ): Promise<{
    frames: { frameNumber: number; imagePath: string }[];
    videoInfo: { fps: number; totalFrames: number; duration: number };
  }> {
    const tempDir = path.join(__dirname, "temp_frames");

    // Create temp directory
    if (!fs.existsSync(tempDir)) {
      fs.mkdirSync(tempDir, { recursive: true });
    }

    // Get video info first
    const videoInfo = await this.getVideoInfo(videoPath);

    return new Promise((resolve, reject) => {
      const outputPattern = path.join(tempDir, "frame_%06d.jpg");

      // Use FFmpeg to extract frames
      const ffmpeg = spawn("ffmpeg", [
        "-i",
        videoPath,
        "-vf",
        `select=not(mod(n\\,${frameSkip}))`,
        "-vsync",
        "vfr",
        "-q:v",
        "2",
        outputPattern,
        "-y",
      ]);

      ffmpeg.on("close", (code) => {
        if (code === 0) {
          // Read extracted frames
          const frameFiles = fs
            .readdirSync(tempDir)
            .filter(
              (file) => file.startsWith("frame_") && file.endsWith(".jpg")
            )
            .sort()
            .map((file, index) => ({
              frameNumber: index * frameSkip,
              imagePath: path.join(tempDir, file),
            }));

          resolve({ frames: frameFiles, videoInfo });
        } else {
          reject(new Error(`FFmpeg failed with code ${code}`));
        }
      });

      ffmpeg.on("error", reject);
    });
  }

  /**
   * Get video information using FFprobe
   */
  private async getVideoInfo(
    videoPath: string
  ): Promise<{ fps: number; totalFrames: number; duration: number }> {
    return new Promise((resolve, reject) => {
      const ffprobe = spawn("ffprobe", [
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        videoPath,
      ]);

      let stdout = "";
      ffprobe.stdout.on("data", (data) => {
        stdout += data;
      });

      ffprobe.on("close", (code) => {
        if (code === 0) {
          try {
            const info = JSON.parse(stdout);
            const videoStream = info.streams.find(
              (s: any) => s.codec_type === "video"
            );

            const fps = eval(videoStream.r_frame_rate); // e.g., "30/1" -> 30
            const duration = parseFloat(info.format.duration);
            const totalFrames = Math.floor(fps * duration);

            resolve({ fps, totalFrames, duration });
          } catch (error) {
            reject(error);
          }
        } else {
          reject(new Error(`FFprobe failed with code ${code}`));
        }
      });
    });
  }

  /**
   * Generate unique plate ID
   */
  private generatePlateId(): string {
    return `plate_${Date.now()}_${(++this.plateCounter)
      .toString()
      .padStart(4, "0")}`;
  }

  /**
   * Process a single frame
   */
  private async processFrame(
    framePath: string,
    frameNumber: number,
    fps: number
  ): Promise<PlateDetection[]> {
    try {
      // Read frame
      const imageBuffer = fs.readFileSync(framePath);

      // Get original image dimensions using Sharp
      const metadata = await sharp(imageBuffer).metadata();
      const originalWidth = metadata.width || 1920;
      const originalHeight = metadata.height || 1080;

      // Preprocess image
      const preprocessed = await this.preprocessImage(imageBuffer);

      // Run inference
      const detections = await this.runInference(preprocessed);

      // Scale detections back to original image size
      const scaleX = originalWidth / this.inputWidth;
      const scaleY = originalHeight / this.inputHeight;

      // Convert to plate detections
      const plateDetections: PlateDetection[] = detections.map((detection) => ({
        unique_plate_id: this.generatePlateId(),
        frame_number: frameNumber,
        timestamp_seconds: Math.round((frameNumber / fps) * 1000) / 1000,
        license_plate_bbox: [
          Math.round(detection.x1 * scaleX * 100) / 100,
          Math.round(detection.y1 * scaleY * 100) / 100,
          Math.round(detection.x2 * scaleX * 100) / 100,
          Math.round(detection.y2 * scaleY * 100) / 100,
        ],
        detection_score: Math.round(detection.confidence * 10000) / 10000,
      }));

      return plateDetections;
    } catch (error) {
      console.error(`Error processing frame ${frameNumber}:`, error);
      return [];
    }
  }

  /**
   * Process entire video and return JSON results
   */
  async processVideo(
    videoPath: string,
    frameSkip: number = 3,
    outputDir?: string
  ): Promise<VideoProcessingResult> {
    const startTime = Date.now();

    console.log(`Processing video: ${videoPath}`);
    console.log(`Frame skip: ${frameSkip}`);

    try {
      // Extract frames
      console.log("Extracting frames...");
      const { frames, videoInfo } = await this.extractFrames(
        videoPath,
        frameSkip
      );

      console.log(`Extracted ${frames.length} frames`);

      // Process frames
      const allDetections: PlateDetection[] = [];

      for (let i = 0; i < frames.length; i++) {
        const frame = frames[i];
        console.log(`Processing frame ${i + 1}/${frames.length}`);

        const frameDetections = await this.processFrame(
          frame.imagePath,
          frame.frameNumber,
          videoInfo.fps
        );

        allDetections.push(...frameDetections);
      }

      // Cleanup temp frames
      this.cleanupTempFrames();

      const processingTime = (Date.now() - startTime) / 1000;

      const result: VideoProcessingResult = {
        video_path: videoPath,
        video_info: {
          total_frames: videoInfo.totalFrames,
          fps: videoInfo.fps,
          duration_seconds: videoInfo.duration,
          frame_skip: frameSkip,
        },
        detections: allDetections.sort(
          (a, b) => a.frame_number - b.frame_number
        ),
        processing_summary: {
          frames_processed: frames.length,
          total_plates_detected: allDetections.length,
          total_processing_time_seconds: Math.round(processingTime * 100) / 100,
        },
      };

      // Save results if output directory is provided
      if (outputDir) {
        if (!fs.existsSync(outputDir)) {
          fs.mkdirSync(outputDir, { recursive: true });
        }

        const baseName = path.basename(videoPath, path.extname(videoPath));
        const outputPath = path.join(outputDir, `${baseName}_detections.json`);

        fs.writeFileSync(outputPath, JSON.stringify(result, null, 2));
        console.log(`Results saved to: ${outputPath}`);
      }

      console.log(`\n=== Processing Complete ===`);
      console.log(`Total time: ${processingTime}s`);
      console.log(`Plates detected: ${allDetections.length}`);
      console.log(
        `Processing speed: ${(frames.length / processingTime).toFixed(
          2
        )} frames/second`
      );

      return result;
    } catch (error) {
      throw new Error(`Video processing failed: ${error}`);
    }
  }

  /**
   * Clean up temporary frame files
   */
  private cleanupTempFrames(): void {
    const tempDir = path.join(__dirname, "temp_frames");
    if (fs.existsSync(tempDir)) {
      fs.rmSync(tempDir, { recursive: true, force: true });
    }
  }

  /**
   * Dispose of resources
   */
  async dispose(): Promise<void> {
    if (this.session) {
      await this.session.release();
      this.session = null;
    }
    this.cleanupTempFrames();
  }
}

// Example usage
async function main() {
  const detector = new LicensePlateDetectorTS(
    "/Users/swaminathang/LicensePlateDetectorModel/model/best.onnx", // Update this path
    0.25 // confidence threshold
  );

  try {
    // Initialize the model
    await detector.initialize();

    // Process video
    const results = await detector.processVideo(
      "./videos/Dashcam1.mp4", // Update this path
      10, // frame skip
      "./output" // output directory
    );

    console.log("\nSample detection:");
    if (results.detections.length > 0) {
      console.log(JSON.stringify(results.detections[0], null, 2));
    }
  } catch (error) {
    console.error("Error:", error);
  } finally {
    await detector.dispose();
  }
}

// Run if this is the main module
if (require.main === module) {
  main().catch(console.error);
}
