import * as ort from 'onnxruntime-web/webgpu';
import { Detection, ModelMetadata } from './types';

/**
 * Object detection using YOLOv8 - vehicle detection ONNX model via ONNX Runtime Web
 */
export class ObjectDetector {
  private session: ort.InferenceSession | null = null;
  private metadata: ModelMetadata | null = null;
  private isInitialized = false;
  private preprocessCanvas: HTMLCanvasElement | null = null;
  private preprocessCtx: CanvasRenderingContext2D | null = null;
  private tempCanvas: HTMLCanvasElement | null = null;
  private tempCtx: CanvasRenderingContext2D | null = null;

  /**
   * Initializes the ONNX model session and loads metadata
   */
  async initialize(): Promise<void> {
    try {
      // Configure ONNX Runtime Web to use CDN for WASM files FIRST
      // This ensures WASM files are always available, even with base path configurations
      // Must be set before any ONNX Runtime operations
      ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.0/dist/';
      ort.env.wasm.simd = true;
      ort.env.wasm.proxy = false;

      // Get base path for GitHub Pages compatibility
      const basePath = import.meta.env.BASE_URL || '/';
      
      // Load model metadata
      const metadataResponse = await fetch(`${basePath}models/model-metadata-vehicle-detection.json`);
      this.metadata = await metadataResponse.json();
      
      // Try different execution providers in order of preference.
      // On mobile browsers, prefer WASM for better stability.
      // Thay đổi thứ tự ưu tiên: Ưu tiên GPU để giảm tải cho CPU
      const executionProviders = [
        { name: 'webgpu', priority: 1 },
        { name: 'webgl', priority: 2 },
        { name: 'wasm', priority: 3 }
      ];

      let lastError: Error | null = null;
      
      for (const provider of executionProviders) {
        try {
          // 1. Log chính xác tên provider đang thử (ví dụ: 'webgpu')
          console.log(`Trying execution provider: ${provider.name}`);

          // Thêm cấu hình giảm tải bộ nhớ trong InferenceSession.create
          this.session = await ort.InferenceSession.create(`${basePath}models/yolo_vehicle_detection_model.onnx`, {
            executionProviders: [provider.name],
            graphOptimizationLevel: 'all',
            enableCpuMemArena: false, // Tắt arena để tiết kiệm RAM trên mobile
            enableMemPattern: false    // Tắt memory pattern nếu vẫn còn crash
          });

          console.log('Model input names:', this.session.inputNames);
          console.log('Model output names:', this.session.outputNames);
          console.log('Model input metadata:', this.session.inputMetadata);
          console.log('Model output metadata:', this.session.outputMetadata);

          // 2. Nếu thành công
          this.isInitialized = true;
          console.log(`Object detector initialized successfully with provider: ${provider.name}`);
          return; // Thoát hàm initialize

        } catch (error) {
          // 3. Fix log khi lỗi: dùng provider.name thay vì .join()
          console.warn(`Failed with provider ${provider.name}:`, error);
          lastError = error as Error;
          // Loop sẽ tự động 'continue' sang provider tiếp theo trong danh sách
        }
      }

      // If all providers failed, throw the last error
      throw lastError || new Error('All execution providers failed');
      
    } catch (error) {
      console.error('Failed to initialize object detector:', error);
      throw error;
    }
  }

  /**
   * Detects objects in an image frame
   * @param imageData - Image data from video frame
   * @returns Array of detected objects with bounding boxes and confidence scores
   */
  async detectObjects(imageData: ImageData): Promise<Detection[]> {
    if (!this.session || !this.metadata || !this.isInitialized) {
      throw new Error('Detector not initialized');
    }

    if (!this.session || !this.isInitialized) return [];

    let inputTensor: ort.Tensor | null = null;
    let results: ort.InferenceSession.ReturnType | null = null;

    try {
      // 1. Tiền xử lý
      inputTensor = this.preprocessImage(imageData);

      // 2. Chạy Inference
      const inputName = this.session.inputNames[0];
      results = await this.session.run({ [inputName]: inputTensor });

      // 3. Lấy kết quả đầu ra
      const outputName = this.session.outputNames[0];
      const outputTensor = results[outputName];

      // 4. Hậu xử lý (NMS, Scaling...)
      console.log('Output tensor shape:', outputTensor.dims);
      console.log('Output tensor', outputTensor);

      const detections = this.postprocessResults(outputTensor, imageData.width, imageData.height);
      console.log('Detected objects:', detections);
      console.log(
        `[Frame] Final detections (${detections.length}):`,
        detections.map((detection, index) => ({
          index,
          class: detection.class,
          confidence: Number(detection.confidence.toFixed(3)),
          box: {
            x: Math.round(detection.x),
            y: Math.round(detection.y),
            width: Math.round(detection.width),
            height: Math.round(detection.height)
          }
        }))
      );
      
      return detections;

    } catch (error) {
      console.error('Detection failed:', error);
      return [];
    } finally {
      // 5. GIẢI PHÓNG BỘ NHỚ - CỰC KỲ QUAN TRỌNG
      if (inputTensor) {
        inputTensor.dispose(); // Xóa tensor đầu vào
      }
      if (results) {
        // Duyệt qua tất cả output tensors để giải phóng
        for (const key in results) {
          results[key].dispose();
        }
      }
    }
  }

  /**
   * Preprocesses image for model input: resizes, pads, and normalizes to [0,1]
   */
  private preprocessImage(imageData: ImageData): ort.Tensor {
    const [inputWidth, inputHeight] = this.metadata!.inputSize;

    this.ensurePreprocessCanvases(inputWidth, inputHeight, imageData.width, imageData.height);

    const ctx = this.preprocessCtx!;
    const tempCanvas = this.tempCanvas!;
    const tempCtx = this.tempCtx!;
    
    // Fill canvas with black background (padding)
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, inputWidth, inputHeight);
    
    // Calculate scaling and positioning to maintain aspect ratio
    const aspectRatio = imageData.width / imageData.height;
    const targetAspectRatio = inputWidth / inputHeight;
    
    let drawWidth, drawHeight, offsetX, offsetY;
    
    if (aspectRatio > targetAspectRatio) {
      // Image is wider - fit to width, add padding top/bottom
      drawWidth = inputWidth;
      drawHeight = inputWidth / aspectRatio;
      offsetX = 0;
      offsetY = (inputHeight - drawHeight) / 2;
    } else {
      // Image is taller - fit to height, add padding left/right
      drawHeight = inputHeight;
      drawWidth = inputHeight * aspectRatio;
      offsetX = (inputWidth - drawWidth) / 2;
      offsetY = 0;
    }
    
    // Put the ImageData onto the temporary canvas
    tempCtx.putImageData(imageData, 0, 0);
    
    // Draw the image centered with padding, maintaining aspect ratio
    ctx.drawImage(tempCanvas, 0, 0, imageData.width, imageData.height, 
                  offsetX, offsetY, drawWidth, drawHeight);
    
    const paddedImageData = ctx.getImageData(0, 0, inputWidth, inputHeight);
    
    // Convert to tensor (normalize to 0-1)
    const data = new Float32Array(inputWidth * inputHeight * 3);
    for (let i = 0; i < paddedImageData.data.length; i += 4) {
      const pixelIndex = i / 4;
      data[pixelIndex] = paddedImageData.data[i] / 255;         // R
      data[pixelIndex + inputWidth * inputHeight] = paddedImageData.data[i + 1] / 255;     // G
      data[pixelIndex + 2 * inputWidth * inputHeight] = paddedImageData.data[i + 2] / 255; // B
    }
    
    return new ort.Tensor('float32', data, [1, 3, inputHeight, inputWidth]);
  }

  private ensurePreprocessCanvases(
    inputWidth: number,
    inputHeight: number,
    sourceWidth: number,
    sourceHeight: number
  ): void {
    if (!this.preprocessCanvas || !this.preprocessCtx) {
      this.preprocessCanvas = document.createElement('canvas');
      this.preprocessCtx = this.preprocessCanvas.getContext('2d');
    }

    if (!this.tempCanvas || !this.tempCtx) {
      this.tempCanvas = document.createElement('canvas');
      this.tempCtx = this.tempCanvas.getContext('2d');
    }

    if (!this.preprocessCtx || !this.tempCtx) {
      throw new Error('Canvas 2D context is not available');
    }

    if (this.preprocessCanvas!.width !== inputWidth || this.preprocessCanvas!.height !== inputHeight) {
      this.preprocessCanvas!.width = inputWidth;
      this.preprocessCanvas!.height = inputHeight;
    }

    if (this.tempCanvas!.width !== sourceWidth || this.tempCanvas!.height !== sourceHeight) {
      this.tempCanvas!.width = sourceWidth;
      this.tempCanvas!.height = sourceHeight;
    }
  }

  /**
   * Converts model output to Detection objects with transformed coordinates
   */
  private postprocessResults(output: ort.Tensor, originalWidth: number, originalHeight: number): Detection[] {
    const [inputWidth, inputHeight] = this.metadata!.inputSize;
    const outputData = output.data as Float32Array;
    const detections: Detection[] = [];

    const dims = output.dims;
    const classCount = this.metadata!.classes.length;
    const expectedAttributes = 4 + classCount;

    let numAttributes = 0;
    let numAnchors = 0;
    let getValue: (attributeIndex: number, anchorIndex: number) => number;

    // Support both common YOLO output layouts:
    // - [1, attributes, anchors]
    // - [1, anchors, attributes]
    if (dims.length === 3 && dims[1] === expectedAttributes) {
      numAttributes = dims[1];
      numAnchors = dims[2];
      getValue = (attributeIndex: number, anchorIndex: number) => {
        return outputData[attributeIndex * numAnchors + anchorIndex];
      };
    } else if (dims.length === 3 && dims[2] === expectedAttributes) {
      numAnchors = dims[1];
      numAttributes = dims[2];
      getValue = (attributeIndex: number, anchorIndex: number) => {
        return outputData[anchorIndex * numAttributes + attributeIndex];
      };
    } else {
      console.warn('Unexpected output tensor shape:', dims);
      return [];
    }

    // Tính toán tỷ lệ scale và padding (giống phần trước)
    const scale = Math.min(inputWidth / originalWidth, inputHeight / originalHeight);
    const offsetX = (inputWidth - originalWidth * scale) / 2;
    const offsetY = (inputHeight - originalHeight * scale) / 2;

    for (let i = 0; i < numAnchors; i++) {
        // 1. Tìm Class có điểm cao nhất trong 13 classes (từ index 4 đến 16)
        let maxScore = -Infinity;
        let classId = -1;

        for (let j = 4; j < numAttributes; j++) {
            const score = getValue(j, i);
            if (score > maxScore) {
                maxScore = score;
                classId = j - 4;
            }
        }

        // 2. Lọc theo ngưỡng Confidence
        if (maxScore < this.metadata!.confidenceThreshold) continue;

        // 3. Giải mã tọa độ [cx, cy, w, h] từ dữ liệu Transposed
        const cx = getValue(0, i);
        const cy = getValue(1, i);
        const w = getValue(2, i);
        const h = getValue(3, i);
        console.log(`Anchor ${i}: cx=${cx}, cy=${cy}, w=${w}, h=${h}`);

        // Chuyển từ Center (cx, cy) sang Top-Left (x1, y1)
        const x1 = cx - w / 2;
        const y1 = cy - h / 2;

        // 4. Map ngược về tọa độ ảnh gốc
        const realX = (x1 - offsetX) / scale;
        const realY = (y1 - offsetY) / scale;
        const realW = w / scale;
        const realH = h / scale;

        const clampedX = Math.max(0, Math.min(realX, originalWidth));
        const clampedY = Math.max(0, Math.min(realY, originalHeight));
        const clampedMaxX = Math.max(0, Math.min(realX + realW, originalWidth));
        const clampedMaxY = Math.max(0, Math.min(realY + realH, originalHeight));
        const clampedWidth = clampedMaxX - clampedX;
        const clampedHeight = clampedMaxY - clampedY;

        if (clampedWidth <= 1 || clampedHeight <= 1) continue;

        detections.push({
          x: clampedX,
          y: clampedY,
          width: clampedWidth,
          height: clampedHeight,
          confidence: maxScore,
          class: this.metadata!.classes[classId] || `class_${classId}`
        });
    }

    // 5. NMS (Rất quan trọng vì 8400 anchors sẽ tạo ra rất nhiều box trùng nhau)
    // Tuy nhiên trước khi NMS, cần lọc bớt các box có confidence thấp
    const filtered_detections = detections.filter(d => d.confidence > this.metadata!.confidenceThreshold);
    return this.applyNMS(filtered_detections);
}

  /**
   * Applies Non-Maximum Suppression to remove overlapping detections
   */
  private applyNMS(detections: Detection[]): Detection[] {
    const detectionsByClass = new Map<string, Detection[]>();
    for (const detection of detections) {
      const existing = detectionsByClass.get(detection.class);
      if (existing) {
        existing.push(detection);
      } else {
        detectionsByClass.set(detection.class, [detection]);
      }
    }

    const filtered: Detection[] = [];

    for (const classDetections of detectionsByClass.values()) {
      classDetections.sort((a, b) => b.confidence - a.confidence);

      const used = new Set<number>();

      for (let i = 0; i < classDetections.length; i++) {
        if (used.has(i)) continue;

        const detection = classDetections[i];
        filtered.push(detection);
        used.add(i);

        for (let j = i + 1; j < classDetections.length; j++) {
          if (used.has(j)) continue;

          const other = classDetections[j];
          const iou = this.calculateIoU(detection, other);

          if (iou > this.metadata!.nmsThreshold) {
            used.add(j);
          }
        }
      }
    }

    return filtered.sort((a, b) => b.confidence - a.confidence);
  }

  /**
   * Calculates Intersection over Union (IoU) between two detections
   */
  private calculateIoU(det1: Detection, det2: Detection): number {
    const x1 = Math.max(det1.x, det2.x);
    const y1 = Math.max(det1.y, det2.y);
    const x2 = Math.min(det1.x + det1.width, det2.x + det2.width);
    const y2 = Math.min(det1.y + det1.height, det2.y + det2.height);
    
    if (x2 <= x1 || y2 <= y1) return 0;
    
    const intersection = (x2 - x1) * (y2 - y1);
    const area1 = det1.width * det1.height;
    const area2 = det2.width * det2.height;
    const union = area1 + area2 - intersection;
    
    return intersection / union;
  }

  isReady(): boolean {
    return this.isInitialized;
  }

  dispose(): void {
    if (this.session) {
      this.session.release();
      this.session = null;
    }
    this.isInitialized = false;
  }
}
