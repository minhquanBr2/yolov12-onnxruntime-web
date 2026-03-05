import { Detection, DetectionStats } from './types';

export class VideoProcessor {
  private video: HTMLVideoElement | null = null;
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private rotateClockwise90 = false;
  
  // Stats - Chuyển sang lưu trữ giá trị gộp thay vì mảng khổng lồ
  private totalCount = 0;
  private confidenceSum = 0;
  private stats: DetectionStats = {
    totalDetections: 0,
    averageConfidence: 0,
    lastDetectionTime: 0,
    classCounts: {}
  };

  constructor(
    private detectionCallback: (detections: Detection[]) => void,
    private statsCallback: (stats: DetectionStats) => void
  ) {
    this.canvas = document.createElement('canvas');
    this.ctx = this.canvas.getContext('2d', { alpha: false })!; // Tắt alpha để render nhanh hơn
  }

  setVideo(video: HTMLVideoElement): void {
    this.video = video;
  }

  setRotateClockwise90(rotate: boolean): void {
    this.rotateClockwise90 = rotate;
  }

  /**
   * Lấy frame hiện tại - Tối ưu hóa để App.tsx gọi trực tiếp
   */
  getCurrentFrame(): ImageData | null {
    if (!this.video || this.video.videoWidth === 0 || this.video.videoHeight === 0) return null;

    this.updateCanvasSize();
    this.drawCurrentFrame();
    
    // ImageData này sẽ được "Transfer" sang Worker, nên ko lo leak RAM ở đây
    return this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
  }

  /**
   * Cập nhật kết quả mà không gây tràn RAM
   */
  updateDetections(newDetections: Detection[]): void {
    if (newDetections.length > 0) {
      this.updateStats(newDetections);
    }
    
    // Chỉ gửi kết quả frame hiện tại về UI
    this.detectionCallback(newDetections);
    this.statsCallback(this.stats);
  }

  private updateStats(newDetections: Detection[]): void {
    this.stats.totalDetections += newDetections.length;
    this.stats.lastDetectionTime = Date.now();

    newDetections.forEach(d => {
      // 1. Cập nhật tổng confidence để tính trung bình
      this.totalCount++;
      this.confidenceSum += d.confidence;

      // 2. Cập nhật đếm theo Class
      if (!this.stats.classCounts[d.class]) {
        this.stats.classCounts[d.class] = 0;
      }
      this.stats.classCounts[d.class]++;
    });

    // Tính trung bình bằng công thức toán học thay vì duyệt mảng
    this.stats.averageConfidence = this.confidenceSum / this.totalCount;
  }

  private updateCanvasSize(): void {
    if (!this.video) return;
    const targetWidth = this.rotateClockwise90 ? this.video.videoHeight : this.video.videoWidth;
    const targetHeight = this.rotateClockwise90 ? this.video.videoWidth : this.video.videoHeight;

    if (this.canvas.width !== targetWidth || this.canvas.height !== targetHeight) {
      this.canvas.width = targetWidth;
      this.canvas.height = targetHeight;
    }
  }

  private drawCurrentFrame(): void {
    if (!this.video) return;

    if (!this.rotateClockwise90) {
      this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
    } else {
      this.ctx.save();
      this.ctx.translate(this.canvas.width, 0);
      this.ctx.rotate(Math.PI / 2);
      this.ctx.drawImage(this.video, 0, 0, this.video.videoWidth, this.video.videoHeight);
      this.ctx.restore();
    }
  }

  reset(): void {
    this.totalCount = 0;
    this.confidenceSum = 0;
    this.stats = {
      totalDetections: 0,
      averageConfidence: 0,
      lastDetectionTime: 0,
      classCounts: {}
    };
    this.statsCallback(this.stats);
  }
}