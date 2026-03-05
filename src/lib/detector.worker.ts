import { ObjectDetector } from './object-detector';
import { sendLog } from './utils';

let detector: ObjectDetector | null = null;

self.onmessage = async (e) => {
  const { type, payload } = e.data;

  if (type === 'init') {
    try {
      sendLog('worker', 'Worker: Starting initialization...');
      detector = new ObjectDetector();
      await detector.initialize();
      self.postMessage({ type: 'ready' });
      sendLog('worker', 'Worker: Detector initialized successfully');
    } catch (err: any) {
      sendLog('worker', `Worker Init Error: ${err.message}`);
      self.postMessage({ type: 'error', error: err.message });
    }
  }

  if (type === 'detect' && detector) {
    try {
      const width = payload?.width as number | undefined;
      const height = payload?.height as number | undefined;
      const rawData = payload?.data as Uint8ClampedArray | undefined;

      if (!width || !height || !rawData) {
        throw new Error('Invalid detect payload: missing width/height/data');
      }

      const pixelData = new Uint8ClampedArray(rawData);
      const imageData = new ImageData(pixelData, width, height);

      const pixels = imageData.data;
      const sampleLength = Math.min(pixels.length, 4000);
      let nonZeroRgb = 0;
      let sum = 0;

      for (let i = 0; i < sampleLength; i += 4) {
        const r = pixels[i];
        const g = pixels[i + 1];
        const b = pixels[i + 2];
        sum += r + g + b;
        if (r !== 0 || g !== 0 || b !== 0) {
          nonZeroRgb++;
        }
      }

      const sampledPixels = Math.max(1, Math.floor(sampleLength / 4));
      const meanRgb = sum / (sampledPixels * 3);
      sendLog(
        'worker',
        `Worker frame stats: ${imageData.width}x${imageData.height}, sampled=${sampledPixels}, nonZeroRatio=${(nonZeroRgb / sampledPixels).toFixed(4)}, meanRgb=${meanRgb.toFixed(2)}`
      );

      const detections = await detector.detectObjects(imageData);
      sendLog('worker', `Worker: Detection completed with number of objects: ${detections.length}`);
      // Gửi kết quả về Main Thread
      self.postMessage({ type: 'results', detections });
    } catch (err: any) {
      sendLog('worker', `Worker Detection Error: ${err.message}`);
    }
  }
};