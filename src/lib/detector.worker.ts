import * as ort from 'onnxruntime-web/webgpu';
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
      // Nhận ImageData và xử lý
      const detections = await detector.detectObjects(payload.imageData);
      // Gửi kết quả về Main Thread
      self.postMessage({ type: 'results', detections });
    } catch (err: any) {
      sendLog('worker', `Worker Detection Error: ${err.message}`);
    }
  }
};