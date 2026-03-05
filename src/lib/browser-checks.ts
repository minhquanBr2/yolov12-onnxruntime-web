/**
 * Browser capability checks for ONNX Runtime Web and Web Worker requirements
 */

export interface BrowserCheckResult {
  passed: boolean;
  message: string;
}

// Kiểm tra xem chúng ta đang ở Worker hay Main Thread
const isWorker = typeof window === 'undefined';
const globalScope = isWorker ? self : window;

/**
 * Checks if WebGPU is supported
 */
export function checkWebGPU(): BrowserCheckResult {
  try {
    const gpu = (navigator as any).gpu;
    if (!gpu) {
      return {
        passed: false,
        message: 'WebGPU is not supported. Required for best performance on S25 Ultra.'
      };
    }
    return { passed: true, message: 'WebGPU is supported' };
  } catch (e) {
    return { passed: false, message: 'WebGPU check failed' };
  }
}

/**
 * Checks if WebGL is supported
 */
export function checkWebGL(): BrowserCheckResult {
  try {
    // SỬA LỖI: Tạo canvas 1x1 để test, không dùng biến naturalWidth chưa định nghĩa
    const canvas = typeof OffscreenCanvas !== 'undefined' 
      ? new OffscreenCanvas(1, 1) 
      : document.createElement('canvas');
      
    const gl = canvas.getContext('webgl');
    if (!gl) {
      return { passed: false, message: 'WebGL not supported.' };
    }
    return { passed: true, message: 'WebGL is supported' };
  } catch (e) {
    return { passed: false, message: 'WebGL check failed' };
  }
}

/**
 * Checks if OffscreenCanvas is available (Required for Workers)
 */
export function checkOffscreenCanvas(): BrowserCheckResult {
  if (typeof OffscreenCanvas === 'undefined') {
    return {
      passed: false,
      message: 'OffscreenCanvas is not supported. Worker-based detection will fail.'
    };
  }
  return { passed: true, message: 'OffscreenCanvas is supported' };
}

/**
 * Checks if MediaStream API is supported (Main Thread Only)
 */
export function checkMediaStream(): BrowserCheckResult {
  if (isWorker) return { passed: true, message: 'MediaStream check skipped in Worker' };
  
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    return {
      passed: false,
      message: 'Camera access not supported. Use HTTPS.'
    };
  }
  return { passed: true, message: 'MediaStream API is supported' };
}

/**
 * Checks if SharedArrayBuffer is available (Required for multi-threading ONNX)
 */
export function checkSharedArrayBuffer(): BrowserCheckResult {
  if (typeof SharedArrayBuffer === 'undefined') {
    return {
      passed: false,
      message: 'SharedArrayBuffer not available. Missing COOP/COEP headers.'
    };
  }
  return { passed: true, message: 'SharedArrayBuffer is available' };
}

/**
 * Runs all browser compatibility checks
 */
export function checkBrowserCompatibility() {
  const checks = [
    { name: 'OffscreenCanvas', check: checkOffscreenCanvas },
    { name: 'WebGPU', check: checkWebGPU },
    { name: 'WebGL', check: checkWebGL },
    { name: 'MediaStream', check: checkMediaStream },
    { name: 'SharedArrayBuffer', check: checkSharedArrayBuffer }
  ];
  
  const results: BrowserCheckResult[] = [];
  const errors: string[] = [];
  
  checks.forEach(({ name, check }) => {
    const result = check();
    results.push(result);
    if (!result.passed && name !== 'MediaStream') { // MediaStream không bắt buộc trong Worker
      errors.push(`${name}: ${result.message}`);
    }
  });
  
  return {
    allPassed: errors.length === 0,
    results,
    errors
  };
}