import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60)
  const secs = Math.floor(seconds % 60)
  return `${mins}:${secs.toString().padStart(2, '0')}`
}

export function formatConfidence(confidence: number): string {
  return `${(confidence * 100).toFixed(1)}%`
}

export type LogType = 'info' | 'warn' | 'error' | 'worker';

/**
 * Sends a log message to the Vite dev server middleware at /log
 */
export const sendLog = async (type: LogType, message: any) => {
  try {
    // Using a relative path so it works on both localhost and mobile IP
    await fetch('https://yolo-detection.selab.edu.vn/log', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        type,
        message,
        time: Date.now(),
      }),
    });
  } catch (err) {
    // Fallback to local console if the middleware isn't responding
    console.debug('Remote logger unavailable, falling back to console.');
  }
};