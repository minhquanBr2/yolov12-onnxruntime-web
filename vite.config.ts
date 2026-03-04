import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  // Use base path from environment variable for GitHub Pages, or default to root
  // GitHub Actions will set BASE_PATH environment variable
  base: process.env.BASE_PATH || (process.env.NODE_ENV === 'production' ? '/yolov12-onnxruntime-web/' : '/'),
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
  },
  server: {
    host: '0.0.0.0', // Exposes to local network
    port: 20710,      // Your preferred port
    allowedHosts: ['if-wan4.selab.edu.vn', 'yolo-detection.selab.edu.vn'], // Thêm dòng này
    headers: {
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Resource-Policy': 'cross-origin',
    },
  },
});

