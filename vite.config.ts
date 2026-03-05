import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import fs from 'fs'; // Import the file system module

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    react(),
    {
      name: 'remote-logger-middleware',
      configureServer(server) {
        server.middlewares.use('/log', (req, res, next) => {
          if (req.method === 'POST') {
            let body = '';
            req.on('data', chunk => { body += chunk; });
            req.on('end', () => {
              try {
                const { type, message, time } = JSON.parse(body);
                const date = new Date(time);
                const timestamp = date.toLocaleTimeString();
                const logDate = date.toISOString().split('T')[0]; // YYYY-MM-DD
                
                // 1. Prepare the log entry (text only, no ANSI colors for files)
                const formattedMessage = typeof message === 'object' 
                  ? JSON.stringify(message, null, 2) 
                  : message;
                const logEntry = `[${logDate} ${timestamp}] [${type.toUpperCase()}]: ${formattedMessage}\n`;

                // 2. Write to file (appends to the end)
                // The file will be created in your project root
                fs.appendFile(
                  path.resolve(__dirname, 'mobile-debug.log'), 
                  logEntry, 
                  (err) => {
                    if (err) console.error('Failed to write to log file:', err);
                  }
                );

                // 3. Optional: Keep printing to terminal so you see it live
                const colors = { info: '\x1b[32m', warn: '\x1b[33m', error: '\x1b[31m', worker: '\x1b[36m', reset: '\x1b[0m' };
                const color = colors[type as keyof typeof colors] || colors.info;
                console.log(`${timestamp} [${color}${type.toUpperCase()}${colors.reset}]:`, formattedMessage);

                res.statusCode = 200;
                res.end('OK');
              } catch (e) {
                res.statusCode = 400;
                res.end('Invalid JSON');
              }
            });
          } else {
            next();
          }
        });
      }
    }
  ],
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
    host: '0.0.0.0', // Exposes to external network
    port: 20710,      // Your preferred port
    allowedHosts: ['yolo-detection.selab.edu.vn']
  }
});

