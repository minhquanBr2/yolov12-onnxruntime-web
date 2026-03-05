import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

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
                const timestamp = new Date(time).toLocaleTimeString();
                
                // ANSI Colors for terminal output
                const colors = { info: '\x1b[32m', warn: '\x1b[33m', error: '\x1b[31m', worker: '\x1b[36m', reset: '\x1b[0m' };
                const color = colors[type as keyof typeof colors] || colors.info;

                console.log(`${timestamp} [${color}${type.toUpperCase()}${colors.reset}]:`, 
                  typeof message === 'object' ? JSON.stringify(message, null, 2) : message
                );

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

