import { useState, useRef, useEffect, useCallback } from 'react';
import { FileUpload } from '@/components/file-upload';
import { DetectionOverlay } from '@/components/detection-overlay';
import { ThemeToggle } from '@/components/theme-toggle';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { VideoProcessor } from '@/lib/video-processor';
import { Detection } from '@/lib/types';
import { Play, Square, Info, Camera, CameraOff } from 'lucide-react';
import './globals.css';
import { sendLog } from './lib/utils';

const DETECTION_FPS = 5;
const DETECTION_INTERVAL_MS = 1000 / DETECTION_FPS;

function App() {
  // --- States ---
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isDeviceLandscape] = useState(window.matchMedia('(orientation: landscape)').matches);
  const [isCameraSourcePortrait, setIsCameraSourcePortrait] = useState(false);
  const [cameraSourceAspectRatio, setCameraSourceAspectRatio] = useState(16 / 9);
  const [isDetectorReady, setIsDetectorReady] = useState(false);
  const [videoDimensions, setVideoDimensions] = useState({ width: 640, height: 480 });
  const [imageDimensions, setImageDimensions] = useState({ width: 640, height: 480 });
  const [isInfoDialogOpen, setIsInfoDialogOpen] = useState(false);
  const [inputType, setInputType] = useState<'upload' | 'camera'>('upload');

  // --- Refs (Điều khiển logic ngầm) ---
  const videoRef = useRef<HTMLVideoElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const processorRef = useRef<VideoProcessor | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const workerRef = useRef<Worker | null>(null);
  
  const isProcessingRef = useRef(false);
  const isDetectionInFlightRef = useRef(false);
  const lastDetectionTimeRef = useRef(0);

  // --- Helpers ---
  const shouldRotateCameraToLandscape = isCameraActive && isDeviceLandscape && isCameraSourcePortrait;
  const cameraPreviewTransform = shouldRotateCameraToLandscape
    ? `rotate(90deg) scale(${cameraSourceAspectRatio})`
    : 'none';

  const stopProcessing = useCallback(() => {
    setIsProcessing(false);
    isProcessingRef.current = false;
    isDetectionInFlightRef.current = false;
    if (processorRef.current) processorRef.current.reset();
    setDetections([]);
    sendLog('info', 'Detection system stopped and reset');
  }, []);

  const updateCameraDisplayDimensions = useCallback((sourceWidth: number, sourceHeight: number) => {
    const shouldRotate = isDeviceLandscape && sourceHeight > sourceWidth;
    if (shouldRotate) {
      setVideoDimensions({ width: sourceHeight, height: sourceWidth });
      return;
    }
    setVideoDimensions({ width: sourceWidth, height: sourceHeight });
  }, [isDeviceLandscape]);

  // --- 1. Khởi tạo Worker ---
  useEffect(() => {
    sendLog('info', 'Spawning AI Worker...');
    const worker = new Worker(new URL('./lib/detector.worker.ts', import.meta.url), { type: 'module' });

    worker.onmessage = (e) => {
      const { type, detections: results, error } = e.data;
      if (type === 'ready') { setIsDetectorReady(true); sendLog('info', 'Worker is ready'); }
      if (type === 'results') {
        setDetections(results);
        if (processorRef.current) processorRef.current.updateDetections(results);
        isDetectionInFlightRef.current = false; // Mở khóa cho frame tiếp theo
        if (inputType === 'upload' && selectedImage) setIsProcessing(false);
      }
      if (type === 'error') { sendLog('error', `Worker error: ${error}`); stopProcessing(); }
    };

    worker.postMessage({ type: 'init' });
    workerRef.current = worker;
    return () => worker.terminate();
  }, [inputType, selectedImage, stopProcessing]);

  // --- 2. Xử lý Camera Stream ---
  const handleCameraStart = useCallback((stream: MediaStream) => {
    sendLog('info', 'Camera stream connected');
    streamRef.current = stream;
    setIsCameraActive(true);
    
    setTimeout(() => {
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play().catch(e => sendLog('error', `Play error: ${e.message}`));
      }
    }, 150);
  }, []);

  const handleCameraStop = useCallback(() => {
    stopProcessing();
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    setIsCameraActive(false);
    if (videoRef.current) videoRef.current.srcObject = null;
  }, [stopProcessing]);

  // --- 3. Xử lý File Upload (Video/Image) ---
  const handleVideoSelect = useCallback((file: File) => {
    stopProcessing();
    setSelectedFile(file);
    setSelectedImage(null);
    setIsCameraActive(false);
  }, [stopProcessing]);

  const handleImageSelect = useCallback((file: File) => {
    stopProcessing();
    setSelectedImage(file);
    setSelectedFile(null);
    setIsCameraActive(false);
  }, [stopProcessing]);

  // --- DỌN DẸP VIDEO ---
  const handleClearVideo = useCallback(() => {
    // 1. Dừng AI ngay lập tức
    stopProcessing();

    // 2. Thu hồi bộ nhớ Object URL (Quan trọng để tránh leak RAM)
    if (videoRef.current && videoRef.current.src.startsWith('blob:')) {
      URL.revokeObjectURL(videoRef.current.src);
    }
    
    // 3. Reset States & Refs
    setSelectedFile(null);
    setDetections([]);
    
    if (videoRef.current) {
      videoRef.current.src = '';
      videoRef.current.srcObject = null;
      videoRef.current.load(); // Buộc video reset trạng thái buffer
    }
    
    sendLog('info', 'Video cleared and memory released');
  }, [stopProcessing]);

  // --- DỌN DẸP IMAGE ---
  const handleClearImage = useCallback(() => {
    // 1. Dừng AI
    stopProcessing();

    // 2. Thu hồi bộ nhớ
    if (imageRef.current && imageRef.current.src.startsWith('blob:')) {
      URL.revokeObjectURL(imageRef.current.src);
    }
    
    // 3. Reset States
    setSelectedImage(null);
    setDetections([]);
    setImageDimensions({ width: 0, height: 0 });

    // 4. Reset Image Element về ảnh trống (1x1 pixel) để tránh hiện icon "ảnh lỗi"
    if (imageRef.current) {
      imageRef.current.src = 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7';
    }
    
    sendLog('info', 'Image cleared and memory released');
  }, [stopProcessing]);

  // --- 4. Logic Detection Loop ---
  const startProcessing = useCallback(() => {
    if (!videoRef.current || !isDetectorReady || !workerRef.current) return;

    setIsProcessing(true);
    isProcessingRef.current = true;
    
    processorRef.current = new VideoProcessor((res) => setDetections(res), () => {});
    processorRef.current.setVideo(videoRef.current);
    processorRef.current.setRotateClockwise90(shouldRotateCameraToLandscape);

    const detectLoop = (timestamp: number) => {
      if (!isProcessingRef.current || !workerRef.current || !processorRef.current) return;

      const elapsed = timestamp - lastDetectionTimeRef.current;
      if (!isDetectionInFlightRef.current && elapsed >= DETECTION_INTERVAL_MS) {
        const frame = processorRef.current.getCurrentFrame();
        if (frame) {
          isDetectionInFlightRef.current = true;
          lastDetectionTimeRef.current = timestamp;
          workerRef.current.postMessage(
            {
              type: 'detect',
              payload: {
                width: frame.width,
                height: frame.height,
                data: frame.data
              }
            },
            [frame.data.buffer]
          );
        }
      }
      requestAnimationFrame(detectLoop);
    };
    requestAnimationFrame(detectLoop);
  }, [isDetectorReady, shouldRotateCameraToLandscape]);

  // --- 5. Effect xử lý Image Detection tự động ---
  useEffect(() => {
    if (selectedImage && imageRef.current && isDetectorReady) {
      const url = URL.createObjectURL(selectedImage);
      imageRef.current.src = url;
      imageRef.current.onload = () => {
        const { naturalWidth, naturalHeight } = imageRef.current!;
        setImageDimensions({ width: naturalWidth, height: naturalHeight });
        setIsProcessing(true);
        const canvas = new OffscreenCanvas(naturalWidth, naturalHeight);
        const ctx = canvas.getContext('2d')!;
        ctx.drawImage(imageRef.current!, 0, 0);
        const imageData = ctx.getImageData(0, 0, naturalWidth, naturalHeight);
        workerRef.current?.postMessage(
          {
            type: 'detect',
            payload: {
              width: imageData.width,
              height: imageData.height,
              data: imageData.data
            }
          },
          [imageData.data.buffer]
        );
      };
      return () => URL.revokeObjectURL(url);
    }
  }, [selectedImage, isDetectorReady]);

  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <div className="w-full max-w-6xl">
        {/* Header */}
        <div className="relative text-center mb-8">
          {/* Info Button and Theme Toggle */}
          <div className="absolute top-0 right-0 flex gap-2 sm:right-4">
            <ThemeToggle />
            <Dialog open={isInfoDialogOpen} onOpenChange={setIsInfoDialogOpen}>
              <DialogTrigger asChild>
                  <Button
                  variant="ghost"
                  size="icon"
                  aria-label="View model information"
                >
                  <Info className="h-5 w-5 text-foreground" />
                </Button>
              </DialogTrigger>
              <DialogContent className="max-w-2xl">
                <DialogHeader>
                  <DialogTitle className="text-2xl font-bold text-foreground mb-2">
                    Model & Technology Information
                  </DialogTitle>
                  <DialogDescription className="text-base text-muted-foreground">
                    Core information about the AI model and tech stack
                  </DialogDescription>
                </DialogHeader>
                <div className="space-y-6 py-4">
                  {/* Model Information */}
                  <div className="space-y-3">
                    <h3 className="text-lg font-bold text-foreground border-b border-border pb-2">
                      AI Model
                    </h3>
                    <div className="space-y-2 pl-4">
                      <div className="flex items-start">
                        <span className="font-semibold text-foreground min-w-[120px]">Model:</span>
                        <span className="text-muted-foreground">yolov12n.onnx</span>
                      </div>
                      <div className="flex items-start">
                        <span className="font-semibold text-foreground min-w-[120px]">Size:</span>
                        <span className="text-muted-foreground">11.9 MB</span>
                      </div>
                      <div className="flex items-start">
                        <span className="font-semibold text-foreground min-w-[120px]">Runtime:</span>
                        <span className="text-muted-foreground">ONNX Runtime through onnxruntime-web package</span>
                      </div>
                      <div className="flex items-start">
                        <span className="font-semibold text-foreground min-w-[120px]">Repository:</span>
                        <a 
                          href="https://github.com/sunsmarterjie/yolov12" 
                          target="_blank" 
                          rel="noopener noreferrer"
                          className="text-muted-foreground hover:text-foreground underline"
                        >
                          https://github.com/sunsmarterjie/yolov12
                        </a>
                      </div>
                    </div>
                  </div>

                  {/* Tech Stack */}
                  <div className="space-y-3">
                    <h3 className="text-lg font-bold text-foreground border-b border-border pb-2">
                      Technology Stack
                    </h3>
                    <div className="space-y-2 pl-4">
                      <div className="flex items-start">
                        <span className="font-semibold text-foreground min-w-[140px]">Framework:</span>
                        <span className="text-muted-foreground">React 19 with Vite</span>
                      </div>
                      <div className="flex items-start">
                        <span className="font-semibold text-foreground min-w-[140px]">Runtime:</span>
                        <span className="text-muted-foreground">Browser (Client-side)</span>
                      </div>
                      <div className="flex items-start">
                        <span className="font-semibold text-foreground min-w-[140px]">UI Library:</span>
                        <span className="text-muted-foreground">React 19</span>
                      </div>
                      <div className="flex items-start">
                        <span className="font-semibold text-foreground min-w-[140px]">Language:</span>
                        <span className="text-muted-foreground">TypeScript 5</span>
                      </div>
                    </div>
                  </div>
                </div>
              </DialogContent>
            </Dialog>
          </div>

          <h1 className="text-4xl font-bold text-foreground mb-4">
            YOLOv12 Object Detection
          </h1>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Real-time object detection powered by YOLOv12 and ONNX Runtime Web
          </p>
        </div>

        {/* Tabs at the Top */}
        <div className="flex justify-center">
          <Tabs value={inputType} onValueChange={(v) => setInputType(v as 'upload' | 'camera')} className="w-auto">
            <TabsList className="inline-flex w-auto p-1 rounded-lg">
              <TabsTrigger 
                value="upload" 
                className="font-bold text-foreground"
              >
                Video/Image Upload
              </TabsTrigger>
              <TabsTrigger 
                value="camera"
                className="font-bold text-foreground"
              >
                Live Camera
              </TabsTrigger>
            </TabsList>
          </Tabs>
        </div>

        {/* Main Content Area */}
        <Card>
          <CardContent className="p-6">
            <div className="w-full">
              {/* Upload Section - Unified Video and Image */}
              {inputType === 'upload' && !selectedFile && !selectedImage && (
                <FileUpload
                  onVideoSelect={handleVideoSelect}
                  onImageSelect={handleImageSelect}
                  onClear={() => {
                    if (selectedFile) {
                      handleClearVideo();
                    }
                    if (selectedImage) {
                      handleClearImage();
                    }
                  }}
                  selectedVideo={selectedFile}
                  selectedImage={selectedImage}
                />
              )}

              {inputType === 'camera' && (
                <div className="space-y-6">
                  {/* Video Preview - Always visible */}
                  <div className="relative bg-card rounded overflow-hidden min-h-[400px] max-h-[600px] flex items-center justify-center">
                    {!isCameraActive ? (
                      <div className="text-center">
                        <Camera className="h-16 w-16 mx-auto mb-4 opacity-50" />
                        <p className="text-lg">Camera Not Started</p>
                        <p className="text-sm mt-2">Click &quot;Start Camera&quot; below to begin</p>
                      </div>
                    ) : (
                      <>
                        <div className="relative w-full h-[60vh] min-h-[400px] max-h-[600px] overflow-hidden">
                          <div
                            className="absolute inset-0"
                            style={{
                              transform: cameraPreviewTransform,
                              transformOrigin: 'center center'
                            }}
                          >
                            <video
                              ref={videoRef}
                              className="w-full h-full object-contain"
                              muted
                              playsInline
                              autoPlay
                              preload="none"
                              onLoadedMetadata={() => {
                                if (videoRef.current) {
                                  const { videoWidth, videoHeight } = videoRef.current;
                                  setIsCameraSourcePortrait(videoHeight > videoWidth);
                                  setCameraSourceAspectRatio(videoHeight / videoWidth);
                                  updateCameraDisplayDimensions(videoWidth, videoHeight);
                                }
                              }}
                            />
                            <DetectionOverlay
                              detections={detections}
                              videoWidth={videoDimensions.width}
                              videoHeight={videoDimensions.height}
                              className="absolute inset-0"
                            />
                          </div>
                        </div>
                      </>
                    )}
                  </div>

                  {/* Camera and Detection Controls */}
                  <div className="flex justify-center space-x-3">
                    {/* Camera Control */}
                    {!isCameraActive ? (
                      <Button
                        onClick={async () => {
                          try {
                            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                              console.error('Camera access not supported in this browser');
                              return;
                            }
                            
                            let stream;
                            try {
                              stream = await navigator.mediaDevices.getUserMedia({
                                video: {
                                  width: { ideal: 1280 },
                                  height: { ideal: 720 },
                                  aspectRatio: { ideal: 16 / 9 },
                                  frameRate: { ideal: 30 },
                                  facingMode: { ideal: 'environment' }
                                },
                                audio: false
                              });
                            } catch {
                              stream = await navigator.mediaDevices.getUserMedia({
                                video: true,
                                audio: false
                              });
                            }
                            
                            handleCameraStart(stream);
                          } catch (err) {
                            const errorMessage = err instanceof Error ? err.message : 'Failed to access camera';
                            console.error('Camera access error:', errorMessage, err);
                          }
                        }}
                        variant="outline"
                        className="px-6 py-2"
                      >
                        <Camera className="h-4 w-4 mr-2" />
                        Start Camera
                      </Button>
                    ) : (
                      <Button
                        onClick={handleCameraStop}
                        variant="outline"
                        className="px-6 py-2"
                      >
                        <CameraOff className="h-4 w-4 mr-2" />
                        Stop Camera
                      </Button>
                    )}

                    {/* Detection Control */}
                    {!isProcessing ? (
                      <Button
                        onClick={startProcessing}
                        disabled={!isCameraActive}
                        variant="outline"
                        className="px-6 py-2"
                      >
                        <Play className="h-4 w-4 mr-2" />
                        Start Detection
                      </Button>
                    ) : (
                      <Button
                        onClick={stopProcessing}
                        variant="outline"
                        className="px-6 py-2"
                      >
                        <Square className="h-4 w-4 mr-2" />
                        Stop Detection
                      </Button>
                    )}
                  </div>
                </div>
              )}

                {/* Video Preview Section - Only show for uploaded videos, not camera */}
                {selectedFile && !isCameraActive && (
                  <div>
                    <div className="relative bg-background rounded overflow-hidden">
                      <video
                        ref={videoRef}
                        className="w-full h-auto min-h-[300px] max-h-[600px] object-contain"
                        controls={!isCameraActive}
                        muted
                        playsInline
                        autoPlay={isCameraActive}
                        preload={isCameraActive ? "none" : "metadata"}
                        onLoadedMetadata={() => {
                          if (videoRef.current) {
                            const { videoWidth, videoHeight } = videoRef.current;
                            sendLog('info', `Video dimensions: ${videoWidth}x${videoHeight}`);
                            setVideoDimensions({ width: videoWidth, height: videoHeight });
                          }
                        }}
                        onCanPlay={() => {
                          sendLog('info', 'Video can play');
                        }}
                        onPlaying={() => {
                          sendLog('info', 'Video is playing');
                        }}
                        onError={() => {
                          sendLog('error', 'Video error:');
                          const error = videoRef.current?.error;
                          if (error) {
                            sendLog('error', 'Video error details: ' + JSON.stringify({
                              code: error.code,
                              message: error.message
                            }));
                          }
                        }}
                        onEnded={() => {
                          // Stop detection when uploaded video ends (not for camera streams)
                          if (!isCameraActive && selectedFile && isProcessing) {
                            sendLog('info', 'Video ended, stopping detection');
                            stopProcessing();
                          }
                        }}
                      />
                      <DetectionOverlay
                        detections={detections}
                        videoWidth={videoDimensions.width}
                        videoHeight={videoDimensions.height}
                        className="absolute inset-0"
                      />
                    </div>

                    {/* Video Controls */}
                    <div className="relative flex justify-center items-center mt-6">
                      {!isProcessing ? (
                        <Button
                          onClick={startProcessing}
                          disabled={!selectedFile}
                          variant="outline"
                          className="px-6 py-2"
                        >
                          <Play className="h-4 w-4 mr-2" />
                          Start Detection
                        </Button>
                      ) : (
                        <Button
                          onClick={stopProcessing}
                          variant="outline"
                          className="px-6 py-2"
                        >
                          <Square className="h-4 w-4 mr-2" />
                          Stop Detection
                        </Button>
                      )}
                      <div className="absolute right-0">
                        <FileUpload
                          onVideoSelect={handleVideoSelect}
                          onImageSelect={handleImageSelect}
                          onClear={() => {
                            if (selectedFile) {
                              handleClearVideo();
                            }
                            if (selectedImage) {
                              handleClearImage();
                            }
                          }}
                          selectedVideo={selectedFile}
                          selectedImage={selectedImage}
                          showCompactBanner={true}
                        />
                      </div>
                    </div>
                  </div>
                )}

              {/* Image Preview Section */}
              {selectedImage && (
                <>
                  <div className="flex justify-center pb-5">
                    <div className="relative inline-block max-w-full overflow-auto" >
                      <img
                        ref={imageRef}
                        alt="Uploaded image"
                        className="block max-w-full w-auto h-auto object-contain"
                        onLoad={() => {
                          if (imageRef.current) {
                            const { naturalWidth, naturalHeight } = imageRef.current;
                            setImageDimensions({ width: naturalWidth, height: naturalHeight });
                          }
                        }}
                      />
                      {imageDimensions.width > 0 && imageDimensions.height > 0 && (
                        <DetectionOverlay
                          detections={detections}
                          videoWidth={imageDimensions.width}
                          videoHeight={imageDimensions.height}
                          className="absolute top-0 left-0"
                        />
                      )}
                    </div>
                  </div>

                  {/* Image Controls */}
                  <div className="relative flex justify-center items-center mt-6">
                    {isProcessing && (
                      <div className="text-center">
                        <p className="text-sm text-muted-foreground">Processing image...</p>
                      </div>
                    )}
                    <div className="absolute right-0">
                      <FileUpload
                        onVideoSelect={handleVideoSelect}
                        onImageSelect={handleImageSelect}
                        onClear={() => {
                          if (selectedFile) {
                            handleClearVideo();
                          }
                          if (selectedImage) {
                            handleClearImage();
                          }
                        }}
                        selectedVideo={selectedFile}
                        selectedImage={selectedImage}
                        showCompactBanner={true}
                      />
                    </div>
                  </div>
                </>
              )}

              {/* Remove the Tabs section from here - it's now at the top of the page */}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

export default App;

