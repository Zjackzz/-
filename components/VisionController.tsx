import React, { useEffect, useRef, useState } from 'react';
import { FilesetResolver, Hands, HandLandmarker, HandLandmarkerResult } from '@mediapipe/tasks-vision';
import { HandGesture, HandState } from '../types';
import { Camera, Upload, AlertCircle, Loader2 } from 'lucide-react';

// ==============================================================================
// CONFIGURATION
// ==============================================================================

// Use GitMirror (China friendly) for WASM, but Google for Model (with fallback)
const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm";
const MODEL_URL_GOOGLE = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

interface VisionControllerProps {
  onUpdate: (state: HandState) => void;
}

export const VisionController: React.FC<VisionControllerProps> = ({ onUpdate }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const lastVideoTime = useRef(-1);
  const requestRef = useRef<number>(0);
  const landmarkerRef = useRef<HandLandmarker | null>(null);
  
  // UI States
  const [isStarted, setIsStarted] = useState(false);
  const [status, setStatus] = useState<string>(''); // 'initializing', 'loading_model', 'ready', 'error'
  const [errorMessage, setErrorMessage] = useState<string>('');
  const [showManualUpload, setShowManualUpload] = useState(false);

  // Initialize on user interaction
  const startExperience = async () => {
    setIsStarted(true);
    setStatus('initializing');
    
    try {
        await initVision();
    } catch (e) {
        // Errors handled inside initVision
    }
  };

  const initVision = async () => {
    try {
      setStatus('loading_wasm');
      // 1. Load WASM
      const vision = await FilesetResolver.forVisionTasks(WASM_URL);
      
      // 2. Load Model
      setStatus('loading_model');
      
      // Set a timeout to show manual upload if it takes too long (common in China)
      const timeoutId = setTimeout(() => {
        if (!landmarkerRef.current) {
          setShowManualUpload(true);
        }
      }, 3000); // Show fallback after 3 seconds

      try {
        const landmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: MODEL_URL_GOOGLE,
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numHands: 1
        });
        clearTimeout(timeoutId);
        landmarkerRef.current = landmarker;
        startCamera();
      } catch (modelErr) {
        clearTimeout(timeoutId);
        console.warn("Auto-load failed, waiting for manual upload", modelErr);
        setShowManualUpload(true);
        setStatus('waiting_for_file');
        setErrorMessage("无法连接 Google 服务器下载模型");
      }

    } catch (err: any) {
      console.error("Critical Error:", err);
      setStatus('error');
      setErrorMessage(err.message || "初始化失败");
    }
  };

  // Manual File Handler
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    const file = e.target.files[0];
    const url = URL.createObjectURL(file);
    
    try {
        setStatus('loading_model_manual');
        const vision = await FilesetResolver.forVisionTasks(WASM_URL);
        const landmarker = await HandLandmarker.createFromOptions(vision, {
            baseOptions: {
              modelAssetPath: url,
              delegate: "GPU"
            },
            runningMode: "VIDEO",
            numHands: 1
        });
        landmarkerRef.current = landmarker;
        setShowManualUpload(false);
        startCamera();
    } catch (err: any) {
        setStatus('error');
        setErrorMessage("文件无效: " + err.message);
    }
  };

  const startCamera = async () => {
    setStatus('requesting_camera');
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setStatus('error');
        setErrorMessage("浏览器不支持摄像头访问 (需要 HTTPS)");
        return;
    }
    
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: 640, 
          height: 480,
          facingMode: "user" // Front camera
        } 
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.addEventListener('loadeddata', () => {
            setStatus('ready');
            predictWebcam();
        });
      }
    } catch (err: any) {
      console.error("Camera denied:", err);
      setStatus('error');
      setErrorMessage("摄像头权限被拒绝");
    }
  };

  const predictWebcam = () => {
    if (!landmarkerRef.current || !videoRef.current) return;

    if (lastVideoTime.current !== videoRef.current.currentTime) {
      lastVideoTime.current = videoRef.current.currentTime;
      let startTimeMs = performance.now();
      const result = landmarkerRef.current.detectForVideo(videoRef.current, startTimeMs);
      processLandmarks(result);
    }
    
    requestRef.current = requestAnimationFrame(predictWebcam);
  };

  const processLandmarks = (result: HandLandmarkerResult) => {
    if (result.landmarks.length > 0) {
      const landmarks = result.landmarks[0];
      const wrist = landmarks[0];
      const tips = [8, 12, 16, 20].map(i => landmarks[i]);
      
      const dists = tips.map(tip => Math.sqrt(
        Math.pow(tip.x - wrist.x, 2) + 
        Math.pow(tip.y - wrist.y, 2) + 
        Math.pow(tip.z - wrist.z, 2)
      ));
      
      const avgDist = dists.reduce((a, b) => a + b, 0) / dists.length;
      
      const thumbTip = landmarks[4];
      const indexTip = landmarks[8];
      const pinchDist = Math.sqrt(
        Math.pow(thumbTip.x - indexTip.x, 2) +
        Math.pow(thumbTip.y - indexTip.y, 2)
      );

      let gesture = HandGesture.OPEN_PALM;
      if (avgDist < 0.25) { 
        gesture = HandGesture.CLOSED_FIST;
      }
      
      const rotX = (wrist.x - 0.5) * 4; 
      const rotY = (wrist.y - 0.5) * 2;

      onUpdate({
        gesture,
        rotation: { x: rotX, y: rotY },
        pinchDistance: pinchDist,
        isPresent: true
      });

    } else {
      onUpdate({
        gesture: HandGesture.NONE,
        rotation: { x: 0, y: 0 },
        pinchDistance: 1,
        isPresent: false
      });
    }
  };

  // CLEANUP
  useEffect(() => {
    return () => {
        cancelAnimationFrame(requestRef.current);
        if (videoRef.current && videoRef.current.srcObject) {
            (videoRef.current.srcObject as MediaStream).getTracks().forEach(t => t.stop());
        }
    }
  }, []);

  return (
    <>
        {/* Mobile-friendly Preview: Always visible now, removed 'hidden' */}
        <video 
            ref={videoRef} 
            autoPlay 
            playsInline 
            className={`fixed bottom-4 right-4 w-24 h-32 md:w-32 md:h-24 object-cover rounded-lg border-2 border-green-500/50 opacity-80 z-40 scale-x-[-1] transition-opacity duration-500 ${status === 'ready' ? 'opacity-80' : 'opacity-0'}`} 
            muted
        />

        {/* Start / Status Overlay */}
        {(!isStarted || status !== 'ready') && (
            <div className="absolute inset-0 z-50 flex items-center justify-center bg-black/90 backdrop-blur-sm p-6 text-center">
                
                {/* 1. Start Button */}
                {!isStarted && (
                    <div className="space-y-6 max-w-sm">
                        <div className="w-16 h-16 bg-green-900/50 rounded-full flex items-center justify-center mx-auto border border-green-500 animate-pulse">
                            <Camera className="w-8 h-8 text-green-400" />
                        </div>
                        <div>
                            <h2 className="text-2xl font-bold text-white mb-2">准备好了吗？</h2>
                            <p className="text-gray-400 text-sm">此应用需要开启摄像头来识别手势。<br/>数据仅在本地处理，不会上传。</p>
                        </div>
                        <button 
                            onClick={startExperience}
                            className="w-full py-3 px-6 bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-500 hover:to-emerald-500 text-white font-bold rounded-xl shadow-[0_0_20px_rgba(16,185,129,0.3)] transition-all active:scale-95 flex items-center justify-center gap-2"
                        >
                            <Camera size={20} />
                            开启摄像头体验
                        </button>
                    </div>
                )}

                {/* 2. Loading State */}
                {isStarted && status !== 'ready' && status !== 'error' && !showManualUpload && (
                    <div className="space-y-4">
                        <Loader2 className="w-10 h-10 text-green-500 animate-spin mx-auto" />
                        <p className="text-white font-medium">
                            {status === 'loading_wasm' && '正在加载基础组件...'}
                            {status === 'loading_model' && '正在下载 AI 模型...'}
                            {status === 'requesting_camera' && '请允许摄像头权限...'}
                        </p>
                    </div>
                )}

                {/* 3. Manual Upload Fallback */}
                {isStarted && showManualUpload && status !== 'ready' && (
                    <div className="bg-gray-900 p-6 rounded-2xl border border-white/10 max-w-sm w-full">
                        <div className="mb-4 text-yellow-500 flex justify-center"><AlertCircle size={32} /></div>
                        <h3 className="text-lg font-bold text-white mb-2">下载模型太慢？</h3>
                        <p className="text-sm text-gray-400 mb-6">
                           网络连接似乎不畅。请手动选择模型文件 (hand_landmarker.task) 继续。
                        </p>
                        <label className="block w-full cursor-pointer">
                            <div className="w-full py-3 px-4 bg-white/10 hover:bg-white/20 border border-dashed border-white/30 rounded-xl flex items-center justify-center gap-2 text-white transition-colors">
                                <Upload size={18} />
                                <span>点击选择文件</span>
                            </div>
                            <input 
                                type="file" 
                                accept=".task,.bin" 
                                onChange={handleFileUpload} 
                                className="hidden" 
                            />
                        </label>
                        <p className="mt-4 text-xs text-gray-500">
                           如果你没有该文件，<a href="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" target="_blank" rel="noopener noreferrer" className="text-green-400 underline">点击这里下载</a>
                        </p>
                    </div>
                )}

                {/* 4. Error State */}
                {status === 'error' && (
                    <div className="max-w-xs">
                         <div className="text-red-500 mb-2 mx-auto flex justify-center"><AlertCircle size={40} /></div>
                         <p className="text-white font-bold mb-1">出错了</p>
                         <p className="text-red-400 text-sm mb-4">{errorMessage}</p>
                         <button onClick={() => window.location.reload()} className="px-4 py-2 bg-white/10 rounded-lg text-white text-sm">刷新重试</button>
                    </div>
                )}
            </div>
        )}
    </>
  );
};
