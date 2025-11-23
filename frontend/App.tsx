import React, { useRef, useState, useEffect } from 'react';
import WebcamFeed, { WebcamRef } from './components/WebcamFeed';
import StatusPanel from './components/StatusPanel';
import { checkFacePresence } from './services/geminiService';
import { MonitoringStatus } from './types';

const App: React.FC = () => {
  const webcamRef = useRef<WebcamRef>(null);
  const [status, setStatus] = useState<MonitoringStatus>(MonitoringStatus.IDLE);
  const [isProcessing, setIsProcessing] = useState(false);

  useEffect(() => {
    // Polling interval for face detection
    // In a real production app with Live API, this would be a stream event.
    // Here we use polling (every 1s) to balance API costs and responsiveness for the demo.
    const intervalId = setInterval(async () => {
      if (isProcessing || !webcamRef.current) return;

      const frame = webcamRef.current.captureFrame();
      if (frame) {
        setIsProcessing(true);
        setStatus(prev => prev === MonitoringStatus.IDLE ? MonitoringStatus.SCANNING : prev);
        
        try {
          const result = await checkFacePresence(frame);
          
          if (result.faceDetected) {
            setStatus(MonitoringStatus.LOW_ALERT);
          } else {
            setStatus(MonitoringStatus.HIGH_ALERT);
          }
        } catch (e) {
          console.error("Detection cycle failed", e);
        } finally {
          setIsProcessing(false);
        }
      }
    }, 1500); // Check every 1.5 seconds

    return () => clearInterval(intervalId);
  }, [isProcessing]);

  return (
    <div className="min-h-screen bg-slate-950 flex items-center justify-center p-4 md:p-8">
      <div className="w-full max-w-7xl h-[85vh] grid grid-cols-1 md:grid-cols-2 gap-6 md:gap-8">
        
        {/* Left Column: Webcam */}
        <div className="relative flex flex-col h-full">
          <WebcamFeed ref={webcamRef} />
          
          {/* Processing Indicator Overlay */}
          <div className="absolute top-6 right-6 z-10">
            <div className={`
              flex items-center gap-2 px-3 py-1.5 rounded-lg border backdrop-blur-md transition-colors duration-300
              ${isProcessing 
                ? 'bg-blue-500/20 border-blue-400/30 text-blue-200' 
                : 'bg-slate-800/50 border-slate-700/50 text-slate-400'}
            `}>
              <div className={`w-2 h-2 rounded-full ${isProcessing ? 'bg-blue-400 animate-pulse' : 'bg-slate-500'}`} />
              <span className="text-xs font-mono font-medium tracking-wide">
                AI PROCESSING: {isProcessing ? 'BUSY' : 'IDLE'}
              </span>
            </div>
          </div>
        </div>

        {/* Right Column: Status & Messages */}
        <div className="h-full">
          <StatusPanel status={status} />
        </div>

      </div>
    </div>
  );
};

export default App;