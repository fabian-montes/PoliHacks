export interface DetectionResult {
  faceDetected: boolean;
  confidence?: number;
}

export enum MonitoringStatus {
  IDLE = 'IDLE',
  SCANNING = 'SCANNING',
  HIGH_ALERT = 'HIGH_ALERT', // No face
  LOW_ALERT = 'LOW_ALERT',   // Face detected
}