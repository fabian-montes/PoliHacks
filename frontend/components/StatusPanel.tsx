import React from 'react';
import { MonitoringStatus } from '../types';

interface StatusPanelProps {
  status: MonitoringStatus;
}

const StatusPanel: React.FC<StatusPanelProps> = ({ status }) => {
  
  const isHighAlert = status === MonitoringStatus.HIGH_ALERT;
  const isScanning = status === MonitoringStatus.SCANNING || status === MonitoringStatus.IDLE;

  // Determine styles based on status
  let alertBg = "bg-slate-800";
  let alertBorder = "border-slate-700";
  let alertText = "text-slate-400";
  let statusMessage = "Inicializando sistema...";
  let statusIcon = (
    <svg className="w-12 h-12 animate-spin" fill="none" viewBox="0 0 24 24">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
    </svg>
  );

  if (status === MonitoringStatus.HIGH_ALERT) {
    alertBg = "bg-emerald-950/40";
    alertBorder = "border-emerald-500";
    alertText = "text-emerald-500";
    statusMessage = "NIVEL DE RENDIMIENTO ALTO";
    statusIcon = (
      <svg className="w-16 h-16 animate-bounce" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
      </svg>
    );
  } else if (status === MonitoringStatus.LOW_ALERT) {
    alertBg = "bg-red-950/40";
    alertBorder = "border-red-500";
    alertText = "text-red-500";
    statusMessage = "NIVEL DE RENDIMIENTO BAJO";
    statusIcon = (
      <svg className="w-16 h-16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
         <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    );
  }

  return (
    <div className="h-full flex flex-col gap-6">
      
      {/* Header Section */}
      <div className="flex items-center justify-between pb-6 border-b border-slate-800">
        <h1 className="text-2xl font-bold tracking-tight text-white">
          Panel de Control
        </h1>
        <div className="flex items-center gap-2">
          <span className={`w-3 h-3 rounded-full ${isScanning ? 'bg-yellow-500' : 'bg-blue-500'}`}></span>
          <span className="text-xs text-slate-400 font-mono uppercase">
            {isScanning ? 'ESCANEO ACTIVO' : 'MONITOREO'}
          </span>
        </div>
      </div>

      {/* Main Alert Box */}
      <div className={`
        flex-1 flex flex-col items-center justify-center text-center p-8 rounded-2xl border-2 transition-all duration-500
        ${alertBg} ${alertBorder} ${alertText} shadow-lg relative overflow-hidden group
      `}>
        {/* Decorative background glow */}
        <div className={`absolute inset-0 opacity-10 blur-3xl rounded-full transform scale-150 transition-colors duration-500 ${isHighAlert ? 'bg-red-500' : 'bg-emerald-500'}`}></div>
        
        <div className="relative z-10 space-y-4">
          <div className="flex justify-center mb-6">
            {statusIcon}
          </div>
          <h2 className="text-4xl font-black tracking-tighter uppercase drop-shadow-sm">
            {statusMessage}
          </h2>
          <p className="text-sm opacity-80 font-mono">
            {!isHighAlert ? 'SISTEMA NO DETECTA OPERADOR' : 'OPERADOR IDENTIFICADO CORRECTAMENTE'}
          </p>
        </div>
      </div>

      {/* Lorem Ipsum Box */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 relative">
        <h3 className="text-xs font-bold text-slate-500 uppercase mb-3 flex items-center gap-2">
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
          Información del Sistema
        </h3>
        <p className="text-slate-400 text-sm leading-relaxed text-justify">
          Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
        </p>
      </div>

    </div>
  );
};

export default StatusPanel;