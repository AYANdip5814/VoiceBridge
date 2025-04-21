export interface Settings {
  language: string;
  confidence: number;
  autoStart: boolean;
  theme: 'light' | 'dark';
}

export interface TranslationResult {
  text: string;
  confidence: number;
  timestamp: number;
}

export interface VideoSource {
  id: string;
  name: string;
  thumbnail: string;
}

export interface CaptureState {
  isCapturing: boolean;
  isPaused: boolean;
  currentSource: Electron.DesktopCapturerSource | null;
}

export interface ElectronAPI {
  startCapture: (sourceId: string) => Promise<void>;
  stopCapture: () => Promise<void>;
  getVideoSources: () => Promise<VideoSource[]>;
  onTranslationResult: (callback: (result: TranslationResult) => void) => void;
  removeAllListeners: (channel: string) => void;
}

declare global {
  interface Window {
    electron: ElectronAPI;
  }
} 