export enum ParticleShape {
  TREE = 'Tree',
  HEART = 'Heart',
  STAR = 'Star',
  SPHERE = 'Sphere',
  RING = 'Ring'
}

export enum HandGesture {
  NONE = 'None',
  OPEN_PALM = 'Open Palm',
  CLOSED_FIST = 'Closed Fist',
  POINTING = 'Pointing'
}

export interface HandState {
  gesture: HandGesture;
  rotation: { x: number; y: number };
  pinchDistance: number;
  isPresent: boolean;
}

export interface AppState {
  shape: ParticleShape;
  color: string;
  bloomStrength: number;
  setShape: (s: ParticleShape) => void;
  setColor: (c: string) => void;
  setBloomStrength: (n: number) => void;
}