import React from 'react';
import { ParticleShape, HandGesture } from '../types';
import { HexColorPicker } from 'react-colorful';
import { TreePine, Heart, Star, Circle, Hand, Disc } from 'lucide-react';

interface OverlayProps {
  currentShape: ParticleShape;
  setShape: (s: ParticleShape) => void;
  currentColor: string;
  setColor: (c: string) => void;
  currentGesture: HandGesture;
}

export const Overlay: React.FC<OverlayProps> = ({ 
  currentShape, setShape, currentColor, setColor, currentGesture
}) => {
  const [showColor, setShowColor] = React.useState(false);

  const shapes = [
    { id: ParticleShape.TREE, icon: TreePine, label: 'Tree' },
    { id: ParticleShape.HEART, icon: Heart, label: 'Love' },
    { id: ParticleShape.STAR, icon: Star, label: 'Star' },
    { id: ParticleShape.SPHERE, icon: Circle, label: 'Orb' },
    { id: ParticleShape.RING, icon: Disc, label: 'Ring' },
  ];

  return (
    <div className="absolute inset-0 pointer-events-none flex flex-col justify-between p-6">
      
      {/* Header */}
      <div className="flex justify-between items-start pointer-events-auto">
        <div className="bg-black/40 backdrop-blur-md p-4 rounded-xl border border-white/10 text-white">
          <h1 className="text-2xl font-bold bg-gradient-to-r from-yellow-400 to-red-500 bg-clip-text text-transparent">
            Gesture Magic
          </h1>
          <p className="text-xs text-gray-400 mt-1 uppercase tracking-widest">Interactive 3D Particles</p>
        </div>

        {/* Status Indicator */}
        <div className="bg-black/60 backdrop-blur-md px-4 py-2 rounded-full border border-green-500/30 flex items-center gap-3">
          <div className={`w-3 h-3 rounded-full ${currentGesture !== HandGesture.NONE ? 'bg-green-500 animate-pulse' : 'bg-red-900'}`} />
          <span className="text-white font-mono text-sm uppercase">
            {currentGesture === HandGesture.NONE ? 'No Hand Detected' : currentGesture}
          </span>
          <Hand className="w-4 h-4 text-white/50" />
        </div>
      </div>

      {/* Instructions */}
      <div className="absolute top-1/2 left-6 -translate-y-1/2 space-y-4 opacity-50 hover:opacity-100 transition-opacity pointer-events-auto hidden md:block">
         <div className="bg-black/30 backdrop-blur-sm p-3 rounded-lg border-l-4 border-yellow-500 text-white text-sm max-w-[200px]">
           <strong className="text-yellow-400 block mb-1">✊ Fist</strong>
           To collapse / clump particles.
         </div>
         <div className="bg-black/30 backdrop-blur-sm p-3 rounded-lg border-l-4 border-red-500 text-white text-sm max-w-[200px]">
           <strong className="text-red-400 block mb-1">✋ Open Hand</strong>
           To explode / scatter particles.
         </div>
         <div className="bg-black/30 backdrop-blur-sm p-3 rounded-lg border-l-4 border-blue-500 text-white text-sm max-w-[200px]">
           <strong className="text-blue-400 block mb-1">👋 Move Hand</strong>
           Left/Right to rotate view.
         </div>
      </div>

      {/* Bottom Controls */}
      <div className="flex flex-col items-center gap-4 pointer-events-auto">
        
        {/* Shape Selectors */}
        <div className="flex gap-2 bg-black/50 backdrop-blur-xl p-2 rounded-2xl border border-white/10 overflow-x-auto max-w-full">
          {shapes.map((s) => (
            <button
              key={s.id}
              onClick={() => setShape(s.id)}
              className={`p-3 rounded-xl transition-all duration-300 flex flex-col items-center gap-1 min-w-[70px] ${
                currentShape === s.id 
                  ? 'bg-gradient-to-br from-green-800 to-green-900 text-yellow-400 shadow-[0_0_15px_rgba(46,125,50,0.5)]' 
                  : 'hover:bg-white/10 text-gray-400 hover:text-white'
              }`}
            >
              <s.icon size={20} />
              <span className="text-[10px] font-medium tracking-wide">{s.label}</span>
            </button>
          ))}
        </div>

        {/* Color Picker Toggle */}
        <div className="relative">
          <button 
            onClick={() => setShowColor(!showColor)}
            className="flex items-center gap-2 bg-black/50 backdrop-blur-md px-4 py-2 rounded-full border border-white/10 text-white text-sm hover:bg-white/10 transition-colors"
          >
            <div className="w-4 h-4 rounded-full border border-white/50" style={{ background: currentColor }} />
            <span>Customize Color</span>
          </button>
          
          {showColor && (
            <div className="absolute bottom-full mb-4 left-1/2 -translate-x-1/2 bg-black/80 backdrop-blur-xl p-4 rounded-2xl border border-white/10">
              <HexColorPicker color={currentColor} onChange={setColor} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
};