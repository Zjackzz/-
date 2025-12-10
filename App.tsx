import React, { useState, Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import { EffectComposer, Bloom, Vignette } from '@react-three/postprocessing';
import { OrbitControls } from '@react-three/drei';
import { ParticleShape, HandGesture, HandState } from './types';
import { VisionController } from './components/VisionController';
import { ParticleScene } from './components/ParticleScene';
import { Overlay } from './components/Overlay';

const App: React.FC = () => {
  const [shape, setShape] = useState<ParticleShape>(ParticleShape.TREE);
  const [color, setColor] = useState<string>('#0B6623'); // Deep Christmas Green default
  const [handState, setHandState] = useState<HandState>({
    gesture: HandGesture.NONE,
    rotation: { x: 0, y: 0 },
    pinchDistance: 1,
    isPresent: false
  });

  return (
    <div className="w-full h-full relative bg-[#050505]">
      {/* Computer Vision Layer (Non-visual logic) */}
      <VisionController onUpdate={setHandState} />

      {/* 3D Scene */}
      <Canvas
        camera={{ position: [0, 0, 12], fov: 45 }}
        dpr={[1, 2]} // Quality scaling
        gl={{ antialias: false, alpha: false }}
      >
        <color attach="background" args={['#050505']} />
        
        <Suspense fallback={null}>
          <ParticleScene 
            handState={handState} 
            shape={shape} 
            color={color} 
          />
          
          {/* Post Processing for Cinematic Glow */}
          <EffectComposer disableNormalPass>
            <Bloom 
              luminanceThreshold={0.55} // Raised from 0.2 to only bloom very bright things (Star/Highlights)
              mipmapBlur 
              intensity={0.8} // Reduced from 1.5 to prevent "washed out" look
              radius={0.6}
            />
            {/* Reduced darkness to prevent black artifacts/crushed blacks */}
            <Vignette eskil={false} offset={0.1} darkness={0.6} />
          </EffectComposer>
        </Suspense>

        {/* Orbit Controls (Enabled if no hand present to allow mouse fallback) */}
        {!handState.isPresent && (
          <OrbitControls 
            enablePan={false} 
            enableZoom={true} 
            autoRotate 
            autoRotateSpeed={0.5} 
          />
        )}
      </Canvas>

      {/* UI Overlay */}
      <Overlay 
        currentShape={shape} 
        setShape={setShape}
        currentColor={color}
        setColor={setColor}
        currentGesture={handState.gesture}
      />
    </div>
  );
};

export default App;