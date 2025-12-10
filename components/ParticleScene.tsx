import React, { useRef, useMemo, useEffect, useState } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { HandState, HandGesture, ParticleShape } from '../types';
import { generateParticles, PARTICLE_COUNT } from '../utils/geometry';

interface ParticleSceneProps {
  handState: HandState;
  shape: ParticleShape;
  color: string;
}

// A spiky star made of two crossed octahedrons
const TopStar: React.FC<{ expansionFactor: number }> = ({ expansionFactor }) => {
  const groupRef = useRef<THREE.Group>(null);

  useFrame((state) => {
    if (!groupRef.current) return;
    const time = state.clock.getElapsedTime();
    
    // Gentle rotation
    groupRef.current.rotation.y = time * 0.5;
    
    // Floating motion
    const hover = Math.sin(time * 2) * 0.1;
    
    // Position at top of tree (y=4 base), adjusted by expansion
    const targetY = (4.2 + hover) * expansionFactor;
    
    // Smoothly interpolate position
    groupRef.current.position.y = THREE.MathUtils.lerp(groupRef.current.position.y, targetY, 0.1);
    
    // Scale effect based on gesture (shrink if imploding, normal otherwise)
    // If expansionFactor is small (Fist), scale down but stay visible
    // If large (Explode), scale up slightly or stay normal
    const targetScale = Math.max(0.2, expansionFactor < 1 ? expansionFactor : 1);
    
    groupRef.current.scale.setScalar(targetScale);
  });

  return (
    <group ref={groupRef} position={[0, 4.2, 0]}>
      {/* Core Glow */}
      <mesh>
        <octahedronGeometry args={[0.3, 0]} />
        <meshBasicMaterial color="#FFD700" toneMapped={false} />
      </mesh>
      {/* Spikes 1 */}
      <mesh scale={[0.4, 1.2, 0.4]}>
        <octahedronGeometry args={[0.3, 0]} />
        <meshBasicMaterial color="#FDB931" toneMapped={false} transparent opacity={0.8} />
      </mesh>
      {/* Spikes 2 (Rotated) */}
      <mesh scale={[0.4, 1.2, 0.4]} rotation={[0, 0, Math.PI / 2]}>
        <octahedronGeometry args={[0.3, 0]} />
        <meshBasicMaterial color="#FDB931" toneMapped={false} transparent opacity={0.8} />
      </mesh>
      {/* Spikes 3 (Rotated) */}
      <mesh scale={[0.4, 1.2, 0.4]} rotation={[Math.PI / 2, 0, 0]}>
        <octahedronGeometry args={[0.3, 0]} />
        <meshBasicMaterial color="#FDB931" toneMapped={false} transparent opacity={0.8} />
      </mesh>
    </group>
  );
};

export const ParticleScene: React.FC<ParticleSceneProps> = ({ handState, shape, color }) => {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const { camera } = useThree();
  const [currentExpansion, setCurrentExpansion] = useState(1);
  
  // Create dummy object for matrix calculations
  const dummy = useMemo(() => new THREE.Object3D(), []);
  
  // Store target positions for the current shape
  const targetPositions = useMemo(() => new Float32Array(PARTICLE_COUNT * 3), []);
  // Store current positions for lerping
  const currentPositions = useRef(new Float32Array(PARTICLE_COUNT * 3));
  
  // Initialize Geometry on Shape Change
  useEffect(() => {
    const newPositions = generateParticles(shape);
    targetPositions.set(newPositions);
  }, [shape, targetPositions]);

  // Initial random scatter
  useEffect(() => {
    const initial = generateParticles(ParticleShape.SPHERE);
    currentPositions.current.set(initial.map(v => v * 3)); // Start exploded
  }, []);

  // RICH COLOR MIXING LOGIC
  useEffect(() => {
    if (!meshRef.current) return;

    const baseColor = new THREE.Color(color);
    const hsl = { h: 0, s: 0, l: 0 };
    baseColor.getHSL(hsl);
    
    const tempColor = new THREE.Color();
    const tempHsl = { h: 0, s: 0, l: 0 };

    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const rand = Math.random();
      
      // Strategy: 
      // 60% Base Theme
      // 25% Richer/Darker Analogous tones (but not too dark)
      // 15% Metallic/Holiday Accents

      if (rand < 0.6) {
        // Base variation
        tempColor.copy(baseColor);
        // Vary lightness significantly for depth but clamp bottom later
        const lVar = (Math.random() - 0.5) * 0.4; 
        const sVar = (Math.random() - 0.5) * 0.15;
        tempColor.offsetHSL(0, sVar, lVar);
      } else if (rand < 0.85) {
        // Richer tones of similar hue
        tempColor.copy(baseColor);
        tempColor.offsetHSL((Math.random() - 0.5) * 0.1, 0.2, 0); 
      } else {
        // Smart Accents based on Base Hue
        const isGreenish = Math.abs(hsl.h - 0.33) < 0.15;
        const isReddish = Math.abs(hsl.h - 0) < 0.1 || Math.abs(hsl.h - 1) < 0.1;

        if (isGreenish) {
            // Christmas Tree Vibe: Green + Red + Gold
            if (Math.random() > 0.6) tempColor.set(0xD4AF37); // Metallic Gold
            else tempColor.set(0xC41E3A); // Cardinal Red
        } else if (isReddish) {
            // Holiday Red Vibe: Red + Green + Gold
            if (Math.random() > 0.6) tempColor.set(0xD4AF37); // Metallic Gold
            else tempColor.set(0x228B22); // Forest Green
        } else {
            // General Sparkle (Gold/White)
            if (Math.random() > 0.5) tempColor.set(0xFFFFFF); // Pure White Sparkle
            else tempColor.set(0xD4AF37); // Gold
        }
      }

      // Final Brightness Check to eliminate "Black Dots"
      tempColor.getHSL(tempHsl);
      // Ensure L is at least 0.2 (was 0.4). Lowering this allows deeper colors, 
      // preventing the "washed out" look while still avoiding pure black dots.
      if (tempHsl.l < 0.2) {
        tempHsl.l = 0.2 + Math.random() * 0.2;
        tempColor.setHSL(tempHsl.h, tempHsl.s, tempHsl.l);
      }

      meshRef.current.setColorAt(i, tempColor);
    }
    
    if (meshRef.current.instanceColor) {
      meshRef.current.instanceColor.needsUpdate = true;
    }
  }, [color]);

  useFrame((state, delta) => {
    if (!meshRef.current) return;

    // 1. Handle Camera Rotation based on Hand
    if (handState.isPresent) {
      const targetCamX = Math.sin(handState.rotation.x) * 12;
      const targetCamZ = Math.cos(handState.rotation.x) * 12;
      const targetCamY = handState.rotation.y * -5;
      
      camera.position.x = THREE.MathUtils.lerp(camera.position.x, targetCamX, 0.05);
      camera.position.z = THREE.MathUtils.lerp(camera.position.z, targetCamZ, 0.05);
      camera.position.y = THREE.MathUtils.lerp(camera.position.y, targetCamY, 0.05);
      camera.lookAt(0, 0, 0);
    } else {
      // Auto rotate slowly if no hand
      const time = state.clock.getElapsedTime();
      camera.position.x = Math.sin(time * 0.1) * 12;
      camera.position.z = Math.cos(time * 0.1) * 12;
      camera.lookAt(0, 0, 0);
    }

    // 2. Particle Dynamics
    const time = state.clock.getElapsedTime();
    let expansionFactor = 1.0;
    let noiseFactor = 0.05;

    // Gesture Logic
    if (handState.gesture === HandGesture.CLOSED_FIST) {
      expansionFactor = 0.1; // Collapse
      noiseFactor = 0.01;
    } else if (handState.gesture === HandGesture.OPEN_PALM) {
      expansionFactor = 2.5; // Explode
      noiseFactor = 0.5;
    }

    // Smoothly update global expansion state for other components like the Star
    // Note: This is a bit of a hack to sync state inside useFrame, but works for visual effects
    if (Math.abs(currentExpansion - expansionFactor) > 0.01) {
        const next = THREE.MathUtils.lerp(currentExpansion, expansionFactor, 0.1);
        setCurrentExpansion(next);
        expansionFactor = next; // Use the smoothed value for particles
    } else {
        expansionFactor = currentExpansion;
    }


    // Update Particles
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const i3 = i * 3;
      
      // Get Target for this particle
      let tx = targetPositions[i3];
      let ty = targetPositions[i3 + 1];
      let tz = targetPositions[i3 + 2];

      // Apply expansion/contraction relative to center (0,0,0)
      tx *= expansionFactor;
      ty *= expansionFactor;
      tz *= expansionFactor;

      // Add "Christmas Magic" drift (Sine waves)
      ty += Math.sin(time * 0.5 + i) * noiseFactor;
      tx += Math.cos(time * 0.3 + i) * noiseFactor;

      // Lerp current position to target
      const speed = handState.gesture === HandGesture.OPEN_PALM ? 0.1 : 0.03;

      currentPositions.current[i3] = THREE.MathUtils.lerp(currentPositions.current[i3], tx, speed);
      currentPositions.current[i3+1] = THREE.MathUtils.lerp(currentPositions.current[i3+1], ty, speed);
      currentPositions.current[i3+2] = THREE.MathUtils.lerp(currentPositions.current[i3+2], tz, speed);

      // Set Instance Matrix
      dummy.position.set(
        currentPositions.current[i3],
        currentPositions.current[i3 + 1],
        currentPositions.current[i3 + 2]
      );
      
      // Scale particles
      const baseScale = (i % 15 === 0) ? 0.3 : 0.12; 
      dummy.scale.set(baseScale, baseScale, baseScale);
      
      dummy.lookAt(camera.position);
      dummy.updateMatrix();
      meshRef.current.setMatrixAt(i, dummy.matrix);
    }
    
    meshRef.current.instanceMatrix.needsUpdate = true;
  });

  return (
    <>
      {shape === ParticleShape.TREE && <TopStar expansionFactor={currentExpansion} />}
      
      <instancedMesh ref={meshRef} args={[undefined, undefined, PARTICLE_COUNT]}>
        {/* Changed to CircleGeometry for smoother "dots" without square corners */}
        <circleGeometry args={[0.5, 8]} />
        <meshBasicMaterial 
          color="#ffffff" 
          transparent 
          opacity={0.6} // Reduced from 0.9 to reduce overwhelming brightness
          side={THREE.DoubleSide}
          blending={THREE.AdditiveBlending}
          depthWrite={false}
        />
      </instancedMesh>
    </>
  );
};