import * as THREE from 'three';
import { ParticleShape } from '../types';

export const PARTICLE_COUNT = 3000;

const random = (min: number, max: number) => Math.random() * (max - min) + min;

export const generateParticles = (shape: ParticleShape): Float32Array => {
  const data = new Float32Array(PARTICLE_COUNT * 3);

  for (let i = 0; i < PARTICLE_COUNT; i++) {
    const i3 = i * 3;
    let x = 0, y = 0, z = 0;

    switch (shape) {
      case ParticleShape.TREE:
        // Cone spiral
        const t = i / PARTICLE_COUNT;
        const angle = t * Math.PI * 20;
        const radius = (1 - t) * 4; // Wider at bottom
        x = Math.cos(angle) * radius + random(-0.2, 0.2);
        z = Math.sin(angle) * radius + random(-0.2, 0.2);
        y = (t * 8) - 4; // Height from -4 to 4
        break;

      case ParticleShape.HEART:
        // Heart parametric equations
        const scale = 0.25;
        const heartAng = Math.random() * Math.PI * 2;
        // x = 16sin^3(t)
        // y = 13cos(t) - 5cos(2t) - 2cos(3t) - cos(4t)
        x = scale * 16 * Math.pow(Math.sin(heartAng), 3);
        y = scale * (13 * Math.cos(heartAng) - 5 * Math.cos(2 * heartAng) - 2 * Math.cos(3 * heartAng) - Math.cos(4 * heartAng));
        z = random(-1, 1) * scale * 5; 
        break;

      case ParticleShape.SPHERE:
        const s_u = Math.random();
        const s_v = Math.random();
        const s_theta = 2 * Math.PI * s_u;
        const s_phi = Math.acos(2 * s_v - 1);
        const s_r = 4.5;
        x = s_r * Math.sin(s_phi) * Math.cos(s_theta);
        y = s_r * Math.sin(s_phi) * Math.sin(s_theta);
        z = s_r * Math.cos(s_phi);
        break;

      case ParticleShape.STAR:
        // Using radial distribution with spikes
        const st_angle = Math.random() * Math.PI * 2;
        const outerRadius = 5;
        const innerRadius = 2;
        const rVar = Math.sin(st_angle * 5) > 0 ? outerRadius : innerRadius;
        const finalR = Math.random() * rVar;
        x = Math.cos(st_angle) * finalR;
        y = Math.sin(st_angle) * finalR;
        z = random(-1, 1);
        break;
        
      case ParticleShape.RING:
        const r_angle = Math.random() * Math.PI * 2;
        const r_dist = 4 + random(-0.5, 0.5);
        x = Math.cos(r_angle) * r_dist;
        z = Math.sin(r_angle) * r_dist;
        y = random(-1, 1);
        break;
    }

    data[i3] = x;
    data[i3 + 1] = y;
    data[i3 + 2] = z;
  }

  return data;
};