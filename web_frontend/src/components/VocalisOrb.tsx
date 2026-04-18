"use client";

import { useEffect, useRef } from "react";

type AssistantState = "idle" | "listening" | "thinking" | "speaking" | "error";

interface VocalisOrbProps {
  state: AssistantState;
}

export default function VocalisOrb({ state }: VocalisOrbProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animationFrameId: number;
    let time = 0;

    const colors = {
      idle: { r: 59, g: 130, b: 246 },      // Blue
      listening: { r: 52, g: 211, b: 153 }, // Emerald
      thinking: { r: 139, g: 92, b: 246 },  // Purple
      speaking: { r: 251, g: 146, b: 60 },  // Orange
      error: { r: 239, g: 68, b: 68 },     // Red
    };

    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      canvas.width = canvas.clientWidth * dpr;
      canvas.height = canvas.clientHeight * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0); // Reset and scale correctly
    };

    window.addEventListener("resize", resize);
    resize();

    const draw = () => {
      time += state === "listening" ? 0.05 : 0.02;
      const { width, height } = canvas.getBoundingClientRect();
      ctx.clearRect(0, 0, width, height);

      const center = { x: width / 2, y: height / 2 };
      const baseRadius = width * 0.3;
      const targetColor = colors[state];

      // Layered Glow Effect
      for (let i = 0; i < 3; i++) {
        const pulse = Math.sin(time + i) * 10;
        const radius = baseRadius + pulse + (i * 20);
        
        const gradient = ctx.createRadialGradient(
          center.x, center.y, radius * 0.2,
          center.x, center.y, radius
        );
        
        const alpha = 0.15 / (i + 1);
        gradient.addColorStop(0, `rgba(${targetColor.r}, ${targetColor.g}, ${targetColor.b}, ${alpha})`);
        gradient.addColorStop(1, "rgba(0, 0, 0, 0)");

        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(center.x, center.y, radius, 0, Math.PI * 2);
        ctx.fill();
      }

      // Procedural Plasma Core
      ctx.save();
      ctx.beginPath();
      for (let i = 0; i < 360; i += 2) {
        const angle = (i * Math.PI) / 180;
        // Wavy boundary logic
        const wave = Math.sin(angle * 4 + time * 2) * 5 + 
                     Math.cos(angle * 8 - time * 1.5) * 3;
        
        const r = baseRadius * 0.8 + wave;
        const x = center.x + Math.cos(angle) * r;
        const y = center.y + Math.sin(angle) * r;

        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.closePath();
      
      const coreGradient = ctx.createRadialGradient(
        center.x - 20, center.y - 20, 0,
        center.x, center.y, baseRadius
      );
      coreGradient.addColorStop(0, `rgba(${targetColor.r + 50}, ${targetColor.g + 50}, ${targetColor.b + 50}, 0.8)`);
      coreGradient.addColorStop(0.5, `rgba(${targetColor.r}, ${targetColor.g}, ${targetColor.b}, 0.5)`);
      coreGradient.addColorStop(1, "rgba(0,0,0,0)");
      
      ctx.fillStyle = coreGradient;
      ctx.fill();
      ctx.restore();

      animationFrameId = requestAnimationFrame(draw);
    };

    draw();

    return () => {
      window.removeEventListener("resize", resize);
      cancelAnimationFrame(animationFrameId);
    };
  }, [state]);

  return <canvas ref={canvasRef} className="w-full h-full" />;
}
