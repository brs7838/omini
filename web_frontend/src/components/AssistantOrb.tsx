"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useMemo } from "react";

type AssistantState = "idle" | "listening" | "thinking" | "speaking" | "error";

interface AssistantOrbProps {
  state: AssistantState;
}

export default function AssistantOrb({ state }: AssistantOrbProps) {
  const colors = useMemo(() => {
    switch (state) {
      case "listening":
        return ["#06b6d4", "#22c55e", "#06b6d4"]; // Cyan to Green
      case "thinking":
        return ["#6366f1", "#a855f7", "#6366f1"]; // Indigo to Purple
      case "speaking":
        return ["#f43f5e", "#fb923c", "#f43f5e"]; // Rose to Orange
      case "error":
        return ["#ef4444", "#7f1d1d", "#ef4444"]; // Red
      default:
        return ["#3b82f6", "#8b5cf6", "#3b82f6"]; // Blue to Violet
    }
  }, [state]);

  return (
    <div className="relative flex items-center justify-center w-64 h-64">
      {/* Background Glow */}
      <AnimatePresence mode="wait">
        <motion.div
          key={state}
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 0.4, scale: 1.2 }}
          exit={{ opacity: 0, scale: 0.8 }}
          transition={{ duration: 1 }}
          className="absolute inset-0 rounded-full blur-3xl"
          style={{
            background: `radial-gradient(circle, ${colors[0]} 0%, transparent 70%)`,
          }}
        />
      </AnimatePresence>

      {/* Main Orb */}
      <motion.div
        animate={{
          scale: state === "listening" ? [1, 1.1, 1] : state === "speaking" ? [1, 1.2, 1] : [1, 1.05, 1],
          boxShadow: [
            `0 0 20px ${colors[0]}`,
            `0 0 40px ${colors[1]}`,
            `0 0 20px ${colors[0]}`,
          ],
        }}
        transition={{
          duration: state === "listening" ? 0.8 : 2,
          repeat: Infinity,
          ease: "easeInOut",
        }}
        className="relative z-10 w-48 h-48 rounded-full flex items-center justify-center overflow-hidden"
        style={{
          background: `radial-gradient(circle at 30% 30%, ${colors[0]} 0%, ${colors[1]} 50%, ${colors[2]} 100%)`,
        }}
      >
        {/* Shimmer Effect */}
        <motion.div
          animate={{
            rotate: 360,
          }}
          transition={{
            duration: 4,
            repeat: Infinity,
            ease: "linear",
          }}
          className="absolute inset-0 opacity-30"
          style={{
            background: "conic-gradient(from 0deg, transparent, white, transparent)",
          }}
        />

        {/* Inner Core */}
        <div className="w-40 h-40 rounded-full bg-black/20 backdrop-blur-sm border border-white/10" />
      </motion.div>

      {/* Status Ring */}
      <svg className="absolute inset-0 w-full h-full -rotate-90">
        <motion.circle
          cx="128"
          cy="128"
          r="110"
          stroke="white"
          strokeWidth="2"
          strokeDasharray="10 20"
          fill="transparent"
          className="opacity-10"
          animate={{
            rotate: 360,
          }}
          transition={{
            duration: 10,
            repeat: Infinity,
            ease: "linear",
          }}
        />
      </svg>
    </div>
  );
}
