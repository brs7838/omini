"use client";

import { useState } from "react";
import { Phone, X, Delete } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface DialerProps {
  isOpen: boolean;
  onClose: () => void;
  onDial: (number: string) => void;
}

export default function Dialer({ isOpen, onClose, onDial }: DialerProps) {
  const [number, setNumber] = useState("");
  const [isDialing, setIsDialing] = useState(false);

  const handleKeypad = (digit: string) => {
    if (number.length < 15) setNumber(prev => prev + digit);
  };

  const handleBackspace = () => {
    setNumber(prev => prev.slice(0, -1));
  };

  const handleCall = async () => {
    if (number.length >= 3 && !isDialing) {
      setIsDialing(true);
      try {
        await onDial(number);
        setNumber("");
        onClose();
      } finally {
        setIsDialing(false);
      }
    }
  };

  const buttons = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "*", "0", "#"];

  return (
    <AnimatePresence>
      {isOpen && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4">
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="absolute inset-0 bg-black/60 backdrop-blur-sm"
          />
          
          <motion.div
            initial={{ scale: 0.9, opacity: 0, y: 20 }}
            animate={{ scale: 1, opacity: 1, y: 0 }}
            exit={{ scale: 0.9, opacity: 0, y: 20 }}
            className="relative w-full max-w-[320px] bg-zinc-900/90 border border-white/10 rounded-[2.5rem] shadow-2xl overflow-hidden p-8 backdrop-blur-2xl"
          >
            {/* Header */}
            <div className="flex justify-between items-center mb-8">
               <h3 className="text-xs font-bold uppercase tracking-[0.2em] text-zinc-500">New Call</h3>
               <button onClick={onClose} className="p-2 -mr-2 text-zinc-400 hover:text-white transition-colors">
                  <X className="w-4 h-4" />
               </button>
            </div>

            {/* Display */}
            <div className="flex flex-col items-center mb-8">
              <div className="w-full text-center h-12 flex items-center justify-center overflow-hidden">
                <span className="text-3xl font-light tracking-wider text-white truncate px-4">
                  {number || "..."}
                </span>
              </div>
            </div>

            {/* Keypad */}
            <div className="grid grid-cols-3 gap-4 mb-8">
               {buttons.map(btn => (
                 <button
                   key={btn}
                   onClick={() => handleKeypad(btn)}
                   className="w-16 h-16 rounded-full bg-white/5 border border-white/5 hover:bg-white/10 hover:border-white/10 active:scale-95 transition-all flex flex-col items-center justify-center group"
                 >
                   <span className="text-xl font-medium text-zinc-300 group-hover:text-white">{btn}</span>
                 </button>
               ))}
            </div>

            {/* Actions */}
            <div className="flex items-center justify-between gap-4">
              <button 
                onClick={handleBackspace}
                className="w-16 h-16 rounded-full flex items-center justify-center text-zinc-500 hover:text-zinc-300 transition-colors"
                disabled={!number}
              >
                <Delete className="w-6 h-6" />
              </button>

              <button
                onClick={handleCall}
                disabled={!number || isDialing}
                className={`w-20 h-20 rounded-full flex items-center justify-center transition-all ${
                  number && !isDialing
                    ? 'bg-emerald-500 text-black shadow-[0_0_30px_rgba(16,185,129,0.4)] hover:scale-105 active:scale-95' 
                    : 'bg-zinc-800 text-zinc-600 grayscale cursor-not-allowed'
                }`}
              >
                {isDialing ? (
                  <motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 2, ease: "linear" }}>
                    <Phone className="w-8 h-8 opacity-50" />
                  </motion.div>
                ) : (
                  <Phone className="w-8 h-8 fill-current" />
                )}
              </button>

              <div className="w-16 h-16" /> {/* Spacer for balance */}
            </div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
}
