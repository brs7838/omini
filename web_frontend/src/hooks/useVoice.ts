"use client";

import { useState, useRef, useCallback, useEffect } from "react";

export type AssistantState = "idle" | "listening" | "thinking" | "speaking" | "error";

interface Message {
  role: "user" | "ai";
  text: string;
}

export function useVoice(activeVoiceId: string = "ravi") {
  const [state, setState] = useState<AssistantState>("idle");
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLive, setIsLive] = useState(false);
  const [isCloud, setIsCloud] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const nextStartTimeRef = useRef<number>(0);
  const connectRef = useRef<((cloud: boolean) => void) | null>(null);
  const scheduledNodesRef = useRef<AudioBufferSourceNode[]>([]);

  // --- Audio playback (declared before connect so it can be referenced) ---
  const playSerializedAudio = useCallback(async (data: Blob, sampleRate: number) => {
    if (!audioContextRef.current) return;

    const arrayBuffer = await data.arrayBuffer();
    const int16Array = new Int16Array(arrayBuffer);
    const float32Array = new Float32Array(int16Array.length);

    for (let i = 0; i < int16Array.length; i++) {
      float32Array[i] = int16Array[i] / 32768.0;
    }

    const buffer = audioContextRef.current.createBuffer(1, float32Array.length, sampleRate);
    buffer.getChannelData(0).set(float32Array);

    const source = audioContextRef.current.createBufferSource();
    source.buffer = buffer;
    source.connect(audioContextRef.current.destination);

    const now = audioContextRef.current.currentTime;
    let startTime = Math.max(nextStartTimeRef.current, now);

    console.log(`[Audio] Playing chunk: ${buffer.duration.toFixed(3)}s @ ${startTime.toFixed(3)}s (ContextTime: ${now.toFixed(3)}s)`);
    source.start(startTime);
    scheduledNodesRef.current.push(source);
    source.onended = () => {
      // Clean up reference after playback to prevent memory leaks
      scheduledNodesRef.current = scheduledNodesRef.current.filter(n => n !== source);
    };

    nextStartTimeRef.current = startTime + buffer.duration;

    setState("speaking");
  }, []);

  // --- WebSocket connection ---
  const connect = useCallback((cloud: boolean) => {
    if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);

    const endpoint = cloud ? "ws/openai" : "ws";
    const ws = new WebSocket(`ws://localhost:8000/${endpoint}?voice_id=${activeVoiceId}`);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log(`Connected to Vaani Backend (${cloud ? 'Cloud' : 'Local'} Mode)`);
      setIsLive(true);
      setState("idle");
      reconnectAttemptsRef.current = 0;
    };

    ws.onmessage = async (event) => {
      if (typeof event.data === "string") {
        const msg = JSON.parse(event.data);
        if (msg.type === "transcript") {
          setMessages((prev) => [...prev, msg.data]);
        } else if (msg.type === "status") {
          if (msg.data === "thinking") setState("thinking");
          if (msg.data === "listening") setState("listening");
          if (msg.data === "idle") setState("idle");
          if (msg.data === "interrupted") {
            setState("idle");
            scheduledNodesRef.current.forEach(node => {
              try { node.stop(); } catch (e) {}
            });
            scheduledNodesRef.current = [];
            nextStartTimeRef.current = audioContextRef.current?.currentTime || 0;
          }
        } else if (msg.type === "llm_chunk") {
          setState("speaking");
        }
      } else {
        if (audioContextRef.current?.state === "suspended") {
          audioContextRef.current.resume();
        }
        console.log(`[Audio] Received chunk: ${event.data.byteLength} bytes`);
        playSerializedAudio(event.data, 24000);
      }
    };

    ws.onclose = () => {
      if (reconnectAttemptsRef.current < 5) {
        const delay = Math.min(1000 * Math.pow(2, reconnectAttemptsRef.current), 10000);
        console.log(`WS Closed. Reconnecting in ${delay}ms... (Attempt ${reconnectAttemptsRef.current + 1})`);
        reconnectTimeoutRef.current = setTimeout(() => {
          reconnectAttemptsRef.current += 1;
          connectRef.current?.(cloud);
        }, delay);
        setState("thinking");
      } else {
        setIsLive(false);
        setState("error");
      }
    };
  }, [playSerializedAudio, activeVoiceId]);

  // Sync ref in effect so the reconnect timeout always calls the latest connect
  useEffect(() => {
    connectRef.current = connect;
  }, [connect]);

  // --- Mic capture ---
  const startStreaming = async () => {
    try {
      if (!audioContextRef.current) {
        // 24kHz for both modes: high-quality playback, Whisper handles input resampling
        audioContextRef.current = new AudioContext({ sampleRate: 24000 });
      }

      // Chrome creates the AudioContext in "suspended" state per autoplay policy.
      // Without an explicit resume, ScriptProcessorNode.onaudioprocess still fires
      // but inputBuffer samples are all zero → backend sees silent audio.
      if (audioContextRef.current.state === "suspended") {
        await audioContextRef.current.resume();
      }

      // Echo cancellation + noise suppression: critical for barge-in detection.
      // autoGainControl OFF — AGC was clipping mic audio (peak > 1.0 after resample),
      // which caused Whisper to hallucinate gibberish. Natural volume gives Whisper
      // clean mel-specs; its internal log-normalization handles quiet speech fine.
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: false }
      });
      sourceRef.current = audioContextRef.current.createMediaStreamSource(stream);

      processorRef.current = audioContextRef.current.createScriptProcessor(4096, 1, 1);

      // Diagnostic: warn (once) if the first ~1s of mic capture is all-zero, so
      // the user can immediately see mic is muted / suspended even though the
      // stream is live.
      let zeroStreak = 0;
      let diagWarned = false;

      processorRef.current.onaudioprocess = (e) => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          const inputData = e.inputBuffer.getChannelData(0);

          // Quick peak check for client-side diagnostics.
          let peak = 0;
          for (let i = 0; i < inputData.length; i++) {
            const a = Math.abs(inputData[i]);
            if (a > peak) peak = a;
          }
          if (!diagWarned) {
            if (peak === 0) {
              zeroStreak += 1;
              if (zeroStreak === 6) { // ~1s of pure silence
                console.warn("[Mic] 1s of pure-zero samples — check OS mic mute / browser permissions.");
                diagWarned = true;
              }
            } else {
              zeroStreak = 0;
            }
          }

          const pcmData = new Int16Array(inputData.length);
          for (let i = 0; i < inputData.length; i++) {
            pcmData[i] = Math.max(-1, Math.min(1, inputData[i])) * 0x7FFF;
          }
          wsRef.current.send(pcmData.buffer);

          if (!isCloud) {
            if (peak > 0.05) {
              setState("listening");
            }
          }
        }
      };

      sourceRef.current.connect(processorRef.current);
      processorRef.current.connect(audioContextRef.current.destination);

      if (!wsRef.current) connect(isCloud);
    } catch (err) {
      console.error("Mic error:", err);
      setState("error");
    }
  };

  const stopStreaming = () => {
    processorRef.current?.disconnect();
    sourceRef.current?.disconnect();
    wsRef.current?.close();
    wsRef.current = null;
    setIsLive(false);
    setState("idle");
  };

  const toggleLive = () => {
    if (isLive) stopStreaming();
    else startStreaming();
  };

  const toggleCloud = () => {
    if (isLive) stopStreaming();
    setIsCloud(!isCloud);
  };

  const switchVoice = (voiceId: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "switch_voice", voice_id: voiceId }));
    }
  };

  return { state, messages, isLive, isCloud, toggleLive, toggleCloud, switchVoice };
}
