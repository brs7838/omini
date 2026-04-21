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
  // [BARGE-IN FIX] TTS playback is routed through a MediaStreamDestinationNode →
  // hidden <audio> element. This makes the browser's echo canceller see the TTS
  // as standard playback (which it knows how to subtract from the mic) instead of
  // raw WebAudio destination output (which Chrome's AEC tends to over-suppress,
  // muting the entire mic input — that's why rms=0.000 was showing while AI spoke).
  const playbackDestRef = useRef<MediaStreamAudioDestinationNode | null>(null);
  const playbackAudioElRef = useRef<HTMLAudioElement | null>(null);
  // Silent sink for the ScriptProcessor (it must connect to *something* to run,
  // but routing it through a gain=0 node prevents a mic→speaker feedback loop).
  const micSinkRef = useRef<GainNode | null>(null);
  // Active <audio> elements playing current TTS blobs — tracked so barge-in /
  // interrupt can stop them immediately.
  const activePlaybackElsRef = useRef<HTMLAudioElement[]>([]);

  // Wrap int16 PCM samples (mono, little-endian) in a minimal RIFF/WAVE header
  // so the browser's native audio decoder interprets them at the stated rate.
  // This is the SAME path the TTS lab uses, which is why the lab's playback is
  // pitch-stable while our old createBuffer(...) path was not — Web Audio's
  // internal 24 k→48 k resampler (AudioContext runs at hardware rate) was
  // introducing audible pitch modulation on long samples; <audio> + blob URL
  // hands the rate conversion to the OS SRC which is clean.
  const pcmToWavBlob = (pcmBytes: ArrayBuffer, sampleRate: number): Blob => {
    const pcmLen = pcmBytes.byteLength;
    const headerLen = 44;
    const buf = new ArrayBuffer(headerLen + pcmLen);
    const view = new DataView(buf);
    const writeAscii = (offset: number, s: string) => {
      for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i));
    };
    writeAscii(0, "RIFF");
    view.setUint32(4, 36 + pcmLen, true);
    writeAscii(8, "WAVE");
    writeAscii(12, "fmt ");
    view.setUint32(16, 16, true);               // PCM fmt chunk size
    view.setUint16(20, 1, true);                // PCM format
    view.setUint16(22, 1, true);                // channels (mono)
    view.setUint32(24, sampleRate, true);       // sample rate
    view.setUint32(28, sampleRate * 2, true);   // byte rate (1ch * 2 bytes)
    view.setUint16(32, 2, true);                // block align
    view.setUint16(34, 16, true);               // bits per sample
    writeAscii(36, "data");
    view.setUint32(40, pcmLen, true);
    new Uint8Array(buf, headerLen).set(new Uint8Array(pcmBytes));
    return new Blob([buf], { type: "audio/wav" });
  };

  // Hard-stop all currently-playing TTS elements. `pause()` alone is not
  // always sufficient to halt an <audio> element that has an in-flight blob
  // decoder — detaching `src` + `load()` drops the media resource and frees
  // buffered samples so there's no audible tail after barge-in or a new reply.
  const stopAllPlayback = () => {
    activePlaybackElsRef.current.forEach(el => {
      try {
        el.pause();
        el.removeAttribute("src");
        el.load();
      } catch { /* element may already be detached */ }
    });
    activePlaybackElsRef.current = [];
  };

  // --- Audio playback (declared before connect so it can be referenced) ---
  const playSerializedAudio = useCallback(async (data: Blob, sampleRate: number) => {
    // Kill any previous reply's playback before starting this one. Without
    // this, a delayed "interrupted" event or a quick back-to-back turn let
    // the old <audio> element bleed into the first ~second of the new one —
    // audible as a brief voice overlap at the start of each reply.
    stopAllPlayback();

    const pcmBuffer = await data.arrayBuffer();
    const wavBlob = pcmToWavBlob(pcmBuffer, sampleRate);
    const url = URL.createObjectURL(wavBlob);

    const el = new Audio();
    el.src = url;
    el.autoplay = true;
    el.preload = "auto";
    // Explicit play() — autoplay alone isn't guaranteed to fire synchronously
    // on every browser/version, and we want playback to start the moment the
    // blob is ready so replies aren't perceived as laggy.
    el.play().catch(err => console.warn("[Audio] play() rejected:", err));

    activePlaybackElsRef.current.push(el);
    const cleanup = () => {
      URL.revokeObjectURL(url);
      activePlaybackElsRef.current = activePlaybackElsRef.current.filter(a => a !== el);
      // If nothing else is playing, drop back to idle so the UI mic reopens.
      if (activePlaybackElsRef.current.length === 0) {
        setState(prev => (prev === "speaking" ? "idle" : prev));
      }
    };
    el.onended = cleanup;
    el.onerror = cleanup;

    console.log(`[Audio] Playing WAV blob: ${(pcmBuffer.byteLength / 2 / sampleRate).toFixed(3)}s`);
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
              try { node.stop(); } catch { }
            });
            scheduledNodesRef.current = [];
            // Hard-stop the blob-URL <audio> elements (pause + detach + load)
            // so barge-in halts TTS instantly instead of letting the decoder
            // finish buffered samples.
            stopAllPlayback();
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

    ws.onclose = (ev) => {
      // Forward the close event to the backend so it appears in backend.log —
      // helps tell StrictMode remounts / ping timeouts / clean user stops apart.
      try {
        fetch("http://127.0.0.1:8000/debug/ws_close", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          keepalive: true,
          body: JSON.stringify({
            code: ev.code, reason: ev.reason, wasClean: ev.wasClean, ts: Date.now(),
          }),
        }).catch(() => {});
      } catch {}
      console.log(`[WS onclose] code=${ev.code} reason=${ev.reason} wasClean=${ev.wasClean}`);
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
        setState("idle"); 
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
        // Use native OS sample rate (48kHz on Windows, 44100 on Mac).
        // Forcing 24kHz here causes Chrome on Windows to feed all-zero samples
        // to createScriptProcessor because WASAPI captures at 48kHz and Chrome's
        // internal resampler has a known silent-buffer bug at non-native rates.
        // We downsample to 24kHz in JS before sending so the server still gets 24kHz.
        audioContextRef.current = new AudioContext();
      }

      // Echo cancellation + noise suppression: critical for barge-in detection.
      // autoGainControl OFF — AGC was clipping mic audio (peak > 1.0 after resample),
      // which caused Whisper to hallucinate gibberish. Natural volume gives Whisper
      // clean mel-specs; its internal log-normalization handles quiet speech fine.
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: false }
      });

      // Diagnostic: log each audio track's state so if mic is muted/disabled at the
      // OS or browser level, we see it immediately in the console.
      stream.getAudioTracks().forEach(t => {
        const msg = `[Mic] track: label=${t.label} enabled=${t.enabled} muted=${t.muted} readyState=${t.readyState}`;
        console.log(msg);
        // [FIX] Forward this to backend so we can see it in Omini logs
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          wsRef.current.send(JSON.stringify({ type: "log_forward", data: msg }));
        }
      });

      sourceRef.current = audioContextRef.current.createMediaStreamSource(stream);

      // [BARGE-IN FIX] Build the playback graph that the browser's AEC can
      // properly identify. TTS BufferSources connect to playbackDest; the
      // resulting MediaStream is played by a hidden <audio> element. This
      // route is what `echoCancellation: true` knows how to subtract — feeding
      // playback straight into AudioContext.destination caused Chrome to
      // hard-mute the mic during AI speech (rms=0.000 in logs).
      if (!playbackDestRef.current) {
        playbackDestRef.current = audioContextRef.current.createMediaStreamDestination();
      }
      if (!playbackAudioElRef.current) {
        const el = new Audio();
        el.autoplay = true;
        // Mark inaudible to AT but keep it in the DOM so the stream actually plays.
        el.setAttribute("aria-hidden", "true");
        el.style.display = "none";
        document.body.appendChild(el);
        el.srcObject = playbackDestRef.current.stream;
        // Some browsers need an explicit play() after srcObject is set.
        el.play().catch(err => console.warn("[Audio] playback element failed to start:", err));
        playbackAudioElRef.current = el;
      }

      processorRef.current = audioContextRef.current.createScriptProcessor(4096, 1, 1);

      // Diagnostic: warn (once) if the first ~1s of mic capture is all-zero, so
      // the user can immediately see mic is muted / suspended even though the
      // stream is live.
      let zeroStreak = 0;
      let diagWarned = false;

      processorRef.current.onaudioprocess = (e) => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          const inputData = e.inputBuffer.getChannelData(0);

          // Quick peak check for client-side diagnostics (on raw native-rate samples).
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

          // Downsample native rate (e.g. 48kHz on Windows) → 24kHz before sending.
          // The server's Silero VAD and Whisper both expect 24kHz input. Using linear
          // interpolation keeps speech quality for ASR without anti-aliasing overhead.
          const TARGET_RATE = 24000;
          const ctxRate = audioContextRef.current!.sampleRate;
          let samples: Float32Array;
          if (ctxRate === TARGET_RATE) {
            samples = inputData;
          } else {
            const ratio = ctxRate / TARGET_RATE;
            const outLen = Math.floor(inputData.length / ratio);
            samples = new Float32Array(outLen);
            for (let i = 0; i < outLen; i++) {
              const src = i * ratio;
              const lo = Math.floor(src);
              const hi = Math.min(lo + 1, inputData.length - 1);
              const frac = src - lo;
              samples[i] = inputData[lo] * (1 - frac) + inputData[hi] * frac;
            }
          }

          const pcmData = new Int16Array(samples.length);
          for (let i = 0; i < samples.length; i++) {
            pcmData[i] = Math.max(-1, Math.min(1, samples[i])) * 0x7FFF;
          }
          wsRef.current.send(pcmData.buffer);

          // Client-side "listening" fallback. Threshold lowered from 0.05 to 0.015
          // because autoGainControl is now OFF, so natural-volume speech peaks at
          // ~0.02-0.1 instead of AGC-boosted ~0.3-1.0. Backend Silero VAD also
          // emits an authoritative "listening" status; this local check just keeps
          // the orb feeling responsive on the first few ms of an utterance.
          if (!isCloud) {
            if (peak > 0.015) {
              setState("listening");
            }
          }
        }
      };

      sourceRef.current.connect(processorRef.current);
      // [BARGE-IN FIX] ScriptProcessor must connect somewhere to keep firing,
      // but we DO NOT want mic audio leaking back to the speakers (which can
      // confuse AEC and feed into the AI's own input). Route through a gain=0
      // node — silent at the speakers, processor still ticks.
      if (!micSinkRef.current) {
        micSinkRef.current = audioContextRef.current.createGain();
        micSinkRef.current.gain.value = 0.0;
        micSinkRef.current.connect(audioContextRef.current.destination);
      }
      processorRef.current.connect(micSinkRef.current);

      // Resume AFTER the graph is connected. Some Chrome versions leave the
      // AudioContext in a "running but hardware silent" state if resume() is called
      // before the source is wired, producing a steady stream of zero samples.
      if (audioContextRef.current.state !== "running") {
        await audioContextRef.current.resume();
      }
      
      const bootMsg = `[Mic] AudioContext state=${audioContextRef.current.state} rate=${audioContextRef.current.sampleRate}`;
      console.log(bootMsg);
      if (wsRef.current?.readyState === WebSocket.OPEN) {
          wsRef.current.send(JSON.stringify({ type: "log_forward", data: bootMsg }));
      }

      if (!wsRef.current) connect(isCloud);
    } catch (err) {
      console.error("Mic error:", err);
      setState("error");
    }
  };

  const stopStreaming = () => {
    if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
    reconnectAttemptsRef.current = 100; // Block auto-reconnect loop
    
    // Instant Silence
    scheduledNodesRef.current.forEach(node => {
      try { node.stop(); } catch { }
    });
    scheduledNodesRef.current = [];
    nextStartTimeRef.current = 0;

    // Tear down the playback <audio> element so a fresh one is built on next
    // session (otherwise an autoplay-suspended element survives the reconnect
    // and TTS plays inaudibly, which previously surfaced as "AI looks like it
    // spoke but I heard nothing").
    if (playbackAudioElRef.current) {
      try {
        playbackAudioElRef.current.pause();
        playbackAudioElRef.current.srcObject = null;
        playbackAudioElRef.current.remove();
      } catch { }
      playbackAudioElRef.current = null;
    }
    playbackDestRef.current = null;
    micSinkRef.current = null;

    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "interrupt" }));
      setTimeout(() => wsRef.current?.close(), 50); // Close after a tiny buffer to let interrupt pass
    } else {
      wsRef.current?.close();
    }
    
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

  const switchModel = (modelId: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "switch_model", model_id: modelId }));
    }
  };

  const resetChat = () => {
    setMessages([]);
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "reset_chat" }));
    }
  };

  return { state, messages, isLive, isCloud, toggleLive, toggleCloud, switchVoice, switchModel, resetChat };
}
