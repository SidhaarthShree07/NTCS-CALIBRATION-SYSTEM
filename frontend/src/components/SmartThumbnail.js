import React, { useEffect, useRef, useState } from 'react';
import Hls from 'hls.js';
import API_URL from '../config';

// SmartThumbnail supports:
// - MJPEG-like streams via <img>
// - HLS (.m3u8) via <video> + hls.js (or Safari native)
// - Snapshot URLs with periodic refresh
// - Fallback to placeholder when unsupported or failing
export default function SmartThumbnail({ src, alt = 'Live thumbnail', refreshMs = 2000, fallbackSrc = '/thumbnail-placeholder.svg' }) {
  const originalSrcRef = useRef(src);
  const videoRef = useRef(null);
  const hlsRef = useRef(null);
  const [status, setStatus] = useState('idle'); // idle | loading | loaded | error
  const [finalSrc, setFinalSrc] = useState('');
  const timeoutRef = useRef(null);

  const lower = (src || '').toLowerCase();
  const isRTSP = lower.startsWith('rtsp://') || lower.startsWith('rtmp://');
  const isHls = lower.includes('.m3u8');
  // More precise MJPEG detection: only if explicitly has mjpeg/mjpg in path or common MJPEG endpoints
  const isMjpeg = (lower.includes('mjpg') || lower.includes('mjpeg') || lower.includes('/video_feed') || lower.includes('/stream.jpg')) && !isHls;
  // Detect if it's a backend stream (cached HTTP stream rendered via backend)
  const isBackendStream = lower.startsWith('http') && !isHls && !isMjpeg;

  // Reset when src changes
  useEffect(() => {
    originalSrcRef.current = src;
    setStatus('idle');
    setFinalSrc('');
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }

    if (!src) {
      console.warn('[SmartThumbnail] No src provided');
      setStatus('error');
      return;
    }

    console.log('[SmartThumbnail] Processing:', src, { isRTSP, isHls, isMjpeg });

    if (isRTSP) {
      console.warn('[SmartThumbnail] RTSP not supported, showing fallback');
      setStatus('error');
      return;
    }

    if (isHls) {
      setStatus('loading');
      
      // Cleanup previous HLS instance
      if (hlsRef.current) {
        try {
          hlsRef.current.destroy();
        } catch (e) {
          console.warn('[SmartThumbnail] Error destroying previous HLS:', e);
        }
        hlsRef.current = null;
      }

      // Delay HLS setup slightly to ensure video ref is mounted
      const timeoutId = setTimeout(() => {
        const video = videoRef.current;
        if (!video) {
          console.error('[SmartThumbnail] Video ref not available for HLS after delay');
          setStatus('error');
          return;
        }

        const canNative = video.canPlayType && video.canPlayType('application/vnd.apple.mpegurl');
        const onPlaying = () => {
          console.log('[SmartThumbnail] HLS playing');
          setStatus('loaded');
        };
        const onError = (e) => {
          console.error('[SmartThumbnail] HLS video error:', e);
          setStatus('error');
        };

        video.addEventListener('playing', onPlaying);
        video.addEventListener('error', onError);

        if (canNative) {
          console.log('[SmartThumbnail] Using native HLS playback');
          video.src = src;
          video.load();
        } else if (Hls.isSupported()) {
          console.log('[SmartThumbnail] Using HLS.js');
          const hls = new Hls({ 
            enableWorker: true, 
            lowLatencyMode: true,
            debug: false
          });
          hlsRef.current = hls;

          hls.on(Hls.Events.ERROR, (event, data) => {
            console.error('[SmartThumbnail] HLS.js error:', data.type, data.details);
            if (data.fatal) {
              setStatus('error');
            }
          });

          hls.on(Hls.Events.MANIFEST_PARSED, () => {
            console.log('[SmartThumbnail] HLS manifest parsed');
          });

          hls.attachMedia(video);
          hls.loadSource(src);
        } else {
          console.error('[SmartThumbnail] HLS not supported in this browser');
          setStatus('error');
          return;
        }

        video.muted = true;
        video.autoplay = true;
        video.loop = true;
        video.playsInline = true;

        // Attempt to play
        const playAttempt = video.play();
        if (playAttempt && typeof playAttempt.catch === 'function') {
          playAttempt.catch((e) => {
            console.warn('[SmartThumbnail] Autoplay failed:', e.message);
          });
        }
      }, 100);

      // Capture video ref at effect scope for cleanup
      const videoElement = videoRef.current;

      return () => {
        clearTimeout(timeoutId);
        if (videoElement) {
          videoElement.removeEventListener('playing', () => {});
          videoElement.removeEventListener('error', () => {});
        }
        if (hlsRef.current) {
          try {
            hlsRef.current.destroy();
          } catch (e) {
            console.warn('[SmartThumbnail] Cleanup error:', e);
          }
          hlsRef.current = null;
        }
      };
    }

    if (isMjpeg || isBackendStream) {
      // For backend streams (cached HTTP videos), use video element
      if (isBackendStream) {
        console.log('[SmartThumbnail] Using backend video stream for cached video');
        setStatus('loading');
        
        // Capture video element reference at the start of effect
        const videoElement = videoRef.current;
        let retryTimeout = null;
        
        // Use video element to play cached video
        const timeoutId = setTimeout(() => {
          if (!videoElement) {
            console.error('[SmartThumbnail] Video ref not available for backend stream');
            setStatus('error');
            return;
          }

          const videoUrl = `${API_URL}/api/video_stream?source=${encodeURIComponent(src)}`;
          console.log('[SmartThumbnail] Loading backend video:', videoUrl);
          
          const onPlaying = () => {
            console.log('[SmartThumbnail] Backend video playing');
            setStatus('loaded');
            if (retryTimeout) {
              clearTimeout(retryTimeout);
              retryTimeout = null;
            }
          };
          const onError = (e) => {
            console.warn('[SmartThumbnail] Backend video error, might still be downloading');
            // Video might still be downloading, retry after 2 seconds
            retryTimeout = setTimeout(() => {
              console.log('[SmartThumbnail] Retrying video load...');
              if (videoElement) {
                videoElement.load();
                videoElement.play().catch(() => {
                  console.warn('[SmartThumbnail] Retry play failed, will retry again');
                  // Keep showing loading state and retry will happen via useEffect re-trigger
                });
              }
            }, 2000);
          };

          videoElement.addEventListener('playing', onPlaying);
          videoElement.addEventListener('error', onError);
          videoElement.src = videoUrl;
          videoElement.muted = true;
          videoElement.autoplay = true;
          videoElement.loop = true;
          videoElement.playsInline = true;
          videoElement.load();

          const playAttempt = videoElement.play();
          if (playAttempt && typeof playAttempt.catch === 'function') {
            playAttempt.catch((e) => {
              console.warn('[SmartThumbnail] Backend video autoplay failed:', e.message);
              // Don't set error state immediately, let onError handler retry
            });
          }
        }, 100);

        return () => {
          clearTimeout(timeoutId);
          if (retryTimeout) {
            clearTimeout(retryTimeout);
          }
          // Use captured videoElement reference from effect scope
          if (videoElement) {
            videoElement.removeEventListener('playing', () => {});
            videoElement.removeEventListener('error', () => {});
            videoElement.pause();
            videoElement.src = '';
          }
        };
      }
      
      // Render <img> directly for MJPEG
      console.log('[SmartThumbnail] Using MJPEG image stream');
      setFinalSrc(src);
      setStatus('loaded');
      return;
    }

    // Snapshot flow: preload then display
    console.log('[SmartThumbnail] Using snapshot mode with refresh');
    setStatus('loading');
    const candidate = `${src}${src.includes('?') ? '&' : '?'}_t=${Date.now()}`;
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.src = candidate;
    const onLoad = () => {
      if (timeoutRef.current) { clearTimeout(timeoutRef.current); timeoutRef.current = null; }
      console.log('[SmartThumbnail] Snapshot loaded');
      setFinalSrc(candidate);
      setStatus('loaded');
    };
    const onError = () => {
      if (timeoutRef.current) { clearTimeout(timeoutRef.current); timeoutRef.current = null; }
      console.error('[SmartThumbnail] Snapshot failed, using fallback');
      if (fallbackSrc) {
        setFinalSrc(fallbackSrc);
        setStatus('loaded');
      } else {
        setStatus('error');
      }
    };
    img.onload = onLoad;
    img.onerror = onError;
    timeoutRef.current = setTimeout(() => { 
      img.onload = null; 
      img.onerror = null; 
      console.warn('[SmartThumbnail] Snapshot timeout after 5s');
      onError(); 
    }, 5000);

    return () => {
      img.onload = null;
      img.onerror = null;
      if (timeoutRef.current) { clearTimeout(timeoutRef.current); timeoutRef.current = null; }
    };
  }, [src, isRTSP, isHls, isMjpeg, isBackendStream, fallbackSrc]);

  // Periodic refresh for snapshot URLs only (not for videos or streams)
  useEffect(() => {
    if (!finalSrc) return;
    if (isMjpeg || isHls || isBackendStream) return; // These are all handled as continuous streams/videos
    
    const id = setInterval(() => {
      const s = originalSrcRef.current || src;
      // Refresh snapshot URL
      const updated = `${s}${s.includes('?') ? '&' : '?'}_t=${Date.now()}`;
      setFinalSrc(updated);
    }, refreshMs);
    return () => clearInterval(id);
  }, [finalSrc, isMjpeg, isHls, isBackendStream, refreshMs, src]);

  // Render
  if (isHls || isBackendStream) {
    // Render video element for HLS and backend streams so ref is available
    return (
      <div style={{ position: 'relative', width: '100%', height: '100%' }}>
        <video
          ref={videoRef}
          className="thumb-img"
          style={{ width: '100%', height: '100%', objectFit: 'cover' }}
          muted
          autoPlay
          loop
          playsInline
        />
        {status === 'loading' && (
          <div className="thumb-loader" style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: '#101215'
          }}>
            <svg width="40" height="40" viewBox="0 0 50 50" xmlns="http://www.w3.org/2000/svg">
              <circle cx="25" cy="25" r="20" stroke="#2b78f6" strokeWidth="5" fill="none" strokeLinecap="round">
                <animateTransform attributeName="transform" type="rotate" from="0 25 25" to="360 25 25" dur="1s" repeatCount="indefinite" />
              </circle>
            </svg>
          </div>
        )}
        {status === 'error' && (
          <img src={fallbackSrc} alt="No preview" className="thumb-img" style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            objectFit: 'cover'
          }} />
        )}
      </div>
    );
  }

  if (status === 'loading' || status === 'idle') {
    return (
      <div className="thumb-loader" style={{display:'flex',alignItems:'center',justifyContent:'center',width:'100%',height:'100%',background:'#101215'}}>
        <svg width="40" height="40" viewBox="0 0 50 50" xmlns="http://www.w3.org/2000/svg">
          <circle cx="25" cy="25" r="20" stroke="#2b78f6" strokeWidth="5" fill="none" strokeLinecap="round">
            <animateTransform attributeName="transform" type="rotate" from="0 25 25" to="360 25 25" dur="1s" repeatCount="indefinite" />
          </circle>
        </svg>
      </div>
    );
  }

  if (status === 'error') {
    return <img src={fallbackSrc} alt="No preview" className="thumb-img" />;
  }

  return (
    <img
      src={finalSrc}
      alt={alt}
      className="thumb-img"
      loading="lazy"
      onError={() => setStatus('error')}
      onLoad={() => setStatus('loaded')}
    />
  );
}
