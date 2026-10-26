/**
 * Replay client that streams real window data.
 * Tries local WebSocket first; if running on cloud/Vercel (where WS port 8765 is unavailable),
 * seamlessly streams real precomputed windows locally every 800ms.
 */

// Dynamically import all 27 precomputed replay json files
const replayModules = import.meta.glob('./replay/*.json', { eager: true });

const activeSimulations = {};

export function startReplayStream(deviceName, threshold, onWindow, onError) {
  const hostname = window.location.hostname || '';
  const isLocal = hostname === 'localhost' || hostname === '127.0.0.1' || hostname.startsWith('172.') || hostname.startsWith('192.');

  if (isLocal) {
    try {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${protocol}//${hostname}:8765/stream/replay/${deviceName}?threshold=${threshold}`;
      const ws = new WebSocket(wsUrl);

      let opened = false;
      let hasError = false;
      let fallbackHandle = null;

      // If WS doesn't open within 600ms (e.g. backend not running on port 8765), fallback to client replay
      const connectTimeout = setTimeout(() => {
        if (!opened && !hasError) {
          try { ws.close(); } catch (e) {}
          fallbackHandle = fallbackClientStream(deviceName, threshold, onWindow);
        }
      }, 600);

      ws.onopen = () => {
        opened = true;
        clearTimeout(connectTimeout);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'window') {
            onWindow(data);
          }
        } catch (e) {
          console.error(e);
        }
      };

      ws.onerror = (err) => {
        hasError = true;
        clearTimeout(connectTimeout);
        if (!opened) {
          // Switch to client simulation immediately
          fallbackHandle = fallbackClientStream(deviceName, threshold, onWindow);
        } else if (onError) {
          onError(err);
        }
      };

      ws.onclose = () => {
        clearTimeout(connectTimeout);
      };

      return {
        stop: () => {
          clearTimeout(connectTimeout);
          try { ws.close(); } catch (e) {}
          if (fallbackHandle) fallbackHandle.stop();
          stopFallback(deviceName);
        },
        setThreshold: (newThresh) => {
          if (fallbackHandle) {
            fallbackHandle.setThreshold(newThresh);
          } else if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ action: 'set_threshold', threshold: newThresh }));
          }
        }
      };
    } catch (err) {
      console.warn('WS connection failed, using client replay:', err);
    }
  }

  // Cloud / Vercel fallback
  return fallbackClientStream(deviceName, threshold, onWindow);
}

function fallbackClientStream(deviceName, threshold, onWindow) {
  stopFallback(deviceName);

  // Look for matching replay module (case-insensitive filename match)
  let targetData = null;
  const targetLower = deviceName.toLowerCase();
  for (const path in replayModules) {
    const filename = path.split('/').pop().replace('.json', '').toLowerCase();
    if (filename === targetLower) {
      targetData = replayModules[path].default || replayModules[path];
      break;
    }
  }

  if (!targetData || targetData.length === 0) {
    console.warn(`No offline replay data found for ${deviceName}`);
    return { stop: () => {}, setThreshold: () => {} };
  }

  let index = 0;
  let totalCount = 0;
  let matches = 0;
  let flaggedUnknowns = 0;
  let currentThresh = threshold;

  const dispatchNext = () => {
    const win = targetData[index % targetData.length];
    index++;
    totalCount++;

    const topProb = win.top_prob || 0.5;
    const isKnown = topProb >= currentThresh;
    const verdict = isKnown ? 'KNOWN' : 'UNKNOWN';
    const trueLabel = win.true_label || deviceName;

    if (isKnown && win.closest_profile === trueLabel) {
      matches++;
    }
    if (!isKnown) {
      flaggedUnknowns++;
    }

    const payload = {
      type: 'window',
      device_name: deviceName,
      window_index: win.window_index || totalCount,
      iats: win.iats || [],
      features: win.features || win.scaled_features || {},
      verdict,
      label: isKnown ? win.closest_profile : `Unfamiliar Device (Closest: ${win.closest_profile})`,
      closest_profile: win.closest_profile,
      top_prob: topProb,
      probs: win.probs || {},
      threshold: currentThresh,
      running_stats: {
        total_windows: totalCount,
        accuracy_pct: Math.round((matches / totalCount) * 1000) / 10,
        flagged_unknown_pct: Math.round((flaggedUnknowns / totalCount) * 1000) / 10,
      }
    };

    onWindow(payload);
  };

  // Dispatch first window immediately for instant feedback
  dispatchNext();

  const timer = setInterval(dispatchNext, 800);
  activeSimulations[deviceName] = timer;

  return {
    stop: () => stopFallback(deviceName),
    setThreshold: (t) => { currentThresh = t; }
  };
}

function stopFallback(deviceName) {
  if (activeSimulations[deviceName]) {
    clearInterval(activeSimulations[deviceName]);
    delete activeSimulations[deviceName];
  }
}
