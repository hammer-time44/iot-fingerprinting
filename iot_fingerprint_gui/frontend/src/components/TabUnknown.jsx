import React, { useState, useEffect, useRef } from 'react';
import { Search, Play, Square, AlertCircle, Sliders, Filter } from 'lucide-react';
import ProbabilityBars from './ProbabilityBars';
import IatTimelineChart from './IatTimelineChart';
import FeaturesTable from './FeaturesTable';

export default function TabUnknown({ threshold = 0.84, setThreshold }) {
  const [evalData, setEvalData] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [activeStreams, setActiveStreams] = useState({});
  const [streamData, setStreamData] = useState({});
  const socketsRef = useRef({});

  // Fetch full evaluation data from live backend
  useEffect(() => {
    fetch('/api/evaluation')
      .then((res) => res.json())
      .then((data) => setEvalData(data))
      .catch((err) => console.error('Error fetching evaluation data from backend:', err));
  }, []);

  // Cleanup WebSockets on unmount
  useEffect(() => {
    return () => {
      Object.values(socketsRef.current).forEach((ws) => ws && ws.close());
    };
  }, []);

  // Update running sockets when threshold changes
  useEffect(() => {
    Object.values(socketsRef.current).forEach((ws) => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ action: 'set_threshold', threshold }));
      }
    });
  }, [threshold]);

  const toggleStream = (devName) => {
    const isRunning = activeStreams[devName];

    if (isRunning) {
      if (socketsRef.current[devName]) {
        socketsRef.current[devName].close();
        delete socketsRef.current[devName];
      }
      setActiveStreams((prev) => ({ ...prev, [devName]: false }));
    } else {
      setActiveStreams((prev) => ({ ...prev, [devName]: true }));
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const host = window.location.hostname || '127.0.0.1';
      const wsUrl = `${protocol}//${host}:8765/stream/replay/${devName}?threshold=${threshold}`;

      try {
        const ws = new WebSocket(wsUrl);
        socketsRef.current[devName] = ws;

        ws.onmessage = (event) => {
          const data = JSON.parse(event.data);
          if (data.type === 'window') {
            setStreamData((prev) => ({
              ...prev,
              [devName]: data,
            }));
          }
        };

        ws.onerror = (err) => {
          console.error(`WebSocket error for ${devName}:`, err);
          setActiveStreams((prev) => ({ ...prev, [devName]: false }));
        };

        ws.onclose = () => {
          setActiveStreams((prev) => ({ ...prev, [devName]: false }));
        };
      } catch (err) {
        console.error('Failed to open WebSocket:', err);
        setActiveStreams((prev) => ({ ...prev, [devName]: false }));
      }
    }
  };

  // Filter 22 unknown devices (type == 'unknown' or 'unknown_offline')
  const unknownDevices = (evalData?.devices || []).filter(
    (d) => d.type === 'unknown' || d.type === 'unknown_offline'
  );

  const filteredDevices = unknownDevices.filter((d) =>
    d.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    d.category.toLowerCase().includes(searchQuery.toLowerCase())
  );

  // Approximate false alarm and detection estimates based on threshold slider
  // Calibrated baseline: 0.84 -> 3.94% FA, 67.58% Detection
  const dynamicFaEstimate = Math.max(0.2, (1.0 - threshold) * 24.6).toFixed(1);
  const dynamicDetEstimate = Math.min(99.0, Math.max(20.0, 67.58 + (threshold - 0.84) * 85.0)).toFixed(1);

  return (
    <div className="space-y-8">
      {/* Header & Threshold Controls */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between pb-4 border-b border-gray-200 dark:border-gray-800 gap-4">
        <div>
          <h2 className="text-lg font-bold text-gray-900 dark:text-white">
            Unseen Hardware: 22 Unknown Devices
          </h2>
          <p className="text-xs text-gray-500 dark:text-gray-400">
            Devices never included in training. Replay traffic through the same pipeline to observe closed-set model responses.
          </p>
        </div>

        {/* Dynamic Threshold Slider */}
        <div className="flex items-center space-x-4 bg-gray-50 dark:bg-gray-800/80 p-3 rounded-lg border border-gray-200 dark:border-gray-700">
          <Sliders className="w-4 h-4 text-slate-600 dark:text-slate-400" />
          <div className="space-y-1">
            <div className="flex justify-between text-xs font-medium text-gray-700 dark:text-gray-300">
              <span>Confidence Threshold: <strong className="font-mono">{threshold.toFixed(2)}</strong></span>
              <span className="text-[10px] text-gray-500 font-normal">(Calibrated: 0.84)</span>
            </div>
            <input
              type="range"
              min="0.50"
              max="0.98"
              step="0.01"
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              className="w-48 h-1.5 bg-gray-300 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer accent-indigo-600"
            />
          </div>
          <div className="text-[11px] font-mono border-l border-gray-200 dark:border-gray-700 pl-3 text-gray-600 dark:text-gray-400">
            <div>Unknown Detected: <strong className="text-indigo-600 dark:text-indigo-400">~{dynamicDetEstimate}%</strong></div>
            <div>Known False Alarms: <strong>~{dynamicFaEstimate}%</strong></div>
          </div>
        </div>
      </div>

      {/* Search Bar */}
      <div className="flex items-center space-x-3">
        <div className="relative flex-1">
          <Search className="w-4 h-4 absolute left-3 top-2.5 text-gray-400" />
          <input
            type="text"
            placeholder="Search 22 unknown devices by name or category (e.g. Camera, Bulb, Scale, Sensor)..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-9 pr-4 py-2 text-xs rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 text-gray-900 dark:text-white focus:outline-none focus:ring-1 focus:ring-indigo-500"
          />
        </div>
        <div className="text-xs text-gray-500 dark:text-gray-400">
          Showing {filteredDevices.length} of {unknownDevices.length} devices
        </div>
      </div>

      {/* Streamable Device Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
        {filteredDevices.slice(0, 9).map((dev) => {
          const isRunning = !!activeStreams[dev.name];
          const isOffline = dev.type === 'unknown_offline';
          const data = streamData[dev.name];

          return (
            <div
              key={dev.name}
              className={`rounded-lg border transition-all ${
                isRunning
                  ? 'border-indigo-400 dark:border-indigo-600 shadow-sm bg-white dark:bg-gray-800/95'
                  : 'border-gray-200 dark:border-gray-800 bg-gray-50/50 dark:bg-gray-900/40'
              } p-4 flex flex-col justify-between`}
            >
              <div>
                {/* Header & Toggle */}
                <div className="flex justify-between items-start mb-3">
                  <div>
                    <h3 className="font-semibold text-sm text-gray-900 dark:text-white">{dev.name}</h3>
                    <p className="text-[11px] text-gray-500 dark:text-gray-400">{dev.category}</p>
                  </div>
                  {isOffline ? (
                    <span className="text-[10px] font-mono px-2 py-0.5 bg-gray-200 dark:bg-gray-800 text-gray-500 rounded">
                      Offline in trace
                    </span>
                  ) : (
                    <button
                      onClick={() => toggleStream(dev.name)}
                      className={`flex items-center space-x-1 px-2.5 py-1 text-xs font-medium rounded transition-colors ${
                        isRunning
                          ? 'bg-rose-100 text-rose-800 dark:bg-rose-950/60 dark:text-rose-300 hover:bg-rose-200'
                          : 'bg-indigo-600 text-white hover:bg-indigo-700 shadow-sm'
                      }`}
                    >
                      {isRunning ? (
                        <>
                          <Square className="w-3 h-3 fill-current" />
                          <span>Stop</span>
                        </>
                      ) : (
                        <>
                          <Play className="w-3 h-3 fill-current" />
                          <span>Replay</span>
                        </>
                      )}
                    </button>
                  )}
                </div>

                {/* Verdict Badge */}
                {data ? (
                  <div className="mb-3 p-2 bg-gray-100 dark:bg-gray-900 rounded border border-gray-200 dark:border-gray-800 flex justify-between items-center text-xs">
                    <div>
                      <span className={`font-semibold ${data.verdict === 'UNKNOWN' ? 'text-amber-600 dark:text-amber-400' : 'text-rose-600 dark:text-rose-400'}`}>
                        {data.verdict === 'UNKNOWN'
                          ? `UNKNOWN: no strong match (closest: ${data.closest_profile}, ${(data.top_prob * 100).toFixed(1)}%)`
                          : `KNOWN (MISCLASSIFICATION): ${data.label} (${(data.top_prob * 100).toFixed(1)}%)`}
                      </span>
                      <div className="text-[10px] text-gray-500">
                        Top Confidence: {(data.top_prob * 100).toFixed(1)}% &middot; Window #{data.window_index}
                      </div>
                    </div>
                    {data.running_stats && (
                      <span className="font-mono text-[11px] px-2 py-0.5 bg-slate-200 dark:bg-slate-800 rounded">
                        Flagged: {data.running_stats.flagged_unknown_pct}%
                      </span>
                    )}
                  </div>
                ) : (
                  <div className="mb-3 p-2 rounded border border-dashed border-gray-200 dark:border-gray-800 text-[11px] text-gray-400 text-center">
                    {isOffline ? 'No packet traces recorded for this hardware.' : 'Stream paused. Toggle Replay to feed traffic.'}
                  </div>
                )}

                {/* Probability Distribution */}
                {!isOffline && (
                  <>
                    <div className="mb-3">
                      <div className="text-[11px] font-medium text-gray-700 dark:text-gray-300 mb-1.5 flex items-center justify-between">
                        <span>5-Class Probability Distribution</span>
                      </div>
                      <ProbabilityBars probs={data?.probs} dominant={data?.closest_profile} />
                    </div>

                    <div className="mb-3">
                      <IatTimelineChart iats={data?.iats} height={60} />
                    </div>

                    <div>
                      <div className="text-[11px] font-medium text-gray-700 dark:text-gray-300 mb-1">
                        Latest Window Features (13 moments)
                      </div>
                      <FeaturesTable features={data?.features} />
                    </div>
                  </>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Summary Table Across All 22 Unknown Devices */}
      <div className="border border-gray-200 dark:border-gray-800 rounded-lg bg-white dark:bg-gray-900 p-5 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-gray-200 dark:border-gray-800 pb-3">
          <div>
            <h3 className="font-bold text-sm text-gray-900 dark:text-white">
              Full Evaluation Matrix: All 22 Unknown Devices
            </h3>
            <p className="text-xs text-gray-500 dark:text-gray-400">
              Evaluated under the fixed 5-class Random Forest with validation threshold ({threshold.toFixed(2)}).
            </p>
          </div>
          <span className="text-xs font-mono px-2.5 py-1 bg-amber-100 dark:bg-amber-950/60 text-amber-800 dark:text-amber-300 rounded border border-amber-300 dark:border-amber-800">
            Overall Detection Rate: <strong>67.58%</strong> (3,370 / 4,987 windows)
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="bg-gray-100 dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 text-gray-700 dark:text-gray-300">
                <th className="p-2 font-medium">Device Name</th>
                <th className="p-2 font-medium">Category</th>
                <th className="p-2 font-medium text-right">Windows</th>
                <th className="p-2 font-medium text-right">% Flagged UNKNOWN</th>
                <th className="p-2 font-medium">Dominant Mislabeled Profile</th>
                <th className="p-2 font-medium text-right">Avg Confidence</th>
                <th className="p-2 font-medium">Empirical Behavior & Caveats</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100 dark:divide-gray-800 text-[11px]">
              {unknownDevices.map((d) => {
                const isFail = d.is_confident_failure;
                return (
                  <tr
                    key={d.name}
                    className={`hover:bg-gray-50 dark:hover:bg-gray-800/40 ${
                      isFail ? 'bg-rose-50/40 dark:bg-rose-950/20' : ''
                    }`}
                  >
                    <td className="p-2 font-semibold text-gray-900 dark:text-white">{d.name}</td>
                    <td className="p-2 text-gray-500">{d.category}</td>
                    <td className="p-2 text-right font-mono">{d.windows_tested}</td>
                    <td className="p-2 text-right font-mono font-bold">
                      {d.windows_tested > 0 ? (
                        <span className={d.flagged_unknown_pct >= 90 ? 'text-emerald-600 dark:text-emerald-400' : isFail ? 'text-rose-600 dark:text-rose-400' : 'text-amber-600'}>
                          {d.flagged_unknown_pct}%
                        </span>
                      ) : (
                        <span className="text-gray-400">0%</span>
                      )}
                    </td>
                    <td className="p-2 font-mono text-gray-700 dark:text-gray-300">
                      {d.dominant_label !== 'N/A' ? `${d.dominant_label} (${d.dominant_share_pct}%)` : '-'}
                    </td>
                    <td className="p-2 text-right font-mono">
                      {d.avg_confidence > 0 ? d.avg_confidence.toFixed(4) : '-'}
                    </td>
                    <td className="p-2 text-gray-600 dark:text-gray-400">
                      {isFail ? (
                        <span className="inline-flex items-center text-rose-700 dark:text-rose-300 font-medium">
                          <AlertCircle className="w-3 h-3 mr-1 flex-shrink-0" />
                          {d.notes}
                        </span>
                      ) : (
                        d.notes
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
