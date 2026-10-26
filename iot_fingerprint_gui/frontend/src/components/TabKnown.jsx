import React, { useState, useEffect, useRef } from 'react';
import { Play, Square, Activity, CheckCircle, BarChart3, Database } from 'lucide-react';
import ProbabilityBars from './ProbabilityBars';
import IatTimelineChart from './IatTimelineChart';
import FeaturesTable from './FeaturesTable';

const KNOWN_DEVICES = [
  { id: 'AmazonEcho', name: 'Amazon Echo', category: 'Smart Voice Speaker', mac: '44:65:0d:56:cc:d3' },
  { id: 'BelkinWemoMotionSensor', name: 'Belkin WeMo Motion', category: 'PIR Motion Sensor', mac: 'ec:1a:59:83:28:11' },
  { id: 'PhilipsHue', name: 'Philips Hue Bridge', category: 'ZigBee Gateway', mac: '00:17:88:24:76:ff' },
  { id: 'SamsungCamera', name: 'Samsung SmartCam', category: 'IP Security Camera', mac: '00:16:6c:ab:6b:88' },
  { id: 'TPLinkSmartPlug', name: 'TP-Link Smart Plug', category: 'Power Switch', mac: '50:c7:bf:00:56:39' },
];

export default function TabKnown({ threshold = 0.84 }) {
  const [activeStreams, setActiveStreams] = useState({});
  const [streamData, setStreamData] = useState({});
  const [modelInfo, setModelInfo] = useState(null);
  const socketsRef = useRef({});

  // Fetch model info & global metrics
  useEffect(() => {
    fetch('/api/model/info')
      .then((res) => res.json())
      .then((data) => setModelInfo(data))
      .catch((err) => console.error('Error fetching model info:', err));
  }, []);

  // Cleanup WebSockets on unmount
  useEffect(() => {
    return () => {
      Object.values(socketsRef.current).forEach((ws) => ws && ws.close());
    };
  }, []);

  const toggleStream = (devId) => {
    const isRunning = activeStreams[devId];

    if (isRunning) {
      if (socketsRef.current[devId]) {
        socketsRef.current[devId].close();
        delete socketsRef.current[devId];
      }
      setActiveStreams((prev) => ({ ...prev, [devId]: false }));
    } else {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const host = window.location.hostname || '127.0.0.1';
      const wsUrl = `${protocol}//${host}:8765/stream/replay/${devId}?threshold=${threshold}`;

      try {
        const ws = new WebSocket(wsUrl);
        socketsRef.current[devId] = ws;

        ws.onopen = () => {
          setActiveStreams((prev) => ({ ...prev, [devId]: true }));
        };

        ws.onmessage = (event) => {
          const data = JSON.parse(event.data);
          if (data.type === 'window') {
            setStreamData((prev) => ({
              ...prev,
              [devId]: data,
            }));
          }
        };

        ws.onerror = (err) => {
          console.error(`WebSocket error for ${devId}:`, err);
          setActiveStreams((prev) => ({ ...prev, [devId]: false }));
        };

        ws.onclose = () => {
          setActiveStreams((prev) => ({ ...prev, [devId]: false }));
        };
      } catch (err) {
        console.error('Failed to open WebSocket:', err);
      }
    }
  };

  return (
    <div className="space-y-8">
      {/* Intro Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-gray-200 dark:border-gray-800 gap-2">
        <div>
          <h2 className="text-lg font-bold text-gray-900 dark:text-white">
            Trained Inventory: 5 Known Devices
          </h2>
          <p className="text-xs text-gray-500 dark:text-gray-400">
            Replay held-out test windows through the live Random Forest model. Toggle switches to stream packet sequences.
          </p>
        </div>
        <div className="text-xs font-mono px-3 py-1 bg-slate-100 dark:bg-slate-800 rounded border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300">
          Held-out Test Accuracy: <strong>99.34%</strong> &middot; Test False Alarms: <strong>3.94%</strong>
        </div>
      </div>

      {/* Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
        {KNOWN_DEVICES.map((dev) => {
          const isRunning = !!activeStreams[dev.id];
          const data = streamData[dev.id];

          return (
            <div
              key={dev.id}
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
                    <p className="text-[11px] text-gray-500 dark:text-gray-400">
                      {dev.category} &middot; <span className="font-mono">{dev.mac}</span>
                    </p>
                  </div>
                  <button
                    onClick={() => toggleStream(dev.id)}
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
                </div>

                {/* Verdict Badge */}
                {data ? (
                  <div className="mb-3 p-2 bg-gray-100 dark:bg-gray-900 rounded border border-gray-200 dark:border-gray-800 flex justify-between items-center text-xs">
                    <div>
                      <span className="font-semibold text-gray-900 dark:text-white">
                        {data.verdict === 'KNOWN' ? `KNOWN: ${data.label}` : 'LOW CONFIDENCE'}
                      </span>
                      <div className="text-[10px] text-gray-500">
                        Top Confidence: {(data.top_prob * 100).toFixed(1)}% &middot; Window #{data.window_index}
                      </div>
                    </div>
                    {data.running_stats && (
                      <span className="font-mono text-[11px] px-2 py-0.5 bg-slate-200 dark:bg-slate-800 rounded">
                        Acc: {data.running_stats.accuracy_pct}%
                      </span>
                    )}
                  </div>
                ) : (
                  <div className="mb-3 p-2 rounded border border-dashed border-gray-200 dark:border-gray-800 text-[11px] text-gray-400 text-center">
                    Stream paused. Click Replay to feed held-out test windows.
                  </div>
                )}

                {/* Probability Distribution */}
                <div className="mb-3">
                  <div className="text-[11px] font-medium text-gray-700 dark:text-gray-300 mb-1.5 flex items-center justify-between">
                    <span>5-Class Probability Distribution</span>
                  </div>
                  <ProbabilityBars probs={data?.probs} dominant={data?.closest_profile} />
                </div>

                {/* IAT Timeline Chart */}
                <div className="mb-3">
                  <IatTimelineChart iats={data?.iats} height={60} />
                </div>

                {/* 13 Features Table */}
                <div>
                  <div className="text-[11px] font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Latest Window Features (13 moments)
                  </div>
                  <FeaturesTable features={data?.features} />
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Global Results & Model Insights */}
      {modelInfo && (
        <div className="border border-gray-200 dark:border-gray-800 rounded-lg bg-white dark:bg-gray-900 p-5 space-y-6">
          <div className="flex items-center space-x-2 text-sm font-bold text-gray-900 dark:text-white border-b border-gray-200 dark:border-gray-800 pb-3">
            <BarChart3 className="w-4 h-4 text-slate-600" />
            <span>Benchmark Model Calibration & Feature Importance (Trained on 5 Known Classes)</span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-xs">
            {/* Confusion Matrix */}
            <div>
              <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-2">
                Held-Out Test Set Confusion Matrix (1,067 Windows)
              </h4>
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse border border-gray-200 dark:border-gray-700 text-[11px]">
                  <thead>
                    <tr className="bg-gray-100 dark:bg-gray-800">
                      <th className="p-1.5 border border-gray-200 dark:border-gray-700">True \ Pred</th>
                      {modelInfo.confusion_matrix?.classes?.map((c) => (
                        <th key={c} className="p-1.5 border border-gray-200 dark:border-gray-700 font-mono">
                          {c.substring(0, 6)}..
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {modelInfo.confusion_matrix?.matrix?.map((row, rIdx) => {
                      const cName = modelInfo.confusion_matrix.classes[rIdx];
                      return (
                        <tr key={rIdx} className="hover:bg-gray-50 dark:hover:bg-gray-800/50">
                          <td className="p-1.5 font-medium border border-gray-200 dark:border-gray-700">
                            {cName}
                          </td>
                          {row.map((cell, cIdx) => (
                            <td
                              key={cIdx}
                              className={`p-1.5 font-mono text-center border border-gray-200 dark:border-gray-700 ${
                                rIdx === cIdx ? 'bg-slate-100 dark:bg-slate-800/80 font-bold' : ''
                              }`}
                            >
                              {cell}
                            </td>
                          ))}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Feature Importances */}
            <div>
              <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-2">
                Random Forest Feature Importances (Gini Impurity)
              </h4>
              <div className="space-y-1.5 max-h-48 overflow-y-auto pr-1">
                {modelInfo.feature_importances?.map((item) => (
                  <div key={item.feature} className="space-y-0.5">
                    <div className="flex justify-between text-[11px] text-gray-600 dark:text-gray-400">
                      <span>{item.feature}</span>
                      <span className="font-mono">{(item.importance * 100).toFixed(2)}%</span>
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-gray-700 h-1.5 rounded-full overflow-hidden">
                      <div
                        className="bg-slate-700 dark:bg-slate-300 h-full rounded-full"
                        style={{ width: `${Math.min(item.importance * 500, 100)}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
