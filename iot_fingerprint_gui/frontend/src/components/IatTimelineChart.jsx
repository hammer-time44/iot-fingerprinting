import React from 'react';

export default function IatTimelineChart({ iats = [], height = 80 }) {
  if (!iats || iats.length === 0) {
    return (
      <div
        className="flex items-center justify-center bg-gray-100 dark:bg-gray-800 rounded border border-gray-200 dark:border-gray-700 text-xs text-gray-500"
        style={{ height }}
      >
        Waiting for packet stream...
      </div>
    );
  }

  const values = iats.slice(0, 128);
  const maxVal = Math.max(...values, 0.001);
  const minVal = Math.min(...values, 0.0);
  const width = 360;

  // Generate SVG path for 128 points
  const points = values.map((val, idx) => {
    const x = (idx / (values.length - 1 || 1)) * (width - 10) + 5;
    const norm = (val - minVal) / (maxVal - minVal || 1);
    const y = height - 10 - norm * (height - 20);
    return `${x},${y}`;
  });

  const pathD = `M ${points.join(' L ')}`;

  return (
    <div className="bg-gray-50 dark:bg-gray-900/60 p-2 rounded border border-gray-200 dark:border-gray-800">
      <div className="flex justify-between items-center text-[10px] text-gray-500 mb-1">
        <span>Window IAT Timeline (128 packets)</span>
        <span>Max: {maxVal.toFixed(3)}s</span>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto overflow-visible">
        {/* Baseline grid */}
        <line x1="5" y1={height - 10} x2={width - 5} y2={height - 10} stroke="#94a3b8" strokeWidth="0.5" strokeDasharray="2,2" />
        <line x1="5" y1="10" x2={width - 5} y2="10" stroke="#94a3b8" strokeWidth="0.5" strokeDasharray="2,2" />

        {/* IAT Curve */}
        <path d={pathD} fill="none" stroke="#6366f1" strokeWidth="1.8" className="dark:stroke-indigo-400" />
      </svg>
      <div className="flex justify-between text-[9px] text-gray-400 mt-0.5">
        <span>Packet #1</span>
        <span>Packet #64</span>
        <span>Packet #128</span>
      </div>
    </div>
  );
}
