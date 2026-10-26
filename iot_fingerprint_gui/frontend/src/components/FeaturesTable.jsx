import React from 'react';

export default function FeaturesTable({ features = {} }) {
  if (!features || Object.keys(features).length === 0) {
    return <div className="text-xs text-gray-400">No feature vector extracted yet.</div>;
  }

  const items = [
    { label: 'Mean', key: 'raw_mean' },
    { label: 'Std Dev', key: 'raw_std' },
    { label: 'Median', key: 'raw_median' },
    { label: 'Min IAT', key: 'raw_min' },
    { label: 'Max IAT', key: 'raw_max' },
    { label: 'P25', key: 'raw_p25' },
    { label: 'P75', key: 'raw_p75' },
    { label: 'P95', key: 'raw_p95' },
    { label: 'P99', key: 'raw_p99' },
    { label: 'Skewness', key: 'raw_skew' },
    { label: 'Kurtosis', key: 'raw_kurtosis' },
    { label: 'Signal Energy', key: 'raw_energy' },
    { label: 'Entropy (bits)', key: 'raw_entropy' },
  ];

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 gap-1.5 text-xs">
      {items.map(({ label, key }) => {
        const val = features[key];
        const formatted = typeof val === 'number' ? val.toFixed(4) : (val || '-');
        return (
          <div
            key={key}
            className="flex justify-between items-center px-2 py-1 bg-gray-50 dark:bg-gray-800/80 rounded border border-gray-100 dark:border-gray-700/60"
          >
            <span className="text-gray-500 dark:text-gray-400 text-[11px]">{label}:</span>
            <span className="font-mono font-medium text-gray-800 dark:text-gray-200 text-[11px]">
              {formatted}
            </span>
          </div>
        );
      })}
    </div>
  );
}
