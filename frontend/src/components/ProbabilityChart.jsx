function ProbabilityChart({ probabilities }) {
  const entries = Object.entries(probabilities || {}).sort((a, b) => b[1] - a[1]);

  return (
    <div className="probability-chart">
      <h3>Class probabilities</h3>
      {entries.map(([label, value]) => (
        <div className="prob-row" key={label}>
          <div className="prob-label">
            <span>{label}</span>
            <strong>{(value * 100).toFixed(1)}%</strong>
          </div>
          <div className="prob-track">
            <div className="prob-fill" style={{ width: `${Math.max(value * 100, 2)}%` }} />
          </div>
        </div>
      ))}
    </div>
  );
}

export default ProbabilityChart;
