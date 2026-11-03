import { AlertTriangle, CheckCircle2 } from 'lucide-react';

function QualityWarnings({ result }) {
  const warnings = result.quality_warnings || [];
  if (warnings.length === 0) {
    return (
      <div className="quality-ok">
        <CheckCircle2 size={16} /> No major image quality warnings detected.
      </div>
    );
  }

  return (
    <div className="warning-list">
      {warnings.map((warning) => (
        <div key={warning} className="warning-item"><AlertTriangle size={16} /> {warning}</div>
      ))}
    </div>
  );
}

export default QualityWarnings;
