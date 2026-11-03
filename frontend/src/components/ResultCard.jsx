import { useState } from 'react';
import { AlertTriangle, CheckCircle2, Loader2 } from 'lucide-react';
import ProbabilityChart from './ProbabilityChart.jsx';
import QualityWarnings from './QualityWarnings.jsx';
import ReportView from './ReportView.jsx';

function ResultCard({ result, loading, error }) {
  const [technicalMode, setTechnicalMode] = useState(false);

  if (loading) {
    return (
      <section className="panel result-panel center-state">
        <Loader2 className="spin" size={36} />
        <h2>Analyzing image...</h2>
        <p>The backend is preprocessing the upload and running model inference.</p>
      </section>
    );
  }

  if (error) {
    return (
      <section className="panel result-panel center-state error-state">
        <AlertTriangle size={36} />
        <h2>Prediction failed</h2>
        <p>{error}</p>
      </section>
    );
  }

  if (!result) {
    return (
      <section className="panel result-panel center-state muted-state">
        <CheckCircle2 size={36} />
        <h2>Results will appear here</h2>
        <p>After prediction, you will see class probabilities, confidence, risk level, warnings, and report-style output.</p>
      </section>
    );
  }

  const confidencePct = (result.confidence * 100).toFixed(1);

  return (
    <section className="panel result-panel">
      <div className="result-header">
        <div>
          <p className="eyebrow">Prediction result</p>
          <h2>{result.predicted_class}</h2>
        </div>
        <div className="confidence-badge">
          <span>{confidencePct}%</span>
          <small>{result.confidence_level} confidence</small>
        </div>
      </div>

      <div className="risk-card">
        <strong>{result.risk_level}</strong>
        <p>{result.class_description}</p>
      </div>

      <QualityWarnings result={result} />

      {result.uncertainty_warnings?.length > 0 && (
        <div className="warning-list">
          {result.uncertainty_warnings.map((warning) => (
            <div key={warning} className="warning-item"><AlertTriangle size={16} /> {warning}</div>
          ))}
        </div>
      )}

      <div className="mode-toggle">
        <button className={!technicalMode ? 'active' : ''} onClick={() => setTechnicalMode(false)}>Simple mode</button>
        <button className={technicalMode ? 'active' : ''} onClick={() => setTechnicalMode(true)}>Technical mode</button>
      </div>

      {technicalMode ? (
        <>
          <ProbabilityChart probabilities={result.probabilities} />
          <pre className="json-box">{JSON.stringify(result.quality_report, null, 2)}</pre>
        </>
      ) : (
        <ReportView result={result} />
      )}

      <p className="disclaimer-text">{result.disclaimer}</p>
    </section>
  );
}

export default ResultCard;
