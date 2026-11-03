function ReportView({ result }) {
  return (
    <div className="report-card">
      <h3>Doctor-style summary</h3>
      <dl>
        <div>
          <dt>Predicted class</dt>
          <dd>{result.predicted_class}</dd>
        </div>
        <div>
          <dt>Confidence</dt>
          <dd>{(result.confidence * 100).toFixed(1)}%</dd>
        </div>
        <div>
          <dt>Risk group</dt>
          <dd>{result.risk_level}</dd>
        </div>
        <div>
          <dt>Top alternatives</dt>
          <dd>
            {result.top_predictions?.map((item) => `${item.class_name}: ${(item.probability * 100).toFixed(1)}%`).join(' | ')}
          </dd>
        </div>
      </dl>
      <button className="secondary-button" type="button" onClick={() => window.print()}>
        Print / save report
      </button>
    </div>
  );
}

export default ReportView;
