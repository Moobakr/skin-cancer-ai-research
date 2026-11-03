function ModelInsights() {
  return (
    <section className="insights-section">
      <div className="section-header">
        <p className="eyebrow">Model insights</p>
        <h2>Why this project is different</h2>
      </div>
      <div className="insight-grid">
        <article>
          <h3>Patient-level split</h3>
          <p>Training code groups images by patient to reduce leakage between train and test sets.</p>
        </article>
        <article>
          <h3>Multimodal design</h3>
          <p>The final model can combine visual lesion features with clinical metadata features.</p>
        </article>
        <article>
          <h3>Uncertainty-aware output</h3>
          <p>The app warns users when confidence is low or top classes are close together.</p>
        </article>
        <article>
          <h3>Explainability-ready</h3>
          <p>The AI package includes Grad-CAM utilities for heatmap-based model explanation.</p>
        </article>
      </div>
    </section>
  );
}

export default ModelInsights;
