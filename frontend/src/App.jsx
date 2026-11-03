import { useEffect, useState } from 'react';
import { Activity, Brain, FileText, ShieldCheck, Smartphone } from 'lucide-react';
import { getHealth, predictSkinCancer } from './api/predictApi';
import Header from './components/Header.jsx';
import PredictionForm from './components/PredictionForm.jsx';
import ResultCard from './components/ResultCard.jsx';
import ModelInsights from './components/ModelInsights.jsx';
import Disclaimer from './components/Disclaimer.jsx';

function App() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [health, setHealth] = useState(null);

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch(() => setHealth({ status: 'offline', mode: 'unknown' }));
  }, []);

  async function handlePredict({ image, metadata }) {
    setError('');
    setResult(null);

    if (!image) {
      setError('Please upload an image before prediction.');
      return;
    }

    const formData = new FormData();
    formData.append('image', image);
    Object.entries(metadata).forEach(([key, value]) => {
      if (value !== undefined && value !== null && String(value).trim() !== '') {
        formData.append(key, value);
      }
    });

    try {
      setLoading(true);
      const response = await predictSkinCancer(formData);
      setResult(response);
    } catch (err) {
      setError(err.message || 'Something went wrong while predicting.');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <Header health={health} />

      <main>
        <section className="hero-section">
          <div className="hero-copy">
            <p className="eyebrow">Explainable multimodal AI project</p>
            <h1>Skin Lesion Screening Assistant</h1>
            <p className="hero-text">
              Upload a lesion image, add optional clinical metadata, and view a confidence-aware AI prediction with safety warnings and technical details.
            </p>
            <div className="feature-grid">
              <div><Brain size={20} /> EfficientNet transfer learning</div>
              <div><Activity size={20} /> Confidence and risk levels</div>
              <div><ShieldCheck size={20} /> Safety-first wording</div>
              <div><Smartphone size={20} /> Responsive React interface</div>
            </div>
          </div>
          <div className="hero-card">
            <FileText size={32} />
            <h3>Project uniqueness</h3>
            <p>
              This is not only a classifier. It is a full-stack explainable assistant with metadata fusion, image-quality checks, uncertainty warnings, and report-style output.
            </p>
          </div>
        </section>

        <section className="workspace-grid">
          <PredictionForm onPredict={handlePredict} loading={loading} />
          <ResultCard result={result} loading={loading} error={error} />
        </section>

        <ModelInsights />
        <Disclaimer />
      </main>
    </div>
  );
}

export default App;
