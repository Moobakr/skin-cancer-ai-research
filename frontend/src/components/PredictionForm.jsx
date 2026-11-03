import { useState } from 'react';
import { UploadCloud } from 'lucide-react';

const initialMetadata = {
  age: '',
  sex: '',
  anatom_site: '',
  smoke: '',
  drink: ''
};

function PredictionForm({ onPredict, loading }) {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState('');
  const [metadata, setMetadata] = useState(initialMetadata);

  function handleImageChange(event) {
    const selected = event.target.files?.[0];
    if (!selected) return;
    setImage(selected);
    setPreview(URL.createObjectURL(selected));
  }

  function updateMetadata(event) {
    const { name, value } = event.target;
    setMetadata((current) => ({ ...current, [name]: value }));
  }

  function handleSubmit(event) {
    event.preventDefault();
    onPredict({ image, metadata });
  }

  return (
    <section className="panel">
      <div className="panel-header">
        <h2>Analyze lesion image</h2>
        <p>Upload an image and optionally add patient/clinical context.</p>
      </div>

      <form className="prediction-form" onSubmit={handleSubmit}>
        <label className="upload-box">
          <UploadCloud size={32} />
          <span>{image ? image.name : 'Choose JPG or PNG image'}</span>
          <input type="file" accept="image/*" onChange={handleImageChange} />
        </label>

        {preview && (
          <div className="preview-wrapper">
            <img src={preview} alt="Selected skin lesion preview" />
          </div>
        )}

        <div className="form-grid">
          <label>
            Age
            <input name="age" type="number" min="0" max="120" value={metadata.age} onChange={updateMetadata} placeholder="e.g. 52" />
          </label>
          <label>
            Sex / gender
            <select name="sex" value={metadata.sex} onChange={updateMetadata}>
              <option value="">Unknown</option>
              <option value="male">Male</option>
              <option value="female">Female</option>
            </select>
          </label>
          <label>
            Body region
            <select name="anatom_site" value={metadata.anatom_site} onChange={updateMetadata}>
              <option value="">Unknown</option>
              {['ABDOMEN', 'ARM', 'BACK', 'CHEST', 'EAR', 'FACE', 'FOOT', 'FOREARM', 'HAND', 'LIP', 'NECK', 'NOSE', 'SCALP', 'THIGH'].map((region) => (
                <option key={region} value={region}>{region.charAt(0) + region.slice(1).toLowerCase()}</option>
              ))}
            </select>
          </label>
          <label>
            Smoking history
            <select name="smoke" value={metadata.smoke} onChange={updateMetadata}>
              <option value="">Unknown</option>
              <option value="true">Yes</option>
              <option value="false">No</option>
            </select>
          </label>
          <label>
            Drinking history
            <select name="drink" value={metadata.drink} onChange={updateMetadata}>
              <option value="">Unknown</option>
              <option value="true">Yes</option>
              <option value="false">No</option>
            </select>
          </label>
        </div>

        <button className="primary-button" type="submit" disabled={loading}>
          {loading ? 'Analyzing...' : 'Analyze image'}
        </button>
      </form>
    </section>
  );
}

export default PredictionForm;
