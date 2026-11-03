import { Microscope } from 'lucide-react';

function Header({ health }) {
  const mode = health?.mode || 'checking';
  const isReal = mode === 'real';

  return (
    <header className="topbar">
      <div className="brand">
        <div className="brand-icon"><Microscope size={24} /></div>
        <div>
          <strong>Skin Cancer AI Assistant</strong>
          <span>PAD-UFES-20 deep learning project</span>
        </div>
      </div>
      <div className={`status-pill ${isReal ? 'status-real' : 'status-demo'}`}>
        {isReal ? 'Real model loaded' : `Mode: ${mode}`}
      </div>
    </header>
  );
}

export default Header;
