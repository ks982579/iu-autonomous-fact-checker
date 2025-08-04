import { createRoot } from 'react-dom/client';
import FloatingFactChecker from './FloatingFactChecker';
import './content.css';

const CONTAINER_ID = 'fact-checker-floating-container';

function injectFloatingFactChecker() {
  if (document.getElementById(CONTAINER_ID)) {
    return;
  }

  const container = document.createElement('div');
  container.id = CONTAINER_ID;
  document.body.appendChild(container);

  const root = createRoot(container);
  root.render(<FloatingFactChecker />);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', injectFloatingFactChecker);
} else {
  injectFloatingFactChecker();
}