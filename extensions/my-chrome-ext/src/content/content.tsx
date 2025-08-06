import { createRoot } from 'react-dom/client';
import FloatingFactChecker from './FloatingFactChecker';
import './content.css';

const CONTAINER_ID = 'fact-checker-floating-container';

let factCheckerRoot: ReturnType<typeof createRoot> | null = null;

function showFloatingFactChecker() {
  const existingContainer = document.getElementById(CONTAINER_ID);
  if (existingContainer) {
    existingContainer.style.display = 'block';
    return;
  }

  const container = document.createElement('div');
  container.id = CONTAINER_ID;
  document.body.appendChild(container);

  factCheckerRoot = createRoot(container);
  factCheckerRoot.render(<FloatingFactChecker onClose={hideFloatingFactChecker} />);
}

function hideFloatingFactChecker() {
  const container = document.getElementById(CONTAINER_ID);
  if (container) {
    container.style.display = 'none';
  }
}

// Listen for messages from popup
chrome.runtime.onMessage.addListener((request: any, _sender: any, sendResponse: (response: any) => void) => {
  if (request.action === 'showFactChecker') {
    showFloatingFactChecker();
    sendResponse({ success: true });
  } else if (request.action === 'hideFactChecker') {
    hideFloatingFactChecker();
    sendResponse({ success: true });
  }
});

// Don't show the fact checker by default