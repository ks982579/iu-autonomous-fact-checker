import React, { useState, useRef, useCallback } from 'react';
import { useHealth } from '../context/HealthContext';

interface Position {
  x: number;
  y: number;
}

interface FloatingFactCheckerProps {
  onClose?: () => void;
}

interface FactCheckResult {
  claim: string;
  verdict: string;
  confidence: number;
  evidence_count: number;
}

interface FactCheckResponse {
  original_text: string;
  is_political: boolean;
  extracted_claims: Array<{
    text: string;
    confidence: number;
    is_factual_claim: boolean;
  }>;
  fact_check_results: FactCheckResult[];
  processing_time_ms: number;
  success: boolean;
  message?: string;
}

const FloatingFactChecker: React.FC<FloatingFactCheckerProps> = ({ onClose }) => {
  const [position, setPosition] = useState<Position>({ x: 50, y: 50 });
  const [isDragging, setIsDragging] = useState(false);
  const [isMinimized, setIsMinimized] = useState(false);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState<FactCheckResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const dragOffset = useRef<Position>({ x: 0, y: 0 });
  const containerRef = useRef<HTMLDivElement>(null);
  
  // Get health status from context
  const { health } = useHealth();

  // Load config from extension storage
  const getConfig = async () => {
    return new Promise((resolve) => {
      chrome.storage.local.get(['config'], (result) => {
        resolve(result.config || {
          api: {
            base_url: 'http://localhost:8000',
            request_timeout_ms: 60000
          }
        });
      });
    });
  };

  // Fact-check API call
  const handleFactCheck = async () => {
    if (!inputText.trim()) {
      setError('Please enter some text to fact-check');
      return;
    }

    if (health.status !== 'healthy') {
      setError('API is not available. Please check the connection.');
      return;
    }

    setIsLoading(true);
    setError(null);
    setResults(null);

    try {
      const config: any = await getConfig();
      const baseUrl = config.api?.base_url || 'http://localhost:8000';
      const timeout = config.api?.request_timeout_ms || 30000;

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), timeout);

      const response = await fetch(`${baseUrl}/fact-check`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: inputText }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data: FactCheckResponse = await response.json();
      setResults(data);

    } catch (error: any) {
      if (error.name === 'AbortError') {
        setError('Request timed out. The analysis is taking too long.');
      } else {
        setError(`Failed to analyze: ${error.message}`);
      }
    } finally {
      setIsLoading(false);
    }
  };

  // Clear function
  const handleClear = () => {
    setInputText('');
    setResults(null);
    setError(null);
  };

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (!containerRef.current) return;
    
    const rect = containerRef.current.getBoundingClientRect();
    dragOffset.current = {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    };
    setIsDragging(true);
  }, []);

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!isDragging) return;
    
    setPosition({
      x: e.clientX - dragOffset.current.x,
      y: e.clientY - dragOffset.current.y
    });
  }, [isDragging]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  React.useEffect(() => {
    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      return () => {
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [isDragging, handleMouseMove, handleMouseUp]);

  return (
    <div
      ref={containerRef}
      className={`fact-checker-container ${isDragging ? 'dragging' : ''}`}
      style={{
        position: 'fixed',
        left: `${position.x}px`,
        top: `${position.y}px`,
        zIndex: 2147483647,
      }}
    >
      <div 
        className="fact-checker-header" 
        onMouseDown={handleMouseDown}
      >
        <div className="fact-checker-title">
          <span className="terminal-prompt">$</span> FACT-CHECKER v1.0
        </div>
        <div className="fact-checker-controls">
          <button 
            className="minimize-btn"
            onClick={() => setIsMinimized(!isMinimized)}
          >
            {isMinimized ? '□' : '_'}
          </button>
          <button 
            className="close-btn"
            onClick={onClose}
          >
            ×
          </button>
        </div>
      </div>
      
      {!isMinimized && (
        <div className="fact-checker-content">
          <div className="terminal-line">
            <span className="prompt">&gt;</span> Ready to analyze claims...
          </div>
          <div className="input-section">
            <label className="retro-label">Paste text to fact-check:</label>
            <textarea 
              className="retro-textarea"
              placeholder="Paste social media post or text here..."
              rows={4}
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              disabled={isLoading}
            />
          </div>
          
          {error && (
            <div className="error-section">
              <div className="terminal-line error">
                <span className="prompt">✗</span> {error}
              </div>
            </div>
          )}

          <div className="button-section">
            <button 
              className="retro-button primary"
              onClick={handleFactCheck}
              disabled={isLoading || !inputText.trim() || health.status !== 'healthy'}
            >
              {isLoading ? 'ANALYZING...' : 'ANALYZE CLAIM'}
            </button>
            <button 
              className="retro-button secondary"
              onClick={handleClear}
              disabled={isLoading}
            >
              CLEAR
            </button>
          </div>

          {isLoading && (
            <div className="loading-section">
              <div className="terminal-line">
                <span className="prompt">&gt;</span> Processing claim...
              </div>
              <div className="loading-bar">
                <div className="loading-progress"></div>
              </div>
            </div>
          )}

          {results && (
            <div className="results-section">
              <div className="terminal-line success">
                <span className="prompt">✓</span> Analysis Complete ({results.processing_time_ms}ms)
              </div>
              
              {!results.success && results.message && (
                <div className="result-message">
                  <div className="terminal-line warning">
                    <span className="prompt">!</span> {results.message}
                  </div>
                </div>
              )}

              {results.success && (
                <>
                  <div className="political-status">
                    <span className="status-label">Political Content:</span>
                    <span className={`status-value ${results.is_political ? 'yes' : 'no'}`}>
                      {results.is_political ? 'YES' : 'NO'}
                    </span>
                  </div>

                  {results.extracted_claims.length > 0 && (
                    <div className="claims-section">
                      <div className="section-header">Claims Found ({results.extracted_claims.length}):</div>
                      {results.extracted_claims.map((claim, index) => (
                        <div key={index} className="claim-item">
                          <div className="claim-text">"{claim.text}"</div>
                          <div className="claim-confidence">
                            Confidence: {(claim.confidence * 100).toFixed(1)}%
                          </div>
                        </div>
                      ))}
                    </div>
                  )}

                  {results.fact_check_results.length > 0 && (
                    <div className="verdicts-section">
                      <div className="section-header">Fact-Check Results:</div>
                      {results.fact_check_results.map((result, index) => (
                        <div key={index} className="verdict-item">
                          <div className={`verdict-label verdict-${result.verdict.toLowerCase()}`}>
                            {result.verdict}
                          </div>
                          <div className="verdict-details">
                            <div>Confidence: {(result.confidence * 100).toFixed(1)}%</div>
                            <div>Evidence Sources: {result.evidence_count}</div>
                          </div>
                          <div className="verdict-claim">"{result.claim}"</div>
                        </div>
                      ))}
                    </div>
                  )}
                </>
              )}
            </div>
          )}
          <div className="status-section">
            <div className="status-line">
              <span className="status-label">API STATUS:</span>
              <span className={`status-value status-${health.status}`}>
                {health.status.toUpperCase()}
              </span>
            </div>
            {health.lastChecked && (
              <div className="status-details">
                <span className="status-timestamp">
                  Last: {health.lastChecked.toLocaleTimeString()}
                </span>
                {health.error && (
                  <span className="status-error"> • {health.error}</span>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default FloatingFactChecker;