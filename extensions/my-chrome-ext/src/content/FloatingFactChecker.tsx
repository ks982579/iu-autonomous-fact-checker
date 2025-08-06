import React, { useState, useRef, useCallback } from 'react';

interface Position {
  x: number;
  y: number;
}

interface FloatingFactCheckerProps {
  onClose?: () => void;
}

const FloatingFactChecker: React.FC<FloatingFactCheckerProps> = ({ onClose }) => {
  const [position, setPosition] = useState<Position>({ x: 50, y: 50 });
  const [isDragging, setIsDragging] = useState(false);
  const [isMinimized, setIsMinimized] = useState(false);
  const dragOffset = useRef<Position>({ x: 0, y: 0 });
  const containerRef = useRef<HTMLDivElement>(null);

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
            <label className="retro-label">Select text to fact-check:</label>
            <textarea 
              className="retro-textarea"
              placeholder="Paste or type claim here..."
              rows={3}
            />
          </div>
          <div className="button-section">
            <button className="retro-button primary">
              ANALYZE CLAIM
            </button>
            <button className="retro-button secondary">
              CLEAR
            </button>
          </div>
          <div className="status-section">
            <div className="status-line">
              <span className="status-label">STATUS:</span>
              <span className="status-value">READY</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default FloatingFactChecker;