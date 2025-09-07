// Simple popup without hooks to avoid React issues - there were a few
function SimplePopup() {
  const handleYes = () => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs[0]?.id) {
        chrome.tabs.sendMessage(tabs[0].id, { action: "showFactChecker" });
      }
      window.close();
    });
  };

  const handleNo = () => {
    window.close();
  };

  return (
    <div className="popup-container">
      <div className="popup-header">
        <span className="terminal-prompt">$</span> FACT-CHECKER v1.0
      </div>
      
      <div className="popup-content">
        <p className="popup-question">
          Do you want to open the fact-checking window?
        </p>
        
        <div className="popup-buttons">
          <button 
            className="popup-button yes"
            onClick={handleYes}
          >
            YES
          </button>
          <button 
            className="popup-button no"
            onClick={handleNo}
          >
            NO
          </button>
        </div>
      </div>
    </div>
  );
}

export default SimplePopup;