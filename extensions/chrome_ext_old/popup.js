// Wait for the DOM to be fully loaded
document.addEventListener("DOMContentLoaded", function() {
  const button = document.getElementById("clickButton");
  const status = document.getElementById("status");

  // Add click event listener to the button
  button.addEventListener("click", function() {
    // Show a success message
    status.textContent = "Button clicked! Extension is working!";
    status.className = "success";

    // Get information about the current tab
    chrome.tabs.query({ active: true, currentWindow: true }, function(tabs) {
      const currentTab = tabs[0];
      console.log("Current tab URL:", currentTab.url);
      console.log("Current tab title:", currentTab.title);

      // Update status with tab info
      status.innerHTML = `
        <strong>Success!</strong><br>
        Current page: ${currentTab.title}
      `;
    });
  });
});
