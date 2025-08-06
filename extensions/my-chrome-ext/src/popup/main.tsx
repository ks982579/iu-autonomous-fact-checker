import { createRoot } from "react-dom/client";
import SimplePopup from "./SimplePopup";
import "./Popup.css";

const container = document.getElementById("root");
if (container) {
  const root = createRoot(container);
  root.render(<SimplePopup />);
}
