import React from "react";
import ReactDom from "react-dom/client";
import Popup from "./Popup";
import "./Popup.css";

ReactDom.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <Popup />
  </React.StrictMode>,
);
