// Citation for the following file:
// Date: 2/20/2025
// Adapted from react-starter-app provided in OSU CS340
// The original file was used as a template.
// Source URL: https://github.com/osu-cs340-ecampus/react-starter-app
// Authors: Devin Daniels and Zachary Maes under the supervision of Dr. Michael Curry and Dr. Danielle Safonte

import "./App.css";
import { Routes, Route } from "react-router-dom";
import HomePage from "./pages/HomePage";
import logo from "./assets/sound-wave-svgrepo.svg";

function App() {
  return (
    <div id="wrapper">
      <header>
        <h1>
          <img src ={logo} className="logo" alt='logo'></img>
        </h1>
        <h2>Top-n Music Genre Classification</h2>
      </header>
      <Routes>
        <Route path="*" element={<HomePage />} />
      </Routes>
      <footer>
        <p>&copy; 2025 Amine Kaddour-Djebbar, Clinton Merritt, Jun Seo, Patrick Lim</p>
      </footer>
    </div>
  );
}

export default App;
