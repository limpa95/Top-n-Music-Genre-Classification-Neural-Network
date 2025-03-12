// Citation for the following file:
// Date: 2/20/2025
// Adapted from react-starter-app provided in OSU CS340
// The original file was used as a template.
// Source URL: https://github.com/osu-cs340-ecampus/react-starter-app
// Authors: Devin Daniels and Zachary Maes under the supervision of Dr. Michael Curry and Dr. Danielle Safonte

import { Routes, Route } from 'react-router-dom';
import Uploader from "../components/Uploader/Uploader";
function HomePage() {
  return(
    <div id="pageContainer">
      <p>Welcome! You can use this site to submit a music file and get its predicted genre based on our AI model.
          The file format needs to be .wav.</p>
      <Routes>
      <Route path="/" element={<Uploader />} />
      </Routes>
    </div>
  )
}

export default HomePage;
