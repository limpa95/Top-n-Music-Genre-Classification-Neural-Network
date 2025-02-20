// Citation for the following file:
// Date: 2/20/2025
// Adapted from react-starter-app provided in OSU CS340
// The original file was used as a template.
// Source URL: https://github.com/osu-cs340-ecampus/react-starter-app
// Authors: Devin Daniels and Zachary Maes under the supervision of Dr. Michael Curry and Dr. Danielle Safonte

import { Link } from "react-router-dom";

const Navbar = () => {
  return (
    <header>
      <h1>Top-n Music Genre Classification</h1>
      <nav>
        <ul>
          <li>
            <Link to="/">Home</Link>
          </li>
        </ul>
      </nav>
    </header>
  );
};

export default Navbar;
