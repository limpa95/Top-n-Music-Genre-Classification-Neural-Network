// Citation for the following file:
// Date: 2/20/2025
// Adapted from react-upload-files-tutorial by dariuscosden on GitHub
// The original file was used as a template.
// Source URL: https://github.com/cosdensolutions/code/tree/master/videos/long/react-upload-files-tutorial
// Authors: Darius Coden

import axios from 'axios';
import { useState } from "react";

const Uploader= () => {

    const [file, setFile] = useState(null);
    const [status, setStatus] = useState("idle")

    function handleFileChange(e){
        if (e.target.files) {
            setFile(e.target.files[0]);
        }
    }

    async function handleFileUpload() {
        if (!file) return;

        setStatus('uploading');

        const formData = new FormData();
        formData.append('file', file);

        try{
            await axios.post('https://httpbin.org/post', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                }
            });

            setStatus('success')
        }
        catch{
            setStatus('error')
        };
    }

  return (
    <div>
        <input type="file" onChange={handleFileChange}/>

        {file && (
        <div className="mb-4 text-sm">
          <p>File name: {file.name}</p>
          <p>Size: {(file.size / 1024).toFixed(2)} KB</p>
          <p>Type: {file.type}</p>
        </div>
        )}

        {file && status !== 'uploading' && (
        <button onClick={handleFileUpload}>Upload</button>
        )}

        {status === 'success' && (
        <p>File uploaded successfully!</p>
        )}

        {status === 'error' && (
        <p>Upload failed. Please try again.</p>
        )}

    </div>
  );
};

export default Uploader;
