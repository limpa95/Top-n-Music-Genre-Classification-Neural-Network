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
    const [uploadProgress, setUploadProgress] = useState(0);
    const [graph, setGraph] = useState();
    const [imageUrl, setImageUrl] = useState("");

    function handleFileChange(e){
        if (e.target.files) {
            setFile(e.target.files[0]);
        }
    }

    async function handleFileUpload() {
        if (!file) return;

        setStatus('uploading');
        setUploadProgress(0);

        const formData = new FormData();
        formData.append('file', file);

        try{
            const response = await axios.post("http://18.189.143.41:5000/predict", formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
                responseType: 'blob',
                onUploadProgress: (progressEvent) => {
                    const progress = progressEvent.total
                    ? Math.round((progressEvent.loaded * 100) / progressEvent.total)
                    : 0;
                    setUploadProgress(progress);
                }
            });

            setStatus('success');
            setUploadProgress(100);

            console.log(response)

            const blob = new Blob([response.data], { type: 'image/png' }); // Expecting an image (PNG)
            const imageUrl = URL.createObjectURL(blob);
            setImageUrl(imageUrl); // Store image URL in state to display it
        }
        catch (error) {
            setStatus('error');
            setUploadProgress(0);

            if (error.response) {
                console.error("Error data:", error.response.data)
                console.error("Error status:", error.response.status)
            }
        };
    }

  return (
    <div>
        <input 
            className="fileInput"
            type="file" 
            onChange={handleFileChange}/>

        {file && (
        <div>
          <p>File name: {file.name}</p>
          <p>Size: {(file.size / 1024).toFixed(2)} KB</p>
          <p>Type: {file.type}</p>
        </div>
        )}

        {status === 'uploading' && (
            <div>
            <p>{uploadProgress}% uploaded</p>
            </div>
        )}

        {file && status !== 'uploading' && (
        <button 
            className="buttonUpload"
            onClick={handleFileUpload}>Upload</button>
        )}

        {status === 'success' && (
        <p>File uploaded successfully!</p>
        )}

        {status === 'error' && (
        <p>Upload failed. Please try again.</p>
        )}

        <img src={imageUrl} alt="Top-n graph"/>

    </div>
  );
};

export default Uploader;
