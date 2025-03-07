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
    const [imageUrl, setImageUrl] = useState("");
    const [genre, setGenre] = useState("");
    const [playlist, setPlaylist] = useState([]);

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
            const response = await axios.post("http://127.0.0.1:5000/predict", formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
                responseType: 'json',
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

            setGenre(response.data.genre);

            const hexData = response.data.chart; 
            const binaryString = hexData.match(/.{1,2}/g) 
                .map(byte => String.fromCharCode(parseInt(byte, 16))) 
                .join('');
            const base64Chart = btoa(binaryString); 
            setImageUrl(`data:image/png;base64,${base64Chart}`);

            setPlaylist(response.data.playlist);
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
        <>
            <p>File uploaded successfully!</p>
            <h3>Predicted Genre: {genre}</h3>
            {imageUrl && <img src={imageUrl} alt="Genre Prediction Graph" />}

            <h3>Spotify Recommendations</h3>
            <ul>
                {playlist.map((track, index) => (
                    <li key={index}>
                        <strong>{track.name}</strong> by {track.artists.join(', ')}
                    </li>
                ))}
            </ul>
        </>
        )}

        {status === 'error' && (
        <p>Upload failed. Please try again.</p>
        )}

        <img src={imageUrl} alt="Top-n graph"/>

    </div>
  );
};

export default Uploader;
