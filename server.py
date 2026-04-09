#!/usr/bin/env python3
"""
REST server for X-ray image upload and analysis
Using only Python 3.12 standard library
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import json
import os
import subprocess
from datetime import datetime
import cgi

class CustomServer(BaseHTTPRequestHandler):
    """Handle custom API requests"""
    
    # Class variable to share state across instances
    uploaded_image_path = None
    
    def _send_json_response(self, data, status=200):
        """Send JSON response"""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def _send_error_response(self, message, status=400):
        """Send error response"""
        self._send_json_response({"error": message, "success": False}, status)
    
    def _send_html_response(self, html_content, status=200):
        """Send HTML response"""
        self.send_response(status)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html_content.encode('utf-8'))
    
    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urllib.parse.urlparse(self.path)
        
        # Main HTML interface
        if parsed_path.path == '/':
            self._send_html_response(self._get_html_interface())
        
        # Health check endpoint
        elif parsed_path.path == '/health':
            self._send_json_response({
                "status": "ok",
                "timestamp": datetime.now().isoformat()
            })
        
        # Get current uploaded image path
        elif parsed_path.path == '/current-image':
            print(f"Current image: {CustomServer.uploaded_image_path}")
            if CustomServer.uploaded_image_path and os.path.exists(CustomServer.uploaded_image_path):
                self._send_json_response({
                    "has_image": True,
                    "image_path": CustomServer.uploaded_image_path,
                    "filename": os.path.basename(CustomServer.uploaded_image_path)
                })
            else:
                self._send_json_response({
                    "has_image": False,
                    "image_path": None
                })
        
        else:
            self._send_json_response({
                "message": "Available endpoints:",
                "endpoints": {
                    "GET /": "Web interface",
                    "POST /upload": "Upload X-ray image",
                    "POST /run-main": "Run main.py with uploaded image",
                    "GET /current-image": "Check current uploaded image"
                }
            })
    
    def do_POST(self):
        """Handle POST requests"""
        parsed_path = urllib.parse.urlparse(self.path)
        
        # API 1: Upload image
        if parsed_path.path == '/upload':
            self._handle_image_upload()
        
        # API 2: Run main.py with uploaded image
        elif parsed_path.path == '/run-main':
            self._handle_run_main()
        
        else:
            self._send_error_response("Endpoint not found", 404)
    
    def _handle_image_upload(self):
        """Handle image upload"""
        content_type = self.headers.get('Content-Type', '')
        
        if 'multipart/form-data' not in content_type:
            self._send_error_response("Use multipart/form-data", 400)
            return
        
        try:
            # Parse form data
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={'REQUEST_METHOD': 'POST'}
            )
            
            # Get uploaded file
            if 'image' not in form:
                self._send_error_response("No image field found", 400)
                return
            
            file_item = form['image']
            if not file_item.filename:
                self._send_error_response("No file selected", 400)
                return
            
            # Save image
            os.makedirs('uploaded_images', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            original_filename = os.path.basename(file_item.filename)
            safe_filename = f"{timestamp}_{original_filename}"
            filepath = f'uploaded_images/{safe_filename}'
            
            with open(filepath, 'wb') as f:
                f.write(file_item.file.read())
            
            # Store the path as class variable for persistence
            CustomServer.uploaded_image_path = os.path.abspath(filepath)
            print(f"Image saved and stored at: {CustomServer.uploaded_image_path}")
            
            self._send_json_response({
                "message": "Image uploaded successfully",
                "success": True,
                "filename": safe_filename,
                "path": CustomServer.uploaded_image_path,
                "size": os.path.getsize(filepath)
            }, 201)
        
        except Exception as e:
            self._send_error_response(f"Upload failed: {str(e)}", 500)
    
    def _handle_run_main(self):
        """Run main.py with the uploaded image"""
        print(f"Image path from class variable: {CustomServer.uploaded_image_path}")
        
        # Check if an image has been uploaded
        if not CustomServer.uploaded_image_path:
            self._send_error_response("No image uploaded. Please upload an image first.", 400)
            return
        
        # Check if image file exists
        if not os.path.exists(CustomServer.uploaded_image_path):
            self._send_error_response("Uploaded image file not found. Please upload again.", 404)
            return
        
        # Check if main.py exists
        if not os.path.exists('main.py'):
            self._send_error_response("main.py not found in server directory", 404)
            return
        
        try:
            # Execute main.py with the image path argument - increased timeout to 60 seconds
            print("Starting analysis... This may take up to 60 seconds.")
            result = subprocess.run(
                ['python3', 'main.py', '--image', CustomServer.uploaded_image_path],
                capture_output=True,
                text=True,
                timeout=60  # Increased to 60 seconds for model inference
            )
            
            # Parse the output from main.py
            analysis_result = None
            predictions = {}
            
            if result.stdout:
                print("Raw output from main.py:")
                print(result.stdout)
                
                # Parse the predictions from stdout
                lines = result.stdout.split('\n')
                for line in lines:
                    line = line.strip()
                    # Look for lines like "Nodule:0.6898" or "Nodule: 0.6898"
                    if ':' in line and not line.startswith('=') and not line.startswith('-'):
                        parts = line.split(':')
                        if len(parts) == 2:
                            disease = parts[0].strip()
                            try:
                                # Handle cases where value might have spaces
                                value_str = parts[1].strip()
                                # Extract first number found
                                import re
                                numbers = re.findall(r"[-+]?\d*\.\d+|\d+", value_str)
                                if numbers:
                                    score = float(numbers[0])
                                    predictions[disease] = score
                            except ValueError:
                                continue
                
                # Also try to extract from JSON if present
                if 'JSON OUTPUT:' in result.stdout:
                    lines = result.stdout.split('\n')
                    json_lines = []
                    in_json = False
                    
                    for line in lines:
                        if 'JSON OUTPUT:' in line:
                            in_json = True
                            continue
                        if in_json and line.strip().startswith('{'):
                            json_lines.append(line)
                        elif in_json and line.strip() == '}':
                            json_lines.append(line)
                            break
                        elif in_json and line.strip().endswith('}'):
                            json_lines.append(line)
                            break
                    
                    if json_lines:
                        try:
                            json_str = '\n'.join(json_lines)
                            json_data = json.loads(json_str)
                            if 'predictions' in json_data:
                                predictions = json_data['predictions']
                            elif 'all_predictions' in json_data:
                                predictions = json_data['all_predictions']
                        except json.JSONDecodeError as e:
                            print(f"JSON parse error: {e}")
            
            # Create structured analysis result
            if predictions:
                # Sort predictions by score
                sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
                
                analysis_result = {
                    "predictions": predictions,
                    "sorted_predictions": [{"disease": k, "score": v} for k, v in sorted_predictions],
                    "top_predictions": [{"disease": k, "score": v} for k, v in sorted_predictions[:10]]
                }
            
            # Check if main.py execution failed
            if result.returncode != 0:
                self._send_json_response({
                    "success": False,
                    "error": "Analysis failed",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "return_code": result.returncode,
                    "image_used": CustomServer.uploaded_image_path
                }, 500)
                return
            
            # Return successful response with predictions
            self._send_json_response({
                "success": True,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
                "analysis": analysis_result,
                "predictions": predictions,
                "image_used": CustomServer.uploaded_image_path
            })
        
        except subprocess.TimeoutExpired:
            print("Analysis timed out after 60 seconds")
            self._send_error_response("Analysis timed out (60 seconds). The model is taking too long to process. Please try with a smaller image or check the model configuration.", 408)
        except subprocess.CalledProcessError as e:
            print(f"Execution failed with exit code {e.returncode}")
            self._send_error_response(f"Execution failed with exit code {e.returncode}: {str(e)}", 500)
        except FileNotFoundError:
            self._send_error_response("Python3 not found. Please ensure Python 3 is installed.", 500)
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            self._send_error_response(f"Execution failed: {str(e)}", 500)

    def _get_html_interface(self):
        """Generate HTML interface with image upload and analysis"""
        return """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Chest X-ray Analysis Server</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 900px; margin: auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; margin-top: 0; }
        .section { border: 1px solid #ddd; padding: 20px; margin-bottom: 20px; border-radius: 5px; background: #fafafa; }
        h2 { margin-top: 0; color: #555; }
        .image-preview { max-width: 100%; margin-top: 10px; border: 1px solid #ddd; border-radius: 5px; }
        .image-info { background: #e3f2fd; padding: 10px; margin-top: 10px; border-radius: 5px; font-size: 14px; }
        input[type="file"] { margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px; width: 100%; }
        button { background: #007bff; color: white; border: none; padding: 12px 24px; cursor: pointer; border-radius: 5px; font-size: 14px; margin-right: 10px; }
        button:hover { background: #0056b3; }
        .result { background: #f4f4f4; padding: 15px; margin-top: 15px; white-space: pre-wrap; font-family: monospace; border-radius: 5px; max-height: 500px; overflow-y: auto; }
        .error { color: red; }
        .success { color: green; }
        .loading { color: #007bff; font-style: italic; }
        .status-badge { display: inline-block; padding: 3px 8px; border-radius: 3px; font-size: 12px; font-weight: bold; }
        .status-uploaded { background: #28a745; color: white; }
        .status-not-uploaded { background: #dc3545; color: white; }
        .prediction-table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        .prediction-table th, .prediction-table td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
        .prediction-table th { background: #f0f0f0; }
        .high { color: #dc3545; font-weight: bold; }
        .moderate { color: #fd7e14; font-weight: bold; }
        .low { color: #ffc107; }
        .negative { color: #28a745; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏥 Chest X-ray Analysis Server</h1>
        
        <!-- Upload Image Section -->
        <div class="section">
            <h2>📷 Upload X-ray Image</h2>
            <form id="uploadForm">
                <input type="file" id="imageFile" accept="image/*" required>
                <button type="submit">Upload Image</button>
            </form>
            <div id="uploadResult"></div>
            <div id="imageStatus"></div>
        </div>
        
        <!-- Run Analysis Section -->
        <div class="section">
            <h2>🔬 Run X-ray Analysis</h2>
            <button id="runButton" onclick="runAnalysis()">Analyze Uploaded Image</button>
            <div id="analysisResult"></div>
        </div>
    </div>
    
    <script>
        let currentImagePath = null;
        
        // Check image status on page load and periodically
        async function checkImageStatus() {
            try {
                const response = await fetch('/current-image');
                const data = await response.json();
                
                if (data.has_image) {
                    currentImagePath = data.image_path;
                    document.getElementById('imageStatus').innerHTML = `
                        <div class="image-info">
                            <span class="status-badge status-uploaded">✓ Image Uploaded</span>
                            <strong>Current image:</strong> ${data.filename}<br>
                            <strong>Path:</strong> ${data.image_path}
                        </div>
                    `;
                } else {
                    currentImagePath = null;
                    document.getElementById('imageStatus').innerHTML = `
                        <div class="image-info">
                            <span class="status-badge status-not-uploaded">✗ No Image</span>
                            Please upload an X-ray image first
                        </div>
                    `;
                }
            } catch (error) {
                console.error('Error checking image status:', error);
            }
        }
        
        // Handle image upload
        document.getElementById('uploadForm').onsubmit = async (e) => {
            e.preventDefault();
            const file = document.getElementById('imageFile').files[0];
            if (!file) return;
            
            const formData = new FormData();
            formData.append('image', file);
            
            const resultDiv = document.getElementById('uploadResult');
            resultDiv.innerHTML = '<div class="result loading">Uploading image...</div>';
            
            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                
                if (response.ok && data.success) {
                    resultDiv.innerHTML = `
                        <div class="result success">
                            ✅ ${data.message}<br>
                            Filename: ${data.filename}<br>
                            Size: ${(data.size / 1024).toFixed(2)} KB<br>
                            Path: ${data.path}
                        </div>
                    `;
                    // Show image preview
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        const preview = document.createElement('div');
                        preview.innerHTML = `<img src="${e.target.result}" class="image-preview" style="max-height: 300px;">`;
                        resultDiv.appendChild(preview);
                    };
                    reader.readAsDataURL(file);
                    
                    // Update image status after upload
                    await checkImageStatus();
                } else {
                    resultDiv.innerHTML = `<div class="result error">❌ Error: ${data.error}</div>`;
                }
            } catch (error) {
                resultDiv.innerHTML = `<div class="result error">❌ Error: ${error.message}</div>`;
            }
        };
        
        // Run analysis with uploaded image
        async function runAnalysis() {
            const runButton = document.getElementById('runButton');
            const resultDiv = document.getElementById('analysisResult');
            
            // Check if image is uploaded
            const statusResponse = await fetch('/current-image');
            const statusData = await statusResponse.json();
            
            if (!statusData.has_image) {
                resultDiv.innerHTML = '<div class="result error">❌ No image uploaded. Please upload an image first.</div>';
                return;
            }
            
            runButton.disabled = true;
            runButton.textContent = 'Analyzing...';
            resultDiv.innerHTML = '<div class="result loading">🔍 Analyzing chest X-ray... This may take a few seconds.</div>';
            
            try {
                const response = await fetch('/run-main', {
                    method: 'POST'
                });
                const data = await response.json();
                
                if (response.ok && data.success) {
                    let resultHtml = '<div class="result success">✅ Analysis Complete!</div>';
                    
                    // Display analysis results if available
                    if (data.analysis && data.analysis.interpretation) {
                        resultHtml += '<h3>📊 Analysis Results:</h3>';
                        
                        // Display target diseases
                        if (data.analysis.interpretation.target_diseases) {
                            resultHtml += '<h4>Target Diseases:</h4>';
                            resultHtml += '<table class="prediction-table">';
                            resultHtml += '<tr><th>Disease</th><th>Interpretation</th></tr>';
                            
                            for (const [disease, interpretation] of Object.entries(data.analysis.interpretation.target_diseases)) {
                                let colorClass = '';
                                if (interpretation.includes('High')) colorClass = 'high';
                                else if (interpretation.includes('Moderate')) colorClass = 'moderate';
                                else if (interpretation.includes('Low')) colorClass = 'low';
                                else colorClass = 'negative';
                                
                                resultHtml += `<tr>
                                    <td><strong>${disease}</strong></td>
                                    <td class="${colorClass}">${interpretation}</td>
                                </tr>`;
                            }
                            resultHtml += '</table>';
                        }
                        
                        // Display all predictions
                        if (data.analysis.predictions) {
                            resultHtml += '<h4>All Predictions (Top 10):</h4>';
                            resultHtml += '<table class="prediction-table">';
                            resultHtml += '<tr><th>Finding</th><th>Confidence</th></tr>';
                            
                            const sortedPredictions = Object.entries(data.analysis.predictions)
                                .sort((a, b) => b[1] - a[1])
                                .slice(0, 10);
                            
                            for (const [finding, score] of sortedPredictions) {
                                let colorClass = '';
                                if (score >= 0.7) colorClass = 'high';
                                else if (score >= 0.5) colorClass = 'moderate';
                                else if (score >= 0.3) colorClass = 'low';
                                else colorClass = 'negative';
                                
                                resultHtml += `<tr>
                                    <td>${finding}</td>
                                    <td class="${colorClass}">${(score * 100).toFixed(1)}%</td>
                                </tr>`;
                            }
                            resultHtml += '</table>';
                        }
                        
                        resultHtml += `<div class="image-info">
                            <strong>Image analyzed:</strong> ${data.image_used}
                        </div>`;
                    } else if (data.stdout) {
                        resultHtml += `<pre class="result">${escapeHtml(data.stdout)}</pre>`;
                    }
                    
                    if (data.stderr) {
                        resultHtml += `<h3>⚠️ Errors/Warnings:</h3>`;
                        resultHtml += `<pre class="result error">${escapeHtml(data.stderr)}</pre>`;
                    }
                    
                    resultDiv.innerHTML = resultHtml;
                } else {
                    resultDiv.innerHTML = `<div class="result error">❌ Analysis failed: ${data.error || 'Unknown error'}</div>`;
                    if (data.stderr) {
                        resultDiv.innerHTML += `<pre class="result error">${escapeHtml(data.stderr)}</pre>`;
                    }
                }
            } catch (error) {
                resultDiv.innerHTML = `<div class="result error">❌ Error: ${error.message}</div>`;
            } finally {
                runButton.disabled = false;
                runButton.textContent = 'Analyze Uploaded Image';
            }
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        // Check image status on page load and every 20 seconds
        checkImageStatus();
        setInterval(checkImageStatus, 20000);
    </script>
</body>
</html>
        """
    
    def log_message(self, format, *args):
        """Custom log formatting"""
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {format % args}")

def run_server(port=8080):
    """Run the HTTP server"""
    server_address = ('', port)
    httpd = HTTPServer(server_address, CustomServer)
    print(f"🚀 Chest X-ray Analysis Server running on http://localhost:{port}")
    print("\n📋 Available endpoints:")
    print(f"   GET  /                    - Web interface")
    print(f"   POST /upload             - Upload X-ray image")
    print(f"   POST /run-main           - Run analysis on uploaded image")
    print(f"   GET  /current-image      - Check current uploaded image")
    print("\n✨ Open http://localhost:{} in your browser".format(port))
    print("⚠️  Press Ctrl+C to stop the server\n")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped.")
        httpd.server_close()

if __name__ == '__main__':
    run_server()