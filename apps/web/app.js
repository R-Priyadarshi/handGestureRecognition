// Hand Gesture Recognition Web App
// Uses ONNX Runtime Web with WebGPU/WebAssembly backend

class GestureRecognizer {
    constructor() {
        this.session = null;
        this.hands = null;
        this.camera = null;
        this.isRunning = false;
        
        // Stats
        this.detectionCount = 0;
        this.frameCount = 0;
        this.lastFpsUpdate = Date.now();
        this.latencies = [];
        
        // Gesture labels
        this.labels = {
            0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F',
            6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
            12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R',
            18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X',
            24: 'Y', 25: 'Z'
        };
        
        this.init();
    }
    
    async init() {
        try {
            await this.loadModel();
            await this.initializeHands();
            this.updateStatus('ready', 'Ready! Click "Start Camera" to begin');
        } catch (error) {
            console.error('Initialization error:', error);
            this.updateStatus('error', `Error: ${error.message}`);
        }
    }
    
    async loadModel() {
        this.updateStatus('loading', 'Loading ONNX model...');
        
        try {
            // Try WebGPU first, fall back to WebAssembly
            this.session = await ort.InferenceSession.create('gesture_model.onnx', {
                executionProviders: ['webgpu', 'wasm']
            });
            
            console.log('Model loaded successfully');
            console.log('Using provider:', this.session.executionProviders);
        } catch (error) {
            throw new Error(`Failed to load model: ${error.message}`);
        }
    }
    
    async initializeHands() {
        this.updateStatus('loading', 'Initializing MediaPipe Hands...');
        
        this.hands = new Hands({
            locateFile: (file) => {
                return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
            }
        });
        
        this.hands.setOptions({
            maxNumHands: 1,
            modelComplexity: 1,
            minDetectionConfidence: 0.7,
            minTrackingConfidence: 0.5
        });
        
        this.hands.onResults((results) => this.onHandsResults(results));
    }
    
    async startCamera() {
        const video = document.getElementById('videoElement');
        
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                }
            });
            
            video.srcObject = stream;
            
            this.camera = new Camera(video, {
                onFrame: async () => {
                    await this.hands.send({ image: video });
                },
                width: 1280,
                height: 720
            });
            
            await this.camera.start();
            this.isRunning = true;
            
            document.getElementById('startBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;
            
            this.updateFPS();
        } catch (error) {
            console.error('Camera error:', error);
            this.updateStatus('error', 'Failed to access camera');
        }
    }
    
    stopCamera() {
        if (this.camera) {
            this.camera.stop();
        }
        
        const video = document.getElementById('videoElement');
        if (video.srcObject) {
            video.srcObject.getTracks().forEach(track => track.stop());
        }
        
        this.isRunning = false;
        
        document.getElementById('startBtn').disabled = false;
        document.getElementById('stopBtn').disabled = true;
        document.getElementById('predictionText').textContent = '-';
        document.getElementById('confidenceText').textContent = 'Camera stopped';
    }
    
    async onHandsResults(results) {
        this.frameCount++;
        
        if (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0) {
            document.getElementById('predictionText').textContent = '-';
            document.getElementById('confidenceText').textContent = 'No hand detected';
            return;
        }
        
        const startTime = performance.now();
        
        // Extract and normalize landmarks
        const landmarks = this.extractLandmarks(results.multiHandLandmarks[0]);
        const normalized = this.normalizeLandmarks(landmarks);
        
        // Run inference
        const prediction = await this.predict(normalized);
        
        const latency = performance.now() - startTime;
        this.latencies.push(latency);
        if (this.latencies.length > 30) this.latencies.shift();
        
        // Update UI
        this.displayPrediction(prediction, latency);
        this.detectionCount++;
    }
    
    extractLandmarks(handLandmarks) {
        // Extract x, y coordinates (ignore z for 2D model)
        const coords = [];
        for (const landmark of handLandmarks) {
            coords.push(landmark.x, landmark.y, landmark.z);
        }
        return coords;
    }
    
    normalizeLandmarks(landmarks) {
        // Extract only x, y (21 landmarks * 2 = 42 features)
        const coords2d = [];
        for (let i = 0; i < landmarks.length; i += 3) {
            coords2d.push(landmarks[i], landmarks[i + 1]);
        }
        
        // Simple min-max normalization
        const xCoords = coords2d.filter((_, i) => i % 2 === 0);
        const yCoords = coords2d.filter((_, i) => i % 2 === 1);
        
        const minX = Math.min(...xCoords);
        const minY = Math.min(...yCoords);
        
        const normalized = [];
        for (let i = 0; i < coords2d.length; i += 2) {
            normalized.push(coords2d[i] - minX);
            normalized.push(coords2d[i + 1] - minY);
        }
        
        return new Float32Array(normalized);
    }
    
    async predict(landmarks) {
        // Create tensor
        const inputTensor = new ort.Tensor('float32', landmarks, [1, 42]);
        
        // Run inference
        const feeds = { input: inputTensor };
        const results = await this.session.run(feeds);
        
        // Get output
        const output = results.output.data;
        
        // Apply softmax
        const exp = Array.from(output).map(x => Math.exp(x));
        const sumExp = exp.reduce((a, b) => a + b, 0);
        const probs = exp.map(x => x / sumExp);
        
        // Get prediction
        const classId = probs.indexOf(Math.max(...probs));
        const confidence = probs[classId];
        
        return {
            classId,
            confidence,
            gesture: this.labels[classId] || '?'
        };
    }
    
    displayPrediction(prediction, latency) {
        const predText = document.getElementById('predictionText');
        const confText = document.getElementById('confidenceText');
        
        predText.textContent = prediction.gesture;
        confText.textContent = `Confidence: ${(prediction.confidence * 100).toFixed(1)}%`;
        
        // Color based on confidence
        if (prediction.confidence >= 0.8) {
            predText.style.color = '#28a745';
        } else if (prediction.confidence >= 0.6) {
            predText.style.color = '#ffc107';
        } else {
            predText.style.color = '#6c757d';
        }
        
        // Update latency
        const avgLatency = this.latencies.reduce((a, b) => a + b, 0) / this.latencies.length;
        document.getElementById('latencyValue').textContent = avgLatency.toFixed(1);
    }
    
    updateFPS() {
        if (!this.isRunning) return;
        
        const now = Date.now();
        const elapsed = (now - this.lastFpsUpdate) / 1000;
        
        if (elapsed >= 1.0) {
            const fps = this.frameCount / elapsed;
            document.getElementById('fpsValue').textContent = fps.toFixed(0);
            document.getElementById('detectionValue').textContent = this.detectionCount;
            
            this.frameCount = 0;
            this.lastFpsUpdate = now;
        }
        
        requestAnimationFrame(() => this.updateFPS());
    }
    
    updateStatus(type, message) {
        const status = document.getElementById('status');
        status.className = `status ${type}`;
        status.textContent = message;
    }
}

// Initialize app
let recognizer;

document.addEventListener('DOMContentLoaded', () => {
    recognizer = new GestureRecognizer();
    
    document.getElementById('startBtn').addEventListener('click', () => {
        recognizer.startCamera();
    });
    
    document.getElementById('stopBtn').addEventListener('click', () => {
        recognizer.stopCamera();
    });
});
