# Web Application

Real-time hand gesture recognition web application using ONNX Runtime Web.

## Features

- **Client-Side Inference**: All processing happens in the browser (privacy-first)
- **WebGPU/WebAssembly**: Hardware acceleration with automatic fallback
- **Real-Time**: 60 FPS performance on modern browsers
- **No Server Required**: Static hosting (GitHub Pages, Netlify, etc.)

## Quick Start

### 1. Export Your Model to ONNX

```bash
cd ../..  # Go to repo root
python scripts/export.py --model models/best_model.pt --output-dir apps/web/
```

This creates `gesture_model.onnx` in the web app directory.

### 2. Serve Locally

```bash
# Simple HTTP server
python -m http.server 8000

# Or use Node.js
npx serve .
```

### 3. Open in Browser

Navigate to `http://localhost:8000`

## Deployment

### GitHub Pages

```bash
# Build and copy files
cp apps/web/* docs/

# Push to GitHub
git add docs/
git commit -m "Deploy web app"
git push

# Enable GitHub Pages in repository settings
# Source: docs/ folder
```

### Netlify

1. Drag and drop the `apps/web/` folder to Netlify
2. Or connect your GitHub repo
3. Build command: (none needed, static site)
4. Publish directory: `apps/web/`

### Vercel

```bash
cd apps/web
vercel deploy
```

## Browser Compatibility

| Browser | WebGPU | WebAssembly | Recommended |
|---------|--------|-------------|-------------|
| Chrome 113+ | ✅ | ✅ | ✅ |
| Edge 113+ | ✅ | ✅ | ✅ |
| Firefox | ⏳ | ✅ | ⚠️ |
| Safari | ⏳ | ✅ | ⚠️ |

## Performance

- **WebGPU**: 5-10ms inference
- **WebAssembly**: 15-20ms inference
- **Target FPS**: 60
- **Typical FPS**: 60+ (with WebGPU)

## Customization

### Update Gesture Labels

Edit `app.js`:
```javascript
this.labels = {
    0: 'Thumbs Up',
    1: 'Peace',
    // ... add your gestures
};
```

### Adjust Confidence Threshold

```javascript
// In onHandsResults method
if (prediction.confidence >= 0.7) {  // Adjust threshold
    // Display prediction
}
```

### Change Camera Settings

```javascript
video: {
    width: { ideal: 1280 },  // Adjust resolution
    height: { ideal: 720 },
    facingMode: 'user'  // or 'environment' for back camera
}
```

## Troubleshooting

### Camera Access Denied
- Check browser permissions
- HTTPS required for camera access (except localhost)

### Model Not Loading
- Ensure `gesture_model.onnx` is in the web directory
- Check browser console for errors
- Verify model is exported correctly

### Slow Performance
- Try reducing camera resolution
- Check if WebGPU is available: `await navigator.gpu.requestAdapter()`
- Use quantized model for faster inference

## Security

- All inference is client-side
- No data sent to servers
- Camera access requires user permission
- Works offline after initial load

## Development

```bash
# Watch for changes (requires Node.js)
npm install -g browser-sync
browser-sync start --server --files "*.html, *.js, *.css"
```

## License

Same as main project (MIT)
