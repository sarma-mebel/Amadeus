# Amadeus Frontend

Modern React + TypeScript frontend for the Amadeus Kurisu assistant with side-by-side 3D VRM model and chat interface.

## Features

- **3D VRM Viewer** (Left panel, 55%): Three.js-powered VRM model viewer with Kurisu
  - Smooth animations (breathing, blinking, eye tracking)
  - Environment support (Time Machine, Future Gadget Lab)
  - Interactive camera controls

- **Chat Interface** (Right panel, 45%): Clean messaging UI
  - Real-time message streaming
  - Voice synthesis playback
  - Session persistence
  - Mobile-responsive (stacks vertically on small screens)

## Setup

### Prerequisites

- Node.js 18+ and npm
- Running Amadeus FastAPI backend on `localhost:8000`

### Installation

\`\`\`bash
cd frontend
npm install
\`\`\`

### Development

Start the dev server with hot reload:

\`\`\`bash
npm run dev
\`\`\`

The app will be available at `http://localhost:5173` and automatically proxy API requests to the FastAPI backend at `http://localhost:8000`.

### Build for Production

\`\`\`bash
npm run build
\`\`\`

The build output will be in the `dist/` directory.

### Environment Variables

The frontend accepts the following Vite environment variables:

- `VITE_API_BASE_URL` – optional absolute base URL for API requests. Leave unset to keep relative calls (useful for local development or Capacitor when the backend shares the same origin).
- `VITE_ASSET_BASE_URL` – optional absolute base URL for static assets (VRM model, animations, environments). Defaults to `VITE_API_BASE_URL` when not provided.

## Project Structure

\`\`\`
frontend/
├── src/
│   ├── components/
│   │   ├── VRMViewer.tsx         # 3D VRM model viewer
│   │   ├── VRMViewer.css
│   │   ├── ChatInterface.tsx      # Chat UI
│   │   └── ChatInterface.css
│   ├── App.tsx                    # Main layout (side-by-side)
│   ├── App.css
│   ├── main.tsx                   # Entry point
│   └── index.css
├── vite.config.ts                 # Vite config with proxy
└── package.json
\`\`\`

## API Integration

The frontend connects to these FastAPI endpoints:

- `POST /api/chat` - Send messages and receive responses
- `GET /static/models/Kurisu.vrm` - Load VRM model
- `GET /static/animations/*.vrma` - Load VRMA animations
- `GET /static/environment/*.glb` - Load 3D environments

## Usage

1. Start the FastAPI backend:
   \`\`\`bash
   cd ..  # Go to Amadeus root directory
   uvicorn amadeus.web.app:app --reload
   \`\`\`

2. Start the frontend dev server:
   \`\`\`bash
   cd frontend
   npm run dev
   \`\`\`

3. Open `http://localhost:5173` in your browser

4. Start chatting with Kurisu! The 3D model will animate on the left while you interact via the chat panel on the right.

## Customization

### Changing Layout Proportions

Edit `src/App.css`:

\`\`\`css
.vrm-panel {
  flex: 0 0 60%;  /* Change from 55% to 60% */
}
\`\`\`

### Adding Assistant-UI Components

The current implementation uses a custom chat interface. To integrate `@assistant-ui/react`:

1. The package is already included in dependencies

2. Replace `ChatInterface.tsx` with assistant-ui components - see the [assistant-ui documentation](https://github.com/assistant-ui/assistant-ui) for implementation details

## Tech Stack

- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server
- **Three.js** - 3D rendering
- **@pixiv/three-vrm** - VRM model support
- **@assistant-ui/react** - Chat UI components (ready to integrate)
- **Capacitor** - Native shell for iOS/Android packaging

## Troubleshooting

### VRM model not loading

- Ensure the FastAPI backend is running on `localhost:8000`
- Check that `/static/models/Kurisu.vrm` exists in the parent directory
- Open browser console for error messages

### Chat not working

- Verify the backend API is accessible at `http://localhost:8000/api/chat`
- Check browser console for network errors
- Ensure CORS is properly configured in FastAPI

### Port already in use

Change the port in `package.json`:

\`\`\`json
"dev": "vite --port 3000"
\`\`\`

## Mobile Builds (Capacitor)

Capacitor configuration is included so the Vite bundle can ship inside native shells.

1. Install the native dependencies (requires network access):
   \`\`\`bash
   npm install @capacitor/core @capacitor/cli @capacitor/android @capacitor/ios
   \`\`\`
   Running the install will update `package-lock.json`.

2. Build and sync the web assets into the native projects:
   \`\`\`bash
   npm run cap:sync
   \`\`\`

3. Open the projects in native IDEs:
   \`\`\`bash
   npm run cap:open:android   # Android Studio
   npm run cap:open:ios       # Xcode
   \`\`\`

4. Set `VITE_API_BASE_URL` (for example `https://api.example.com`) before producing release builds so the mobile clients target the correct backend.

Re-run `npm run cap:sync` whenever the frontend is rebuilt so the native projects pick up the latest `dist/` output.

### Mobile Development Helper Scripts

- `./scripts/start_mobile.sh` – starts the FastAPI backend (logs to `logs/backend-mobile.log`) and applies `adb reverse tcp:8000 tcp:8000` so the Android emulator can reach `http://127.0.0.1:8000`.
- `./scripts/stop_mobile.sh` – stops the background backend process and removes the `adb reverse` tunnel when you are done testing.
