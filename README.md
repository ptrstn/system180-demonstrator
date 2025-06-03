# Webcam Viewer

A FastAPI application with Jinja2 templates for viewing multiple webcams simultaneously.

## Features

- Displays three webcams:
  - 2x OAK-1 MAX AI cameras
  - 1x OBSBOT Meet 2 4K Webcam
- Asynchronous video streaming for optimal performance
- Configurable camera settings via environment variables
- Clean, responsive user interface
- Real-time status monitoring

## Requirements

- Python 3.8+
- USB-connected webcams:
  - 2x OAK-1 MAX cameras
  - 1x OBSBOT Meet 2 4K Webcam (or other standard webcam)

## Installation

1. Clone this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Camera settings can be configured by editing the `.env` file. Available settings include:

- Resolution
- Frame rate
- Device IDs
- Stream quality

## Running the Application

Start the application with:

```bash
python -m app.main
```

Or use the NPM script:

```bash
npm run start
```

Then open your browser to http://localhost:8000

## Camera Identification

If you have multiple OAK-1 MAX cameras, you can specify their MX IDs in the `.env` file to ensure consistent identification of left and right cameras.