import os
import threading
import queue
import time
import logging
from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from download_images import download_images, DOWNLOAD_DIR

app = Flask(__name__)

# Global state
log_queue = queue.Queue()
stop_event = threading.Event()
download_thread = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_download():
    global download_thread, stop_event
    
    if download_thread and download_thread.is_alive():
        return jsonify({"status": "error", "message": "Download already running"}), 400
    
    data = request.json
    queries = [q.strip() for q in data.get('queries', '').split(',') if q.strip()]
    min_width = int(data.get('min_width', 1000))
    min_size = int(data.get('min_size', 50))
    suffix = data.get('suffix', 'car studio background')
    validate = data.get('validate', True)
    variance = int(data.get('variance', 500))
    
    stop_event.clear()
    
    def run_wrapper():
        try:
            download_images(
                queries=queries,
                min_width=min_width,
                min_file_size_kb=min_size,
                search_suffix=suffix,
                validate_car=validate,
                variance_threshold=variance,
                stop_event=stop_event,
                log_queue=log_queue
            )
            log_queue.put("DONE")
        except Exception as e:
            log_queue.put(f"ERROR: {str(e)}")

    download_thread = threading.Thread(target=run_wrapper)
    download_thread.start()
    
    return jsonify({"status": "success", "message": "Download started"})

@app.route('/stop', methods=['POST'])
def stop_download():
    global stop_event
    stop_event.set()
    return jsonify({"status": "success", "message": "Stopping download..."})

@app.route('/logs')
def stream_logs():
    def generate():
        while True:
            try:
                message = log_queue.get(timeout=1)
                yield f"data: {message}\n\n"
            except queue.Empty:
                # Send a keep-alive comment to keep the connection open
                yield ": keep-alive\n\n"
    return Response(generate(), mimetype='text/event-stream')

@app.route('/gallery')
def gallery():
    # Find the most recent run directory
    if not os.path.exists(DOWNLOAD_DIR):
        return jsonify([])
        
    subdirs = [os.path.join(DOWNLOAD_DIR, d) for d in os.listdir(DOWNLOAD_DIR) if os.path.isdir(os.path.join(DOWNLOAD_DIR, d)) and d.startswith("run_")]
    if not subdirs:
        return jsonify([])
        
    latest_run = max(subdirs, key=os.path.getmtime)
    images = [f for f in os.listdir(latest_run) if f.endswith(('.jpg', '.png'))]
    
    # Return relative paths for serving
    run_name = os.path.basename(latest_run)
    image_urls = [f"/images/{run_name}/{img}" for img in images]
    return jsonify(image_urls)

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(DOWNLOAD_DIR, filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
