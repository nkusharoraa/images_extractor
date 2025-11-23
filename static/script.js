document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('config-form');
    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');
    const logConsole = document.getElementById('log-console');
    const galleryGrid = document.getElementById('gallery-grid');
    const statusIndicator = document.getElementById('status-indicator');
    const tabs = document.querySelectorAll('.tab');
    const tabContents = document.querySelectorAll('.tab-content');

    let logSource = null;
    let galleryInterval = null;

    // Tab Switching
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            tab.classList.add('active');
            document.getElementById(tab.dataset.target).classList.add('active');
        });
    });

    // Start Download
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const data = {
            queries: document.getElementById('queries').value,
            min_width: document.getElementById('min_width').value,
            min_size: document.getElementById('min_size').value,
            suffix: document.getElementById('suffix').value,
            variance: document.getElementById('variance').value,
            validate: document.getElementById('validate').checked
        };

        try {
            const response = await fetch('/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            
            const result = await response.json();
            
            if (response.ok) {
                setRunningState(true);
                addLog('System', 'Download started...');
                startLogStream();
                startGalleryUpdates();
            } else {
                addLog('Error', result.message);
            }
        } catch (err) {
            addLog('Error', 'Failed to start download: ' + err.message);
        }
    });

    // Stop Download
    stopBtn.addEventListener('click', async () => {
        try {
            await fetch('/stop', { method: 'POST' });
            addLog('System', 'Stop signal sent...');
            stopBtn.disabled = true;
        } catch (err) {
            addLog('Error', 'Failed to stop: ' + err.message);
        }
    });

    function setRunningState(isRunning) {
        startBtn.disabled = isRunning;
        stopBtn.disabled = !isRunning;
        statusIndicator.textContent = isRunning ? 'Running...' : 'Ready';
        statusIndicator.classList.toggle('running', isRunning);
        
        if (!isRunning) {
            if (logSource) {
                logSource.close();
                logSource = null;
            }
            if (galleryInterval) {
                clearInterval(galleryInterval);
                galleryInterval = null;
            }
            // One final gallery update
            updateGallery();
        }
    }

    function startLogStream() {
        if (logSource) logSource.close();
        
        logSource = new EventSource('/logs');
        
        logSource.onmessage = (event) => {
            const message = event.data;
            if (message === 'DONE') {
                addLog('System', 'Process completed.');
                setRunningState(false);
            } else if (message.startsWith('ERROR:')) {
                addLog('Error', message.substring(7));
                setRunningState(false);
            } else {
                addLog('Info', message);
            }
        };
        
        logSource.onerror = () => {
            // Connection lost (or closed), usually fine to just close
            // logSource.close();
        };
    }

    function addLog(type, message) {
        const line = document.createElement('div');
        line.className = 'log-line';
        if (type === 'Error') line.classList.add('error');
        if (type === 'System') line.classList.add('success');
        
        const time = new Date().toLocaleTimeString();
        line.textContent = `[${time}] ${message}`;
        
        logConsole.appendChild(line);
        logConsole.scrollTop = logConsole.scrollHeight;
    }

    function startGalleryUpdates() {
        if (galleryInterval) clearInterval(galleryInterval);
        updateGallery(); // Initial call
        galleryInterval = setInterval(updateGallery, 5000); // Poll every 5s
    }

    async function updateGallery() {
        try {
            const response = await fetch('/gallery');
            const images = await response.json();
            
            // Simple diffing could be better, but clearing is safer for now
            galleryGrid.innerHTML = '';
            
            images.forEach(src => {
                const div = document.createElement('div');
                div.className = 'gallery-item';
                const img = document.createElement('img');
                img.src = src;
                img.loading = 'lazy';
                div.appendChild(img);
                galleryGrid.appendChild(div);
            });
        } catch (err) {
            console.error('Gallery update failed', err);
        }
    }
});
