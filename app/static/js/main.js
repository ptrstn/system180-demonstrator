document.addEventListener('DOMContentLoaded', function() {
    // Update camera status periodically
    function updateCameraStatus() {
        fetch('/status')
            .then(response => response.json())
            .then(data => {
                // Update OAK Left status
                const oakLeftStatus = document.getElementById('oak-left-status');
                const oakLeftFps = document.getElementById('oak-left-fps');
                
                if (data.oak_left.connected) {
                    oakLeftStatus.className = 'inline-block w-3 h-3 rounded-full mr-2 bg-green-500';
                    oakLeftFps.textContent = Math.round(data.oak_left.fps) + ' FPS';
                } else {
                    oakLeftStatus.className = 'inline-block w-3 h-3 rounded-full mr-2 bg-red-500';
                    oakLeftFps.textContent = 'Disconnected';
                }
                
                // Update OBSBOT status
                const obsbotStatus = document.getElementById('obsbot-status');
                const obsbotFps = document.getElementById('obsbot-fps');
                
                if (data.obsbot.connected) {
                    obsbotStatus.className = 'inline-block w-3 h-3 rounded-full mr-2 bg-green-500';
                    obsbotFps.textContent = Math.round(data.obsbot.fps) + ' FPS';
                } else {
                    obsbotStatus.className = 'inline-block w-3 h-3 rounded-full mr-2 bg-red-500';
                    obsbotFps.textContent = 'Disconnected';
                }
                
                // Update OAK Right status
                const oakRightStatus = document.getElementById('oak-right-status');
                const oakRightFps = document.getElementById('oak-right-fps');
                
                if (data.oak_right.connected) {
                    oakRightStatus.className = 'inline-block w-3 h-3 rounded-full mr-2 bg-green-500';
                    oakRightFps.textContent = Math.round(data.oak_right.fps) + ' FPS';
                } else {
                    oakRightStatus.className = 'inline-block w-3 h-3 rounded-full mr-2 bg-red-500';
                    oakRightFps.textContent = 'Disconnected';
                }
            })
            .catch(error => {
                console.error('Error fetching camera status:', error);
            });
    }
    
    // Update status initially and then every 5 seconds
    updateCameraStatus();
    setInterval(updateCameraStatus, 5000);
    
    // Handle stream errors
    const streams = [
        document.getElementById('oak-left-stream'),
        document.getElementById('obsbot-stream'),
        document.getElementById('oak-right-stream')
    ];
    
    streams.forEach(stream => {
        if (stream) {
            stream.onerror = function() {
                // If stream errors, retry after 3 seconds
                setTimeout(() => {
                    if (stream.src) {
                        stream.src = stream.src + '?' + new Date().getTime();
                    }
                }, 3000);
            };
        }
    });
});