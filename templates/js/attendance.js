// Format ISO timestamp to readable format
function formatDateTime(isoString) {
    if (!isoString) return '';
    const date = new Date(isoString);
    return date.toLocaleString('en-US', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit'
    });
}

// Update attendance tables and stats
function updateAttendance() {
    const year = document.getElementById('year').value;
    const month = document.getElementById('month').value;
    const day = document.getElementById('day').value;
    const loadingIndicator = document.getElementById('loading-indicator');
    
    loadingIndicator.style.display = 'block';
    
    fetch(`/attendance_data?year=${year}&month=${month}&day=${day}`)
    .then(response => {
        if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        // Update stats
        document.getElementById('total-staff').textContent = data.stats.total_registered;
        document.getElementById('present-today').textContent = data.stats.present_today;
        document.getElementById('late-today').textContent = data.stats.late_today;
        document.getElementById('absent-today').textContent = data.stats.absent_today;

        // Update present table
        const presentTableContainer = document.getElementById('present-table-container');
        if (data.present.length > 0) {
        let table = `
            <table class="table" id="present-table">
            <thead>
                <tr>
                <th>ID</th>
                <th>Name</th>
                <th>Email</th>
                <th>Phone</th>
                <th>Department</th>
                <th>Timestamp</th>
                <th>Admin Name</th>
                </tr>
            </thead>
            <tbody>
        `;
        data.present.forEach(s => {
            table += `
            <tr>
                <td>${s.student_id}</td>
                <td>${s.name}</td>
                <td>${s.email}</td>
                <td>${s.phone}</td>
                <td>${s.department || ''}</td>
                <td>${formatDateTime(s.timestamp)}</td>
                <td>${s.admin_name || 'Unknown'}</td>
            </tr>
            `;
        });
        table += `</tbody></table>`;
        presentTableContainer.innerHTML = table;
        } else {
        presentTableContainer.innerHTML = '<p>No staff present on this date.</p>';
        }

        // Update late table
        const lateTableContainer = document.getElementById('late-table-container');
        if (data.late.length > 0) {
        let table = `
            <table class="table" id="late-table">
            <thead>
                <tr>
                <th>ID</th>
                <th>Name</th>
                <th>Email</th>
                <th>Phone</th>
                <th>Department</th>
                <th>Timestamp</th>
                <th>Admin Name</th>
                </tr>
            </thead>
            <tbody>
        `;
        data.late.forEach(s => {
            table += `
            <tr>
                <td>${s.student_id}</td>
                <td>${s.name}</td>
                <td>${s.email}</td>
                <td>${s.phone}</td>
                <td>${s.department || ''}</td>
                <td>${formatDateTime(s.timestamp)}</td>
                <td>${s.admin_name || 'Unknown'}</td>
            </tr>
            `;
        });
        table += `</tbody></table>`;
        lateTableContainer.innerHTML = table;
        } else {
        lateTableContainer.innerHTML = '<p>No staff late on this date.</p>';
        }

        // Update absent table
        const absentTableContainer = document.getElementById('absent-table-container');
        if (data.absent.length > 0) {
        let table = `
            <table class="table" id="absent-table">
            <thead>
                <tr>
                <th>ID</th>
                <th>Name</th>
                <th>Email</th>
                <th>Phone</th>
                <th>Department</th>
                <th>Admin Name</th>
                </tr>
            </thead>
            <tbody>
        `;
        data.absent.forEach(s => {
            table += `
            <tr>
                <td>${s.student_id}</td>
                <td>${s.name}</td>
                <td>${s.email}</td>
                <td>${s.phone}</td>
                <td>${s.department || ''}</td>
                <td>${s.admin_name || 'Unknown'}</td>
            </tr>
            `;
        });
        table += `</tbody></table>`;
        absentTableContainer.innerHTML = table;
        } else {
        absentTableContainer.innerHTML = '<p>No staff absent on this date.</p>';
        }

        loadingIndicator.style.display = 'none';
    })
    .catch(error => {
        console.error('Error fetching attendance data:', error);
        loadingIndicator.style.display = 'none';
        alert('Failed to update attendance data. Please check your connection and try again.');
    });
}

// Initialize page
window.onload = function() {
    generateFaceGrid();
    const form = document.getElementById('date-filter-form');
    form.addEventListener('change', updateAttendance);

    // Periodically check video feed status
    const videoFeed = document.getElementById('video-feed');
    const videoError = document.getElementById('video-error');
    videoFeed.onerror = function() {
    videoFeed.style.display = 'none';
    videoError.style.display = 'block';
    };
    videoFeed.onload = function() {
    videoFeed.style.display = 'block';
    videoError.style.display = 'none';
    };

    // Poll attendance data every 5 seconds for real-time updates
    updateAttendance(); // Initial fetch
    setInterval(updateAttendance, 5000); // Poll every 5 seconds
};

// Handle window resize to regenerate face grid
window.onresize = function() {
    generateFaceGrid();
};