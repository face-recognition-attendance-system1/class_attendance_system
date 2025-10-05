  document.querySelectorAll('.fire-btn').forEach(btn => {
      btn.addEventListener('click', function() {
        document.getElementById('fireStudentId').value = this.dataset.studentId;
        new bootstrap.Modal(document.getElementById('fireModal')).show();
      });
    });

    document.querySelectorAll('.delete-btn').forEach(btn => {
      btn.addEventListener('click', function() {
        const studentId = this.dataset.studentId;
        const studentName = this.dataset.studentName;
        document.getElementById('deleteStudentId').value = studentId;
        document.getElementById('deleteStaffName').textContent = `Are you sure you want to permanently delete ${studentName} (ID: ${studentId})?`;
        new bootstrap.Modal(document.getElementById('deleteModal')).show();
      });
    });

    // Confirm Fire
    document.getElementById('confirmFire').addEventListener('click', function() {
      const password = document.getElementById('firePassword').value;
      const studentId = document.getElementById('fireStudentId').value;
      if (!password) {
        alert('Password is required');
        return;
      }
      fetch('/fire_staff', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ student_id: studentId, password: password })
      })
      .then(res => res.json())
      .then(data => {
        if (data.status === 'success') {
          // Close password modal
          bootstrap.Modal.getInstance(document.getElementById('fireModal')).hide();
          // Show success modal
          document.getElementById('successMessage').textContent = 'Staff fired successfully!';
          new bootstrap.Modal(document.getElementById('successModal')).show();
          // Reload after short delay
          setTimeout(() => location.reload(), 2000);
        } else {
          alert(data.message);
        }
      })
      .catch(err => {
        alert('Error: ' + err);
      });
    });

    // Confirm Delete
// Delete button functionality
document.querySelectorAll('.delete-btn').forEach(btn => {
    btn.addEventListener('click', function() {
        const studentId = this.dataset.studentId;
        const studentName = this.dataset.studentName;
        document.getElementById('deleteStudentId').value = studentId;
        document.getElementById('deleteStaffName').textContent = `Are you sure you want to permanently delete ${studentName} (ID: ${studentId})?`;
        new bootstrap.Modal(document.getElementById('deleteModal')).show();
    });
});

// Confirm Delete
document.getElementById('confirmDelete').addEventListener('click', function() {
    const password = document.getElementById('deletePassword').value;
    const studentId = document.getElementById('deleteStudentId').value;
    
    if (!password) {
        alert('Password is required');
        return;
    }
    
    fetch('/delete_staff', {
        method: 'POST',
        headers: { 
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ 
            student_id: studentId, 
            password: password 
        })
    })
    .then(res => {
        if (!res.ok) {
            throw new Error(`HTTP error! status: ${res.status}`);
        }
        return res.json();
    })
    .then(data => {
        if (data.status === 'success') {
            // Close delete modal
            bootstrap.Modal.getInstance(document.getElementById('deleteModal')).hide();
            // Show success modal
            document.getElementById('successMessage').textContent = 'Staff permanently deleted!';
            new bootstrap.Modal(document.getElementById('successModal')).show();
            // Reload after short delay
            setTimeout(() => location.reload(), 2000);
        } else {
            alert('Error: ' + data.message);
        }
    })
    .catch(err => {
        console.error('Delete error:', err);
        alert('Error: Failed to delete staff. Please try again.');
    });
});