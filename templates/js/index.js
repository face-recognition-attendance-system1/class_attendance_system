let intervalId;
function startCamera() {
    const videoContainer = document.getElementById("videoContainer");
    const videoImg = document.getElementById("videoFeed");
    const stopCameraBtn = document.getElementById("stopCameraBtn");
    videoContainer.style.display = "block";
    stopCameraBtn.style.display = "inline-block";
    videoImg.src = "{{ url_for('register_video_feed') }}";
    intervalId = setInterval(checkStatus, 1000);
}

function stopCamera() {
    const videoContainer = document.getElementById("videoContainer");
    const videoImg = document.getElementById("videoFeed");
    const stopCameraBtn = document.getElementById("stopCameraBtn");
    const registerForm = document.getElementById("registerForm");
    const statusElement = document.getElementById("status");

    videoImg.src = "";
    videoContainer.style.display = "none";
    stopCameraBtn.style.display = "none";
    registerForm.style.display = "none";
    statusElement.innerText = "Detecting...";
    statusElement.className = "status-indicator status-detecting";
    clearInterval(intervalId);

    fetch("/stop_camera", {
    method: "POST"
    })
    .then(res => res.json())
    .then(data => {
    console.log(data.message);
    })
    .catch(err => {
    console.error("Error stopping camera:", err);
    alert("Failed to stop camera. Please try again.");
    });
}

function checkStatus() {
    fetch("/recognize_status")
    .then(res => res.json())
    .then(data => {
        const statusElement = document.getElementById("status");
        const registerForm = document.getElementById("registerForm");
        
        if (data.name === "Unknown") {
        statusElement.innerText = "Unknown Face Detected";
        statusElement.className = "status-indicator status-unknown";
        registerForm.style.display = "block";
        } else if (data.name === "No Face") {
        statusElement.innerText = "No Face Detected";
        statusElement.className = "status-indicator status-detecting";
        registerForm.style.display = "none";
        } else if (data.name === "Multiple Faces") {
        statusElement.innerText = "Multiple Faces Detected";
        statusElement.className = "status-indicator status-multiple";
        registerForm.style.display = "none";
        } else {
        statusElement.innerText = "Known Face: " + data.name;
        statusElement.className = "status-indicator status-recognized";
        registerForm.style.display = "none";
        }
    })
    .catch(err => {
        console.error("Error checking status:", err);
        const statusElement = document.getElementById("status");
        statusElement.innerText = "Error Detecting Face";
        statusElement.className = "status-indicator status-error";
        document.getElementById("registerForm").style.display = "none";
    });
}

function showPasswordModal() {
    const studentId = document.getElementById("student_id").value.trim();
    const name = document.getElementById("name").value.trim();
    const email = document.getElementById("email").value.trim();
    const phone = document.getElementById("phone").value.trim();
    const department = document.getElementById("department").value.trim();

    if (!studentId || !name || !email || !phone || !department) {
    alert("Please fill in all fields");
    return;
    }

    // Show password confirmation modal
    const passwordModal = new bootstrap.Modal(document.getElementById('passwordModal'));
    passwordModal.show();
}

// Confirm Registration with Password
document.getElementById('confirmRegistration').addEventListener('click', function() {
    const password = document.getElementById('adminPassword').value;
    
    if (!password) {
    alert('Password is required');
    return;
    }

    const registerBtn = document.getElementById("registerBtn");
    registerBtn.disabled = true;
    registerBtn.innerText = "Registering...";

    const studentId = document.getElementById("student_id").value.trim();
    const name = document.getElementById("name").value.trim();
    const email = document.getElementById("email").value.trim();
    const phone = document.getElementById("phone").value.trim();
    const department = document.getElementById("department").value.trim();

    const payload = {
    student_id: studentId,
    name: name,
    email: email,
    phone: phone,
    department: department,
    password: password  // Include password in the payload
    };

    fetch("/register_unknown", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
    })
    .then(res => {
    if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
    }
    return res.json();
    })
    .then(data => {
    if (data.status === "error") {
        alert(data.message);
        registerBtn.disabled = false;
        registerBtn.innerText = "Register User";
        return;
    }
    
    // Close password modal
    bootstrap.Modal.getInstance(document.getElementById('passwordModal')).hide();
    
    // Show success modal
    document.getElementById("successMessage").textContent = data.message;
    const successModal = new bootstrap.Modal(document.getElementById('successModal'));
    successModal.show();
    
    // Reset form
    document.getElementById("student_id").value = "";
    document.getElementById("name").value = "";
    document.getElementById("email").value = "";
    document.getElementById("phone").value = "";
    document.getElementById("department").value = "";
    document.getElementById("adminPassword").value = "";
    
    // Stop camera after successful registration
    setTimeout(() => {
        stopCamera();
        registerBtn.disabled = false;
        registerBtn.innerText = "Register User";
    }, 2000);
    })
    .catch(err => {
    console.error("Registration error:", err);
    alert("Registration failed: " + err.message);
    registerBtn.disabled = false;
    registerBtn.innerText = "Register User";
    });
});

// Clear password field when modal is closed
document.getElementById('passwordModal').addEventListener('hidden.bs.modal', function () {
    document.getElementById('adminPassword').value = '';
})

