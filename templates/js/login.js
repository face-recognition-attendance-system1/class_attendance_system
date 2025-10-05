 document.addEventListener('DOMContentLoaded', function() {
      const inputs = document.querySelectorAll('.form-control');
      inputs.forEach(input => {
        input.addEventListener('focus', function() {
          this.parentElement.classList.add('focused');
        });
        input.addEventListener('blur', function() {
          this.parentElement.classList.remove('focused');
        });
      });
    });