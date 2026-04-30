document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const browseBtn = document.getElementById('browseBtn');
    
    const previewContainer = document.getElementById('previewContainer');
    const imagePreview = document.getElementById('imagePreview');
    const clearBtn = document.getElementById('clearBtn');
    const analyzeBtn = document.getElementById('analyzeBtn');
    
    const loadingState = document.getElementById('loadingState');
    const resultsSection = document.getElementById('resultsSection');
    
    // Result Elements
    const predClass = document.getElementById('predClass');
    const predConf = document.getElementById('predConf');
    const confBarFill = document.querySelector('.conf-bar-fill');
    
    let currentFile = null;

    // --- Drag and Drop Logic ---
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    });

    // --- File Selection Logic ---
    browseBtn.addEventListener('click', () => {
        fileInput.click();
    });

    dropZone.addEventListener('click', (e) => {
        if(e.target !== browseBtn) {
            fileInput.click();
        }
    });

    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });

    function handleFiles(files) {
        if (files.length === 0) return;
        
        const file = files[0];
        
        // Basic image validation
        if (!file.type.startsWith('image/')) {
            alert('Please select an image file (JPEG/PNG).');
            return;
        }

        currentFile = file;
        const reader = new FileReader();
        
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            showPreview();
        };
        
        reader.readAsDataURL(file);
    }

    // --- UI State Management ---
    function showPreview() {
        dropZone.classList.add('hidden');
        previewContainer.classList.remove('hidden');
        loadingState.classList.add('hidden');
        resetResults();
    }

    function clearImage() {
        currentFile = null;
        fileInput.value = '';
        imagePreview.src = '';
        
        previewContainer.classList.add('hidden');
        dropZone.classList.remove('hidden');
        loadingState.classList.add('hidden');
        resetResults();
    }

    function resetResults() {
        resultsSection.classList.add('disabled');
        // Reset styles and text just in case
        predClass.textContent = '---';
        predClass.className = 'badge';
        predConf.textContent = '0%';
        confBarFill.style.width = '0%';
        confBarFill.className = 'conf-bar-fill';
    }

    clearBtn.addEventListener('click', clearImage);

    // --- Backend API Analysis ---
    analyzeBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        // Hide preview buttons and show loading
        analyzeBtn.disabled = true;
        clearBtn.disabled = true;
        
        previewContainer.classList.add('hidden');
        loadingState.classList.remove('hidden');

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const response = await fetch('http://localhost:8000/analyze', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.statusText}`);
            }

            const data = await response.json();
            displayResults(data);

        } catch (error) {
            console.error('Analysis failed:', error);
            alert('Failed to connect to the backend API. Is the FastAPI server running?');
            
            // Revert UI on error
            loadingState.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            analyzeBtn.disabled = false;
            clearBtn.disabled = false;
        }
    });

    function displayResults(data) {
        // Hide loading, show preview actions back
        loadingState.classList.add('hidden');
        previewContainer.classList.remove('hidden');
        analyzeBtn.disabled = false;
        clearBtn.disabled = false;

        // Enable results section
        resultsSection.classList.remove('disabled');

        if (data.is_malignant) {
            predClass.textContent = data.class_name;
            predClass.className = 'badge badge-warning';
            confBarFill.className = 'conf-bar-fill warning';
        } else {
            predClass.textContent = data.class_name;
            predClass.className = 'badge badge-success';
            confBarFill.className = 'conf-bar-fill success';
        }

        predConf.textContent = data.confidence + '%';
        // Small delay to allow CSS transition to work on width
        setTimeout(() => {
            confBarFill.style.width = data.confidence + '%';
        }, 50);

        // Update the actual images returned from the backend API
        document.getElementById('resOriginal').src = imagePreview.src;
        document.getElementById('resGradcam').src = data.maps.gradcam;
        document.getElementById('resLime').src = data.maps.lime;
        document.getElementById('resShap').src = data.maps.shap;
        
        // Scroll to results smoothly
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
});
