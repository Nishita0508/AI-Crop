const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const browseBtn = document.getElementById('browseBtn');
const previewBox = document.getElementById('previewBox');
const previewImage = document.getElementById('previewImage');
const previewName = document.getElementById('previewName');
const predictBtn = document.getElementById('predictBtn');
const statusText = document.getElementById('statusText');
const resultCard = document.getElementById('resultCard');
const detailsGrid = document.getElementById('detailsGrid');

const diseaseNameEl = document.getElementById('diseaseName');
const confidenceValueEl = document.getElementById('confidenceValue');
const statusValueEl = document.getElementById('statusValue');
const actionValueEl = document.getElementById('actionValue');
const preventionValueEl = document.getElementById('preventionValue');

let selectedFile = null;

function showPreview(file) {
  const url = URL.createObjectURL(file);
  previewImage.src = url;
  previewName.textContent = file.name;
  previewBox.classList.remove('hidden');
  selectedFile = file;
}

browseBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (e) => {
  if (e.target.files && e.target.files[0]) showPreview(e.target.files[0]);
});

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  if (e.dataTransfer.files && e.dataTransfer.files[0]) showPreview(e.dataTransfer.files[0]);
});

predictBtn.addEventListener('click', async () => {
  if (!selectedFile) {
    statusText.textContent = 'Please choose an image before running the prediction.';
    return;
  }

  statusText.textContent = 'Analyzing image...';
  predictBtn.disabled = true;
  resultCard.classList.remove('hidden');
  resultCard.innerHTML = '<h3>Analyzing your image…</h3><p>Please wait while the TensorFlow model inspects the leaf.</p>';
  detailsGrid.classList.add('hidden');

  const formData = new FormData();
  formData.append('image', selectedFile);

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || 'Prediction failed');
    }

    resultCard.classList.add('hidden');
    detailsGrid.classList.remove('hidden');
    resultCard.innerHTML = '';

    diseaseNameEl.textContent = data.disease;
    confidenceValueEl.textContent = `${data.confidence}%`;
    statusValueEl.textContent = data.status;
    actionValueEl.textContent = data.recommended_action;
    preventionValueEl.textContent = data.prevention;
    statusText.textContent = `Prediction complete with ${data.confidence}% confidence.`;
  } catch (error) {
    resultCard.classList.remove('hidden');
    resultCard.innerHTML = `<h3>Prediction error</h3><p>${error.message}</p>`;
    detailsGrid.classList.add('hidden');
    statusText.textContent = error.message;
    statusText.style.color = '#ffd4d4';
  } finally {
    predictBtn.disabled = false;
    statusText.style.color = '';
  }
});
