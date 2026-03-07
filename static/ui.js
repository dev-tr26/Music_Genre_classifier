const audioInput = document.getElementById("audioInput");
const analyzeBtn = document.getElementById("analyzeBtn");
const fileName = document.getElementById("fileName");
const statusText = document.getElementById("status");

const resultBlock = document.getElementById("resultBlock");
const genreText = document.getElementById("genreText");
const confidenceText = document.getElementById("confidenceText");
const bars = document.getElementById("bars");

let selectedFile = null;

audioInput.addEventListener("change", () => {
  selectedFile = audioInput.files[0];
  if (selectedFile) {
    fileName.textContent = selectedFile.name.toUpperCase();
    analyzeBtn.disabled = false;
    statusText.textContent = "READY";
  }
});

analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  statusText.textContent = "PROCESSING...";
  resultBlock.classList.add("hidden");
  bars.innerHTML = "";

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData
    });

    const data = await response.json();

    genreText.textContent = data.genre.toUpperCase();
    confidenceText.textContent =
      Math.round(data.confidence * 100) + "%";

    Object.entries(data.scores).forEach(([_, value]) => {
      const bar = document.createElement("div");
      bar.className = "bar";
      bar.style.width = `${value * 100}%`;
      bars.appendChild(bar);
    });

    resultBlock.classList.remove("hidden");
    statusText.textContent = "DONE";
  } catch {
    statusText.textContent = "ERROR";
  }
});
