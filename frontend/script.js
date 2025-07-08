document.getElementById("predict-form").addEventListener("submit", async function (e) {
  e.preventDefault();

  const songName = document.getElementById("features").value.trim();
  const resultDiv = document.getElementById("result");
  resultDiv.innerText = "🔄 Predicting...";

  try {
    const response = await fetch("https://gingeraipredictor.onrender.com/predict_by_name", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ song_name: songName })
    });

    const data = await response.json();

    if (!response.ok) {
      resultDiv.innerText = `❌ Error: ${data.error || "Something went wrong"}`;
      return;
    }

    resultDiv.innerHTML = `
      <h3>🎧 Prediction Result</h3>
      <p><strong>Track:</strong> ${data.track}</p>
      <p><strong>Artist:</strong> ${data.artist}</p>
      <p><strong>Predicted Popularity:</strong> ${data.predicted_popularity}</p>
      <p><strong>Hit Status:</strong> ${data.hit_prediction}</p>
    `;
  } catch (err) {
    resultDiv.innerText = `❌ Network Error: ${err.message}`;
  }
});