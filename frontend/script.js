document.getElementById("predict-form").addEventListener("submit", async function (e) {
  e.preventDefault();
  const input = document.getElementById("features").value;

  try {
    const response = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ features: input }),
    });

    const data = await response.json();
    document.getElementById("result").innerText = `🎯 Predicted Popularity: ${data.popularity}`;
  } catch (error) {
    document.getElementById("result").innerText = "❌ Error predicting popularity.";
    console.error(error);
  }
});