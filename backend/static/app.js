async function getSong() {
    console.log("🎬 getSong triggered");

    const text = document.getElementById("moodText").value.trim();
    const language = document.getElementById("language").value;
    const resultDiv = document.getElementById("result");
    const player = document.getElementById("player");

   if (!text) { alert("Please enter your mood!"); return; }

   resultDiv.innerText = "🎧 Finding the perfect song for you...";
   player.src = ""; // Clear old video

    try {
    const response = await fetch("/recommend", {

        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ text: text, language: language }),
    });

    const data = await response.json();

     if (response.ok) {
      resultDiv.innerHTML = `🎵 Song: ${data.song_title}<br>😊 Mood: ${data.mood}<br>🌐 Language: ${data.language}`;
      const query = encodeURIComponent(data.song_title + " song");
      player.src = `https://www.youtube.com/embed?listType=search&list=${query}&autoplay=1&mute=1`;
    } else {
      resultDiv.innerHTML = `❌ ${data.error || "No song found."}`;
    }
  } catch (err) {
    console.error(err);
    resultDiv.innerText = "⚠️ Unable to connect to the backend server.";
  }
}
