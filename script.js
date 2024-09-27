document.getElementById('symptom-form').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent the form from refreshing the page

    // Get selected symptoms
    const symptoms = [];
    document.querySelectorAll('input[name="symptom"]:checked').forEach(checkbox => {
        symptoms.push(checkbox.value);
    });

    // Check if no symptoms are selected
    if (symptoms.length === 0) {
        alert("Please select at least one symptom.");
        return;
    }

    // Send symptoms to Flask backend for prediction
    fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ symptoms: symptoms }),
    })
    .then(response => response.json())
    .then(data => {
        // Display the prediction result
        const resultSection = document.getElementById('result');
        const predictionText = document.getElementById('prediction');
        predictionText.textContent = "Predicted Diagnosis: " + data.prediction;
        resultSection.classList.remove('hidden');
    })
    .catch((error) => {
        console.error('Error:', error);
    });
});
