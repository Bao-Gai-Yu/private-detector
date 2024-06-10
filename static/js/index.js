document.getElementById('upload-form').onsubmit = async function (event) {
    event.preventDefault();
    const formData = new FormData();
    const fileField = document.querySelector('input[type="file"]');
    for (const file of fileField.files) {
        formData.append('files', file);
    }

    const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        body: formData
    });

    const results = await response.json();
    const resultDiv = document.getElementById('result');
    resultDiv.innerHTML = '';

    results.forEach(result => {
        if (result.error) {
            resultDiv.innerHTML += `<p>Error: ${result.error}</p>`;
        } else {
            resultDiv.innerHTML += `<p>Probability: ${result.probability.toFixed(2)}% - ${result.filename}</p>`;
        }
    });
}