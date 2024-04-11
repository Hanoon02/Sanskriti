document.getElementById('queryForm').addEventListener('submit', function(event) {
    event.preventDefault();
    var inputData = document.getElementById('input_data').value;
    var imageInput = document.getElementById('image_input').files[0];
    var languageSelect = document.getElementById('language');
    var selectedLanguage = languageSelect.options[languageSelect.selectedIndex].value;
    var formData = new FormData();
    formData.append('input_data', inputData);
    formData.append('image_input', imageInput);
    formData.append('language', selectedLanguage); 
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.text())
    .then(result => {
        var resultDiv = document.createElement('div');
        resultDiv.classList.add('mt-4');
        resultDiv.innerHTML = result;
        var resultsContainer = document.getElementById('a');
        resultsContainer.appendChild(resultDiv);
        document.getElementById('input_data').value = '';
        document.getElementById('image_input').value = '';
    })
    .catch(error => console.error('Error:', error));
});
