let ImgUploadType = 'Image'; // Default value

function changeUploadType(type) {
    const imageUploadField = document.getElementById('imageUploadField');
    if (type === 'Image') {
        imageUploadField.innerHTML = '<input type="file" id="image_input" name="image_input" accept="image/*" class="block">';
    } else if (type === 'Link') {
        imageUploadField.innerHTML = '<input type="text" id="imageLink" name="imageLink" placeholder="Enter image link" class="block w-full border border-black rounded-md shadow-sm py-2 px-2">';
    }
    ImgUploadType = type; 
}

document.getElementById('uploadImageButton').addEventListener('click', function() {
    changeUploadType('Image');
});

document.getElementById('uploadLinkButton').addEventListener('click', function() {
    changeUploadType('Link');
});

document.getElementById('queryForm').addEventListener('submit', function(event) {
    event.preventDefault();
    var inputData = document.getElementById('input_data').value;
    var languageSelect = document.getElementById('language');
    var selectedLanguage = languageSelect.options[languageSelect.selectedIndex].value;
    var formData = new FormData();
    formData.append('input_data', inputData);
    formData.append('language', selectedLanguage); 
    if (ImgUploadType === 'Link') {
        var imageLink =  document.getElementById('imageLink').value;
        fetch('/download-image', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ imageUrl: imageLink })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.error('Error downloading image:', data.error);
            } else {
                fetch(data.imagePath)
                    .then(response => response.blob())
                    .then(blob => {
                        const file = new File([blob], 'image.jpg', { type: 'image/jpeg' });
                        formData.append('image_input', file);
                        sendFormData(formData);
                    })
                    .catch(error => console.error('Error loading image:', error));
            }
        })
        .catch(error => {
            console.error('Error downloading image:', error);
            sendFormData(formData);
        });
    } else {
        var imageInput = document.getElementById('image_input');
        if (imageInput) {
            var imageFile = imageInput.files[0];
            formData.append('image_input', imageFile);
        }
        sendFormData(formData);
    }
    
});

function sendFormData(formData) {
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
        var imageInput = document.getElementById('image_input');
        if (imageInput) {
            imageInput.value = '';
        }
        document.getElementById('imageLink').value = '';
    })
    .catch(error => console.error('Error:', error));
}
