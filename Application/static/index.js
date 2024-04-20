let UploadType = 'Text'; 
let OutputType = 'Text';

function changeUploadType(type) {
    const uploadField = document.getElementById('uploadField');
    if (type === 'Image') {
        uploadField.innerHTML = '<input style="outline: none;" type="file" id="image_input" name="image_input" accept="image/*" class="block text-[20px]"></input>';
    } else if (type === 'Link') {
        uploadField.innerHTML = '<input style="outline: none;" type="text" id="imageLink" name="imageLink" placeholder="Provide image link" class="w-full rounded-md shadow-sm px-2 bg-black"></input>';
    }
    else if (type === 'Text') {
        uploadField.innerHTML = '<input style="outline: none;" placeholder="Write your query..." id="input_data" name="input_data" class="w-full text-[25px] bg-[#16140C] text-[#FCE1B9]"></input>';
    }
    UploadType = type; 
}

function changeOutputType(type) {
    const image = document.getElementById('imageOutput');
    const text = document.getElementById('textOutput');
    const hybrid = document.getElementById('hybrid');
    if (type === 'Text') {
        text.innerHTML = `<div class='text-[25px] bg-[#000000] text-[#FCE1B9] border-4 border-[#48553D] rounded-full px-6 playball-regular-2'>
                            Text Output
                            </div>`
        image.innerHTML = `<div class='text-[25px] bg-[#FCE1B9] text-[#000000] border-4 border-[#48553D] rounded-full px-6 playball-regular-2'>
                            Image Output
                            </div>`
        hybrid.innerHTML = `<div class='text-[25px] bg-[#FCE1B9] text-[#000000] border-4 border-[#48553D] rounded-full px-6 playball-regular-2'>
                            Hybrid
                            </div>`

    } else if (type === 'Image') {
        text.innerHTML = `<div class='text-[25px] bg-[#FCE1B9] text-[#000000] border-4 border-[#48553D] rounded-full px-6 playball-regular-2'>
                            Text Output
                            </div>`
        image.innerHTML = `<div class='text-[25px] bg-[#000000] text-[#FCE1B9] border-4 border-[#48553D] rounded-full px-6 playball-regular-2'>
                            Image Output
                            </div>`
        hybrid.innerHTML = `<div class='text-[25px] bg-[#FCE1B9] text-[#000000] border-4 border-[#48553D] rounded-full px-6 playball-regular-2'>
                            Hybrid
                            </div>`
    }
    else if (type === 'Hybrid') {
        text.innerHTML = `<div class='text-[25px] bg-[#FCE1B9] text-[#000000] border-4 border-[#48553D] rounded-full px-6 playball-regular-2'>
                            Text Output
                            </div>`
        image.innerHTML = `<div class='text-[25px] bg-[#FCE1B9] text-[#000000] border-4 border-[#48553D] rounded-full px-6 playball-regular-2'>
                            Image Output
                            </div>`
        hybrid.innerHTML = `<div class='text-[25px] bg-[#000000] text-[#FCE1B9] border-4 border-[#48553D] rounded-full px-6 playball-regular-2'>
                            Hybrid
                            </div>`
    }
    OutputType = type; 
}

document.getElementById('uploadImageButton').addEventListener('click', function() {
    changeUploadType('Image');
});

document.getElementById('uploadLinkButton').addEventListener('click', function() {
    changeUploadType('Link');
});

document.getElementById('uploadTextButton').addEventListener('click', function() {
    changeUploadType('Text');
});

document.getElementById('textOutput').addEventListener('click', function() {
    changeOutputType('Text');
});

document.getElementById('imageOutput').addEventListener('click', function() {
    changeOutputType('Image');
});

document.getElementById('hybrid').addEventListener('click', function() {
    changeOutputType('Hybrid');
});

document.getElementById('submit_button').addEventListener('click', function(event) {
    event.preventDefault();
    var inputData = ''
    if(UploadType === 'Text'){
        inputData = document.getElementById('input_data').value;
    }
    var languageSelect = document.getElementById('language');
    var selectedLanguage = languageSelect.options[languageSelect.selectedIndex].value;
    var formData = new FormData();
    formData.append('input_data', inputData);
    formData.append('language', selectedLanguage); 
    formData.append('output_type', OutputType)
    formData.append('input_type', UploadType)
    if (UploadType === 'Link') {
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
    var resultsContainer = document.getElementById('a');
    resultsContainer.innerHTML = `
        <div >
            <img class="h-[100px] w-[100px] mt-5" src="../static/loading.svg" alt="Load"></img>
        </div>
    `;
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
        resultsContainer.innerHTML = resultDiv.innerHTML;
        text_input = document.getElementById('input_data')
        if(text_input) text_input.value = '';
        var imageInput = document.getElementById('image_input');
        if (imageInput) {
            imageInput.value = '';
        }
        img_link = document.getElementById('imageLink')
        if(img_link) img_link.value = ''
    })
    .catch(error => console.error('Error:', error));
}

document.getElementById('submit_feedback').addEventListener('click', function(event) {
    event.preventDefault();
    var feedback = document.getElementById('feedback').value;
    fetch('/feedback', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ feedback: feedback })
    })
    document.getElementById('feedback').value = ''
});