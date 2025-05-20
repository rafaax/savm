const form = document.getElementById('form');
form.addEventListener('submit', handleSubmit);

async function handleSubmit(event) {
    event.preventDefault();
    const formData = new FormData(form);
    const data = Object.fromEntries(formData.entries());
    const entries = Object.entries(data);
    const isEmpty = entries.some(([key, value]) => value.trim() === '');
    if (isEmpty) {
        Swal.fire({
            title: 'Erro!',
            text: 'Nenhum campo pode estar vazio.',
            icon: 'warning',
            confirmButtonText: 'OK'
        });
        return;
    }

    for (const entry of entries) {
        const response = await CheckSQLI(entry[1]);
        if (response.is_malicious) {
            Swal.fire({
                title: 'Successo!',
                text: 'Formulário enviado com sucesso.',
                icon: 'success',
                confirmButtonText: 'OK'
            });
            setTimeout(function(){
                window.location.reload();
            }, 5000);
            return;
        }
    }


    const response = await postData('/submit-form', data);

    if (response.status === 'error') {
        Swal.fire({
            title: 'Erro!',
            text: response.message,
            icon: 'warning',
            confirmButtonText: 'OK'
        });
        return;
    }
    
    Swal.fire({
        title: 'Successo!',
        text: 'Formulário enviado com sucesso.',
        icon: 'success',
        confirmButtonText: 'OK'
    });

    setTimeout(function(){
        form.reset();
        window.location.reload();
    }, 5000);
    
}


async function CheckSQLI(field) {
    const response = await fetch("http://localhost:8000/detect-sqli", {
        method: 'POST',
        body: JSON.stringify({
            query: field
        }),
        headers: {
            'Content-Type': 'application/json'
        }
    });
    const responseData = await response.json();
    return responseData;
}

async function postData(url, data) {
    const response = await fetch("http://localhost:8000" + url, {
        method: 'POST',
        body: JSON.stringify(data),
        headers: {
            'Content-Type': 'application/json'
        }
    });
    const responseData = await response.json();
    return responseData;
}