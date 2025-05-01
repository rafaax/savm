const form = document.getElementById('form');
form.addEventListener('submit', handleSubmit);

function handleSubmit(event) {
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
    Swal.fire({
        title: 'Successo!',
        text: 'Formulário enviado com sucesso.',
        icon: 'success',
        confirmButtonText: 'OK'
    });
    console.log(data);
}