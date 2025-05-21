    function showSection(id) {
      document.querySelectorAll('.section').forEach(sec => sec.classList.add('hidden'));
      document.getElementById(id).classList.remove('hidden');
    }

    $(document).ready(function () {
      let requisicoesTable = $('#requisicoesTable').DataTable({
      ajax: {
        url: 'http://127.0.0.1:8000/all-queries-detected',
        dataSrc: '' 
      },
      columns: [
        { data: 'query_text' },
        { data: 'is_malicious_prediction',
          render: function (data) {
            return data ? 'Sim' : 'Não';
          }
        },
        {
          data: 'timestamp',
          render: function (data) {
            let date = new Date(data);
            return date.toLocaleString('pt-BR', { year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit' });
          }
        }
      ],
      order: [[2, 'desc']],
      language: {
        search: "Pesquisar:",
        lengthMenu: "Mostrar _MENU_ registros",
        info: "Mostrando _START_ a _END_ de _TOTAL_ registros",
        paginate: {
          first: "Primeiro",
          last: "Último",
          next: "Próximo",
          previous: "Anterior"
        },
        zeroRecords: "Nenhum registro encontrado"
      }
        });

         let usuariosTable = $('#usuariosTable').DataTable({
      ajax: {
        url: 'http://127.0.0.1:8000/all-users-registred',
        dataSrc: '' 
      },
      columns: [
        { data: 'nome' },
        { data: 'email'},
        { data: 'cpf'},
        { data: 'endereco'},
        {
          data: 'date',
          render: function (data) {
            let date = new Date(data);
            return date.toLocaleString('pt-BR', { year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit' });
          }
        }
      ],
      order: [[4, 'desc']],
      language: {
        search: "Pesquisar:",
        lengthMenu: "Mostrar _MENU_ registros",
        info: "Mostrando _START_ a _END_ de _TOTAL_ registros",
        paginate: {
          first: "Primeiro",
          last: "Último",
          next: "Próximo",
          previous: "Anterior"
        },
        zeroRecords: "Nenhum registro encontrado"
      }
        });
    });