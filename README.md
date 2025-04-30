# SAVM
 
![imagem_gerada(3)](https://github.com/user-attachments/assets/88d9241e-4f8c-44b5-8754-7e12d2aa6dcd)

## SAVM: Support "Sanitization Automation" with Vector Machines
> SAVM é um projeto irônico e inovador inspirado nos tradicionais modelos SVM (Support Vector Machine), mas com um toque de “Sanitization”. Imagine um robô que caça ‘ratos’ (ameaças) em um banco de dados: essa é a essência visual e conceitual do SAVM

O SAVM é uma plataforma baseada em machine learning que utiliza SVM para aprender, de forma contínua, como proteger aplicações web contra ataques de injeção SQL. Ele se apoia em duas principais frentes:

- **Análise Estática:**  
  Exame do código-fonte em busca de padrões inseguros e potenciais vulnerabilidades sem executar a aplicação.

- **Análise Dinâmica:**  
  Monitoramento do comportamento em tempo real, capturando tentativas suspeitas e ataques durante a execução.

## Como funciona?

1. **Coleta de Dados**
   - O sistema monitora todas as interações com a aplicação web.
   - Inputs dos usuários e comandos SQL executados são armazenados para análise.

2. **Treinamento do SVM**
   - Utiliza os dados coletados para treinar um modelo de SVM.
   - O modelo aprende a distinguir entre padrões legítimos e possíveis injeções SQL maliciosas.

3. **Sanitização Automática**
   - O algoritmo aprimora suas estratégias de defesa conforme identifica e aprende sobre novas ameaças.
   - A cada novo ataque detectado, a sanificação é ajustada automaticamente.

4. **Relatórios e Logs**
   - Gera relatórios detalhados sobre tentativas de ataque identificadas.
   - Fornece sugestões de ajustes para desenvolvedores e dispara alertas automáticos quando necessário.
