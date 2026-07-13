# LyraAI

> Uma inteligência artificial treinada com que você quiser.

![Status](https://img.shields.io/badge/Status-Em_Desenvolvimento-warning?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-336791?style=flat-square&logo=postgresql)
![LangChain](https://img.shields.io/badge/LangChain-Framework-black?style=flat-square)

## Sobre o Projeto

**LyraAI** é um projeto de Inteligência Artificial criado para expandir as minhas capacidades na criação de agentes de IA. 

O agente possui a capacidade de **aprender e processar novos conhecimentos a partir de arquivos PDF**, utilizando técnicas de OCR (Optical Character Recognition) para extrair o texto de documentos digitais ou escaneados.
O agente retorna respostas tanto em texto como em áudio.

Acompanhe este repositório para seguir a evolução e o desenvolvimento desta ferramenta!

---

## Tecnologias Utilizadas

- **Linguagem:** Python
- **Gerenciador de Pacotes:** [UV](https://github.com/astral-sh/uv)
- **Modelos Locais:** [Ollama](https://ollama.ai/) (Modelos LLM e de Embedding)
- **Banco de Dados:** PostgreSQL com a extensão `pgvector` (via Docker)
- **Framework de IA:** LangChain, LangGraph
- **Interface de Usuário:** Chainlit
- **Processamento de PDFs e OCR:** `pdf2image` & `pytesseract`

---

