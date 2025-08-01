# 🧠 Legal RAG Ollama

A Retrieval-Augmented Generation (RAG) application that allows users to ask legal questions based on Tasmanian legislation using a custom LangChain pipeline and an Ollama-hosted LLM (e.g., LLaMA 3). Includes a simple Flask web interface. (this can be customed to answer any domain you need)

---

## 🚀 Features

- Legal Question Answering with RAG
- Integration with **LangChain**, **Ollama**, and **HuggingFace embeddings**
- Fast semantic + lexical retrieval (BM25 + FAISS)
- Document grading and web fallback using LangGraph logic
- Flask UI for interactive use

---

## 📁 Project Structure

```
legal-rag-ollama/
├── docker-compose.yml
├── app/
│   ├── app.py              # Flask app for UI
│   ├── main.py             # LangGraph orchestrator
│   ├── graph_function2.py  # RAG + retrieval logic
│   ├── environment.yml     # Conda dependencies
│   ├── start.sh            # Start script
│   ├── Dockerfile
│   ├── templates/
│   │   └── index.html
│   └── tasmania_legislation/  # JSON legislation files
```

---

## ⚙️ Requirements

- Docker Engine (with NVIDIA GPU support)
- NVIDIA GPU + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed
- Tavily API https://app.tavily.com/

---

## 🧪 Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/TuanTran1504/RAG-Ollama-app
   cd RAG-Ollama-app
   ```

2. **Pull the model inside Ollama container**
   *(optional before first run — you can also let it auto-pull)*
   ```bash
   docker compose up -d ollama
   docker exec -it ollama ollama pull llama3
   ```

3. Add API key in ".env" file

4. **Start the full app**
   ```bash
   docker compose --compatibility up --build
   ```

5. **Open your browser**
   Visit [http://localhost:5000](http://localhost:5000)  
   Ask a legal question (e.g. about a Tasmanian Act)

---

## 🧠 Example Question

> **"Under the Local Government Act 1993 (Tas), who has the authority to issue a council proclamation?"**

The model will retrieve relevant sections, reason over them, and return a concise legal answer.

---

## 🛠️ Development Notes

- Ollama is run as a separate container and communicates over `http://ollama:11434`
- Uses `LangGraph` to route, grade, and retry RAG responses
- FAISS + BM25 combined for hybrid retrieval
- Model used: `llama3:latest` via Ollama (configurable)
- Fallback method using Web Search(Tavily) when question is out of context or can not retrieve relevant documents


---

## 📦 Badges

![Docker](https://img.shields.io/badge/docker-ready-blue)
![LangChain](https://img.shields.io/badge/langchain-integrated-brightgreen)
![Ollama](https://img.shields.io/badge/ollama-powered-informational)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📚 Credits

- [LangChain](https://www.langchain.com/)
- [Ollama](https://ollama.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- Tasmanian legislation data (public domain)

---

## 📄 License

MIT License. Use at your own discretion. This is a research/developer prototype and not legal advice.
