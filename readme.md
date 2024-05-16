# Bosh Chat-Bot

## Project Structure
This project is for Chat-Bot for Bosh Hackathon. This chat-bot is based on RAG and uses user content to generate responses.

---

## Tech stack used:

The tech stack used here is,

- langchain
- qdrant
- LLM ollama llama3



---


## Project setup instructions:
	
To install the dependencies use:

```
poetry install

```

<<<<<<< HEAD
For unstructered data:
```
sudo apt-get install poppler-utils
sudo apt install tesseract-ocr
sudo apt install libtesseract-dev
poetry add timm

```

make sure you restart the terminal after installing these dependencies.


To run qdrant make sure you have docker installed and run the following commands:
=======
To run qdrant make sure you have docker installed and run the following commands in `linux`:
>>>>>>> 691c55bb074bc4f30721135f0ef37675577c457a
```
docker pull qdrant/qdrant
docker run -p 6333:6333 -p 6334:6334 \-v $(pwd)/qdrant_storage:/qdrant/storage:z \qdrant/qdrant

```
To run qdrant make sure you have docker installed and run the following commands in `windows`:
```
docker pull qdrant/qdrant
docker run -p 6333:6333 -p 6334:6334 \-v $(cd)/qdrant_storage:/qdrant/storage:z \qdrant/qdrant

```
Then you can check the status of the server by visiting `http://localhost:6333/` in your browser.
and see the dashboard at `http://localhost:6334/dashboard`


To run LLM ollama llama3 use:
```
docker pull ollama/ollama
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
docker exec -it ollama ollama run llama3

```

To run LLM with gpu use:

Ensure you have nvidia container toolkit installed (check ollama image docs) and run the following commands:
```
docker pull ollama/ollama
docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
docker exec -it ollama ollama run llama3

```

Reminder above commands are first time setup, for subsequent runs you can use the following command:
```

docker start ollama
docker exec -it ollama ollama run llama3

```

To run the project use:

```

poetry shell
streamlit run app.py

```

make sure you are in the project directory before running the above command.
then visit `http://localhost:8501/` in your browser to view the app.






