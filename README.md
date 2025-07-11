# Mission Impossible RAG

Chat interface that answers questions using both context retrieved from vectorized movie plot data and LLMs.

**Note:** GPT-3.5 does not have knowledge about the latest Mission Impossible movies. To overcome this limitation, this app uses `.txt` documents placed in the `/scripts` directory (such as `dead-reckoning-plot.txt` and `final-reckoning-plot.txt`) to vectorize and store up-to-date information about these movies in Pinecone. When a user asks a question, the system retrieves relevant context from these documents and augments the user prompt, enabling the model to provide accurate and current answers about the latest Mission Impossible films.

## Features
- **Chat UI**: Simple chat interface at `/chat` for interacting with the RAG system.
- **RAG Backend**: Uses Pinecone for vector search and OpenAI for embeddings and chat completions.
- **Contextualized QA**: Reformulates user questions based on chat history for better retrieval.
- **Streaming Responses**: Answers are streamed to the frontend for a responsive experience.
- **Vectorization Script**: Easily vectorize and upload plot data from `.txt` files in the `scripts/` directory.

### Environment Variables
Create a `.env` file in the root with the following variables:
```
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX=your-pinecone-index-name
OPENAI_API_KEY=your-openai-api-key
```

## Setup
1. **Install dependencies**
   ```sh
   pnpm install
   ```

2. **Prepare vector data**
   - Place `.txt` files (e.g., `dead-reckoning-plot.txt`, `final-reckoning-plot.txt`) in the `scripts/` directory.
   - Run the vectorization script to upload vectors to Pinecone:
     ```sh
     pnpm run run:vectorize-mission-impossible-plot
     ```

3. **Start the development server**
   ```sh
   pnpm dev
   ```

4. **Chat**
   - Visit `http://localhost:3000/chat` to use the chat interface.
   - Ask questions about the uploaded movie plots or general knowledge.

## Project Structure
- `src/app/chat/page.tsx` – Chat UI
- `src/app/api/prompt/route.ts` – RAG API endpoint (retrieves, contextualizes, and answers)
- `scripts/mission-impossible-rag.ts` – Vectorization script for `.txt` files
- `public/` – Static assets
