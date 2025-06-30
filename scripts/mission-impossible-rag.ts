import fs from 'fs/promises';
import path from 'path';
import { config } from 'dotenv';
import { OpenAIEmbeddings } from '@langchain/openai';
import { Pinecone } from '@pinecone-database/pinecone';
import { PineconeStore } from '@langchain/pinecone';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import crypto from 'crypto';

config();

const PINECONE_API_KEY = process.env.PINECONE_API_KEY!;
const PINECONE_INDEX = process.env.PINECONE_INDEX!;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY!;

async function main() {
  try {
    const scriptsDir = __dirname;
    const files = await fs.readdir(scriptsDir);
    const txtFiles = files.filter((file) => file.endsWith('.txt'));
    if (txtFiles.length === 0) {
      console.log('No .txt files found in scripts directory.');
      return;
    }

    // Initialize Pinecone client and index
    const pinecone = new Pinecone({
      apiKey: PINECONE_API_KEY,
    });
    const index = pinecone.Index(PINECONE_INDEX);

    // Vectorizer
    const embeddings = new OpenAIEmbeddings({
      openAIApiKey: OPENAI_API_KEY,
      modelName: 'text-embedding-3-small',
    });

    for (const file of txtFiles) {
      const filePath = path.join(scriptsDir, file);
      const fileText = await fs.readFile(filePath, 'utf-8');
      // Split the text into chunks. From the docs:
      // Text is naturally organized into hierarchical units such as paragraphs, sentences, and words. We can leverage this inherent structure to inform our
      // splitting strategy, creating split that maintain natural language flow, maintain semantic coherence within split, and adapts to varying levels of
      // text granularity. LangChain's RecursiveCharacterTextSplitter implements this concept:
      // - The RecursiveCharacterTextSplitter attempts to keep larger units (e.g., paragraphs) intact.
      // - If a unit exceeds the chunk size, it moves to the next level (e.g., sentences).
      // - This process continues down to the word level if necessary.
      const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 500,
        chunkOverlap: 50,
      });
      const chunks = await splitter.splitText(fileText);
      const docs = chunks.map((chunk, i) => {
        const contentHash = crypto.createHash('md5').update(chunk).digest('hex').slice(0, 8);
        return {
          pageContent: chunk,
          metadata: {
            id: `${file}-chunk-${i}-${contentHash}`,
            source: file,
            chunkIndex: i,
          },
        };
      });
      await PineconeStore.fromDocuments(docs, embeddings, {
        pineconeIndex: index,
        maxConcurrency: 5,
      });
      console.log(`Successfully upserted vectors for ${file} to Pinecone using PineconeStore!`);
    }
  } catch (error) {
    console.error('Error processing and uploading vectors:', error);
    process.exit(1);
  }
}

main();
