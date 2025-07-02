import { NextRequest, NextResponse } from 'next/server';
import { Pinecone } from '@pinecone-database/pinecone';
import { PineconeStore } from '@langchain/pinecone';
import { OpenAIEmbeddings } from '@langchain/openai';
import { ChatOpenAI } from '@langchain/openai';
import { formatDocumentsAsString } from 'langchain/util/document';
import { RunnableSequence } from "@langchain/core/runnables";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from '@langchain/core/output_parsers';
import { LangChainAdapter } from 'ai';

const PINECONE_API_KEY = process.env.PINECONE_API_KEY!;
const PINECONE_INDEX = process.env.PINECONE_INDEX!;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY!;

const createRetriever = async () => {
  const pinecone = new Pinecone({ apiKey: PINECONE_API_KEY });
  const index = pinecone.Index(PINECONE_INDEX);
  const embeddings = new OpenAIEmbeddings({
    openAIApiKey: OPENAI_API_KEY,
    modelName: 'text-embedding-3-small',
  });
  const vectorStore = await PineconeStore.fromExistingIndex(embeddings, { pineconeIndex: index });
  return vectorStore.asRetriever();
};

const promptTemplate = ChatPromptTemplate.fromMessages([
  [
    "human",
    `Use the following context to help answer the question. If the answer is not found in the context, feel free to answer based on your own knowledge. Use three sentences maximum and keep the answer concise.
     Question: {question} 
     Context: {context}
     Answer:`,
  ],
]);

export async function POST(req: NextRequest) {
  try {
    const { messages } = await req.json();

    const lastMessage = messages[messages.length - 1];
    if (!lastMessage || lastMessage.role !== 'user') {
      return NextResponse.json({ error: 'Last message must be from user' }, { status: 400 });
    }
    const prompt = lastMessage.content;

    const retriever = await createRetriever();

    const retrievalChain = RunnableSequence.from([
      (input) => input.question,
      retriever,
      formatDocumentsAsString,
    ]);

    const chat = new ChatOpenAI({
      openAIApiKey: OPENAI_API_KEY,
      modelName: 'gpt-3.5-turbo',
      temperature: 0.2,
    });

    const outputParser = new StringOutputParser();

    const generationChain = RunnableSequence.from([
      {
        question: (input) => input.question,
        context: retrievalChain,
      },
      promptTemplate,
      chat,
      outputParser,
    ]);

    const stream = await generationChain.stream({ question: prompt });

    return LangChainAdapter.toDataStreamResponse(stream);
  } catch (error) {
    return NextResponse.json({ error }, { status: 500 });
  }
} 
