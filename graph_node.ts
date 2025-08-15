type User = {
  userId: string;
  name: string;
  messages: Message[];
  preferences: {
    language: "vi" | "en";
    tone: "friendly" | "formal" | "casual";
    addressingStyle: "tôi" | "mình" | "em";
    interests: Topic[];
  };
};

type Message = {
  messageId: string;
  userId: string;
  timestamp: number;
  sender: "human" | "ai";
  content: string;
  embedding: number[];
  topics: Topic[];
};

type Topic = {
  topicId: string;
  name: string;
  embedding: number[];
  messages: Message[];
};
