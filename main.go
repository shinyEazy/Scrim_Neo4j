package main

import (
	"bufio"
	"context"
	"crypto/rand"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/joho/godotenv"
	"github.com/sashabaranov/go-openai"
)

// Graph node structures matching TypeScript types
type Message struct {
	MessageID string    `json:"messageId"`
	Timestamp int64     `json:"timestamp"`
	Sender    string    `json:"sender"`
	Content   string    `json:"content"`
	Embedding []float64 `json:"embedding"`
}

// Generate a random ID for nodes
func generateID() string {
	b := make([]byte, 16)
	rand.Read(b)
	return fmt.Sprintf("%x", b)
}

// Print a message node that would be added to the graph
func printMessageNode(sender string, content string, client *openai.Client) {
	// Get embedding from OpenAI
	embedding, err := getEmbedding(client, content)
	if err != nil {
		log.Printf("Error getting embedding: %v", err)
		embedding = []float64{} // Fallback to empty embedding
	}
	
	message := Message{
		MessageID: generateID(),
		Timestamp: time.Now().Unix(),
		Sender:    sender,
		Content:   content,
		Embedding: embedding,
	}
	
	saveAsJSONL(message)
}

// Get embedding from OpenAI text-embedding-3-small
func getEmbedding(client *openai.Client, text string) ([]float64, error) {
	resp, err := client.CreateEmbeddings(
		context.Background(),
		openai.EmbeddingRequest{
			Input: []string{text},
			Model: "text-embedding-3-small",
		},
	)
	if err != nil {
		return nil, err
	}
	
	if len(resp.Data) == 0 {
		return nil, fmt.Errorf("no embedding data received")
	}
	
	// Convert []float32 to []float64
	embedding := make([]float64, len(resp.Data[0].Embedding))
	for i, v := range resp.Data[0].Embedding {
		embedding[i] = float64(v)
	}
	return embedding, nil
}

// Save message as JSONL format
func saveAsJSONL(message Message) {
	// Convert message to JSON
	jsonData, err := json.Marshal(message)
	if err != nil {
		log.Printf("Error marshaling message to JSON: %v", err)
		return
	}
	
	// Append to JSONL file
	file, err := os.OpenFile("graph_nodes.jsonl", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		log.Printf("Error opening JSONL file: %v", err)
		return
	}
	defer file.Close()
	
	// Write JSON line
	if _, err := file.WriteString(string(jsonData) + "\n"); err != nil {
		log.Printf("Error writing to JSONL file: %v", err)
	}
}

func main() {
	_ = godotenv.Load()

	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("Error: OPENAI_API_KEY environment variable not set.")
	}

	client := openai.NewClient(apiKey)

	messages := []openai.ChatCompletionMessage{
		{
			Role:    openai.ChatMessageRoleSystem,
			Content: "You are a helpful and friendly chatbot.",
		},
	}

	fmt.Println("🤖 Chatbot is ready! Type 'exit' to end the conversation.")
	fmt.Println("---------------------------------------------------------")

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("You: ")
		if !scanner.Scan() {
			break
		}
		userInput := scanner.Text()

		if userInput == "exit" {
			fmt.Println("Goodbye! 👋")
			break
		}

		// Print user message node
		printMessageNode("human", userInput, client)
		
		messages = append(messages, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: userInput,
		})

		resp, err := client.CreateChatCompletion(
			context.Background(),
			openai.ChatCompletionRequest{
				Model:    "gpt-4o-mini",
				Messages: messages,
			},
		)

		if err != nil {
			fmt.Printf("ChatCompletion error: %v\n", err)
			continue
		}

		chatbotResponse := resp.Choices[0].Message.Content
		fmt.Printf("Bot: %s\n", chatbotResponse)

		// Print bot response node
		printMessageNode("ai", chatbotResponse, client)

		messages = append(messages, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleAssistant,
			Content: chatbotResponse,
		})
	}

	if err := scanner.Err(); err != nil {
		log.Printf("Error reading standard input: %v", err)
	}
}
