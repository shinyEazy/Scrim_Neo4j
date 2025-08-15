package main

import (
	"bufio"
	"context"
	"crypto/rand"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"time"

	"github.com/joho/godotenv"
	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
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
	
	// Save to JSONL
	saveAsJSONL(message)
	
	// Add to Neo4j and create similarity edges in one transaction
	if err := addMessageAndCreateEdges(message); err != nil {
		log.Printf("Error adding message to Neo4j: %v", err)
	}
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

// Neo4j database connection
var neo4jDriver neo4j.Driver

// Initialize Neo4j connection
func initNeo4j() error {
	uri := "neo4j://localhost:7687"
	username := "neo4j"
	password := "123123123"
	
	var err error
	neo4jDriver, err = neo4j.NewDriver(uri, neo4j.BasicAuth(username, password, ""))
	if err != nil {
		return fmt.Errorf("failed to create Neo4j driver: %v", err)
	}
	
	// Test connection
	err = neo4jDriver.VerifyConnectivity()
	if err != nil {
		return fmt.Errorf("failed to connect to Neo4j: %v", err)
	}
	
	fmt.Println("✅ Connected to Neo4j database")
	return nil
}

// Add message and create similarity edges in a single transaction
func addMessageAndCreateEdges(message Message) error {
	session := neo4jDriver.NewSession(neo4j.SessionConfig{})
	defer session.Close()
	
	_, err := session.WriteTransaction(func(tx neo4j.Transaction) (any, error) {
		// First, create the message node
		createQuery := `
			CREATE (m:Message {
				messageId: $messageId,
				timestamp: $timestamp,
				sender: $sender,
				content: $content,
				embedding: $embedding
			})
			RETURN m
		`
		createParams := map[string]any{
			"messageId": message.MessageID,
			"timestamp": message.Timestamp,
			"sender":    message.Sender,
			"content":   message.Content,
			"embedding": message.Embedding,
		}
		
		_, err := tx.Run(createQuery, createParams)
		if err != nil {
			return nil, fmt.Errorf("failed to create message node: %v", err)
		}
		
		fmt.Printf("📊 Added message node to Neo4j: %s\n", message.MessageID)
		
		// Then, find similar messages and create edges
		similarityQuery := `
			MATCH (m1:Message {messageId: $messageId})
			MATCH (m2:Message)
			WHERE m2.messageId <> $messageId
			RETURN m2.messageId as messageId, m2.embedding as embedding, m2.content as content
		`
		similarityParams := map[string]any{
			"messageId": message.MessageID,
		}
		
		result, err := tx.Run(similarityQuery, similarityParams)
		if err != nil {
			return nil, fmt.Errorf("failed to query existing messages: %v", err)
		}
		
		edgesCreated := 0
		totalMessages := 0
		
		for result.Next() {
			totalMessages++
			record := result.Record()
			existingMessageId := record.Values[0].(string)
			existingEmbedding := record.Values[1].([]interface{})
			
			// Convert interface{} to []float64
			embedding := make([]float64, len(existingEmbedding))
			for i, v := range existingEmbedding {
				embedding[i] = v.(float64)
			}
			
			// Calculate similarity
			similarity := cosineSimilarity(message.Embedding, embedding)
			
			// Create edge if similarity > 0.1
			if similarity > 0.7 {
				// Create the edge in the same transaction
				edgeQuery := `
					MATCH (m1:Message {messageId: $messageId1})
					MATCH (m2:Message {messageId: $messageId2})
					MERGE (m1)-[r:CONTEXTUAL_LINK {similarity: $similarity, timestamp: $timestamp}]-(m2)
					RETURN r
				`
				edgeParams := map[string]any{
					"messageId1": message.MessageID,
					"messageId2": existingMessageId,
					"similarity": similarity,
					"timestamp":  time.Now().Unix(),
				}
				
				_, err := tx.Run(edgeQuery, edgeParams)
				if err != nil {
					log.Printf("Failed to create edge: %v", err)
				} else {
					edgesCreated++
				}
			}
		}
		
		if edgesCreated > 0 {
			fmt.Printf("🔗 Created %d similarity edges for message: %s\n", edgesCreated, message.MessageID)
		}
		
		return result.Consume()
	})
	
	if err != nil {
		return fmt.Errorf("failed to add message and create edges: %v", err)
	}
	
	return nil
}

// Calculate cosine similarity between two embeddings
func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0.0
	}
	
	var dotProduct, normA, normB float64
	for i := 0; i < len(a); i++ {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	
	if normA == 0 || normB == 0 {
		return 0.0
	}
	
	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}

func main() {
	_ = godotenv.Load()

	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("Error: OPENAI_API_KEY environment variable not set.")
	}

	// Initialize Neo4j
	if err := initNeo4j(); err != nil {
		log.Fatalf("Failed to initialize Neo4j: %v", err)
	}
	defer neo4jDriver.Close()

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
