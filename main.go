package main

import (
	"bufio"
	"context"
	"crypto/rand"
	"fmt"
	"log"
	"math"
	"os"
	"strings"
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
	Topics    []string  `json:"topics"`
}

type Topic struct {
	TopicID   string    `json:"topicId"`
	Name      string    `json:"name"`
	Embedding []float64 `json:"embedding"`
	Messages  []Message `json:"messages"`
}

type User struct {
	UserID      string    `json:"userId"`
	Name        string    `json:"name"`
	CreatedAt   int64     `json:"createdAt"`
	LastActive  int64     `json:"lastActive"`
	Preferences UserPreferences `json:"preferences"`
}

type UserPreferences struct {
	Language        string   `json:"language"`
	Tone            string   `json:"tone"`
	AddressingStyle string   `json:"addressingStyle"`
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

// Generate a random ID for nodes
func generateID() string {
	b := make([]byte, 16)
	rand.Read(b)
	return fmt.Sprintf("%x", b)
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

// Extract ecommerce topics from content using LLM
func extractTopics(client *openai.Client, content string) ([]string, error) {
	resp, err := client.CreateChatCompletion(
		context.Background(),
		openai.ChatCompletionRequest{
			Model: "gpt-4o-mini",
			Messages: []openai.ChatCompletionMessage{
				{
					Role: openai.ChatMessageRoleSystem,
					Content: `Phân tích nội dung và gán tag thương mại điện tử phù hợp từ danh sách sau:

Danh sách tag có sẵn:
["Áo", "Quần", "Giày", "Túi", "Mũ", "Khuyến mãi", "Giảm giá", "Freeship", "Combo"]

Quy tắc gán tag:
1. Chỉ sử dụng các tag trong danh sách trên
2. Gán tag dựa trên nội dung thực tế của tin nhắn
3. Một tin nhắn có thể có nhiều tag
4. Nếu không có tag phù hợp thì trả về "không có tag"

Trả về danh sách tag phân cách bằng dấu phẩy, không có dấu ngoặc kép.`,
				},
				{
					Role:    openai.ChatMessageRoleUser,
					Content: content,
				},
			},
			MaxTokens: 50,
			Temperature: 0.1,
		},
	)
	
	if err != nil {
		return nil, fmt.Errorf("failed to extract topics: %v", err)
	}
	
	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("no response from topic extraction")
	}
	
	topicsText := resp.Choices[0].Message.Content
	// Clean up and split topics
	topicsText = strings.TrimSpace(topicsText)
	topicsText = strings.Trim(topicsText, `"'`)
	
	// Check if no topics found
	if strings.Contains(strings.ToLower(topicsText), "không có tag") || 
	   strings.Contains(strings.ToLower(topicsText), "no tag") ||
	   strings.TrimSpace(topicsText) == "" {
		return []string{}, nil
	}
	
	// Split by comma and clean each topic
	topics := strings.Split(topicsText, ",")
	var cleanedTopics []string
	
	// Predefined valid tags
	validTags := []string{"Áo", "Quần", "Giày", "Túi", "Mũ", "Khuyến mãi", "Giảm giá", "Freeship", "Combo"}
	
	for _, topic := range topics {
		topic = strings.TrimSpace(topic)
		if topic != "" && topic != "không có tag" {
			// Only include if it's a valid tag
			for _, validTag := range validTags {
				if strings.EqualFold(topic, validTag) {
					cleanedTopics = append(cleanedTopics, validTag)
					break
				}
			}
		}
	}
	
	return cleanedTopics, nil
}

// Print a message node that would be added to the graph
func printMessageNode(sender string, content string, client *openai.Client, userID string) {
	// Get embedding from OpenAI
	embedding, err := getEmbedding(client, content)
	if err != nil {
		log.Printf("Error getting embedding: %v", err)
		embedding = []float64{} // Fallback to empty embedding
	}
	
	// Extract topics from content
	topics, err := extractTopics(client, content)
	if err != nil {
		log.Printf("Error extracting topics: %v", err)
		topics = []string{} // No fallback topic for errors
	}
	
	message := Message{
		MessageID: generateID(),
		Timestamp: time.Now().Unix(),
		Sender:    sender,
		Content:   content,
		Embedding: embedding,
		Topics:    topics,
	}
	
	// Add to Neo4j and create similarity edges in one transaction
	if err := addMessageAndCreateEdges(message, userID); err != nil {
		log.Printf("Error adding message to Neo4j: %v", err)
	}
}

// Add message and create similarity edges in a single transaction
func addMessageAndCreateEdges(message Message, userID string) error {
	session := neo4jDriver.NewSession(neo4j.SessionConfig{})
	defer session.Close()
	
	_, err := session.WriteTransaction(func(tx neo4j.Transaction) (any, error) {
		// First, create the message node
		createQuery := `
			CREATE (m:Message {
				messageId: $messageId,
				userId: $userId,
				timestamp: $timestamp,
				sender: $sender,
				content: $content,
				embedding: $embedding,
				topics: $topics
			})
			RETURN m
		`
		createParams := map[string]any{
			"messageId": message.MessageID,
			"userId":    userID,
			"timestamp": message.Timestamp,
			"sender":    message.Sender,
			"content":   message.Content,
			"embedding": message.Embedding,
			"topics":    message.Topics,
		}
		
		_, err := tx.Run(createQuery, createParams)
		if err != nil {
			return nil, fmt.Errorf("failed to create message node: %v", err)
		}
		
		// Link message to user
		linkQuery := `
			MATCH (u:User {userId: $userId})
			MATCH (m:Message {messageId: $messageId})
			CREATE (u)-[:OWNS]->(m)
			RETURN u, m
		`
		linkParams := map[string]any{
			"userId":    userID,
			"messageId": message.MessageID,
		}
		
		_, err = tx.Run(linkQuery, linkParams)
		if err != nil {
			return nil, fmt.Errorf("failed to link message to user: %v", err)
		}
		
		// Update user's last active timestamp if it's a human message
		if message.Sender == "human" {
			updateQuery := `
				MATCH (u:User {userId: $userId})
				SET u.lastActive = $lastActive
				RETURN u
			`
			updateParams := map[string]any{
				"userId":     userID,
				"lastActive": time.Now().Unix(),
			}
			
			_, err = tx.Run(updateQuery, updateParams)
			if err != nil {
				return nil, fmt.Errorf("failed to update user last active: %v", err)
			}
		}
		
		if len(message.Topics) > 0 {
			fmt.Printf("📊 Added message node to Neo4j: %s (owned by user: %s, topics: %v)\n", message.MessageID, userID, message.Topics)
		} else {
			fmt.Printf("📊 Added message node to Neo4j: %s (owned by user: %s, no topics)\n", message.MessageID, userID)
		}
		
		// Create topic nodes and link messages to them (only if topics exist)
		for _, topicName := range message.Topics {
			// Create or merge topic node
			topicQuery := `
				MERGE (t:Topic {name: $topicName})
				ON CREATE SET t.topicId = $topicId, t.createdAt = $timestamp
				RETURN t
			`
			topicParams := map[string]any{
				"topicName": topicName,
				"topicId":   generateID(),
				"timestamp": time.Now().Unix(),
			}
			
			_, err := tx.Run(topicQuery, topicParams)
			if err != nil {
				log.Printf("Failed to create topic node: %v", err)
				continue
			}
			
			// Link message to topic
			linkTopicQuery := `
				MATCH (m:Message {messageId: $messageId})
				MATCH (t:Topic {name: $topicName})
				MERGE (m)-[:BELONGS_TO]->(t)
				RETURN m, t
			`
			linkTopicParams := map[string]any{
				"messageId": message.MessageID,
				"topicName": topicName,
			}
			
			_, err = tx.Run(linkTopicQuery, linkTopicParams)
			if err != nil {
				log.Printf("Failed to link message to topic: %v", err)
			}
		}
		
		// Then, find similar messages and create edges
		similarityQuery := `
			MATCH (m1:Message {messageId: $messageId})
			MATCH (m2:Message {userId: $userId})
			WHERE m2.messageId <> $messageId
			RETURN m2.messageId as messageId, m2.embedding as embedding, m2.content as content
		`
		similarityParams := map[string]any{
			"messageId": message.MessageID,
			"userId":    userID,
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

			// Create edge if similarity > 0.5
			if similarity > 0.5 {
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

// Create a new user node
func createUser(name string) (string, error) {
	user := User{
		UserID:     generateID(),
		Name:       name,
		CreatedAt:  time.Now().Unix(),
		LastActive: time.Now().Unix(),
		Preferences: UserPreferences{
			Language:        "en",
			Tone:            "friendly",
			AddressingStyle: "you",
		},
	}
	
	fmt.Printf("🔄 Attempting to create user with ID: %s\n", user.UserID)
	
	session := neo4jDriver.NewSession(neo4j.SessionConfig{})
	defer session.Close()
	
	_, err := session.WriteTransaction(func(tx neo4j.Transaction) (any, error) {
		query := `
			CREATE (u:User {
				userId: $userId,
				name: $name,
				createdAt: $createdAt,
				lastActive: $lastActive,
				language: $language,
				tone: $tone,
				addressingStyle: $addressingStyle
			})
			RETURN u
		`
		params := map[string]any{
			"userId":          user.UserID,
			"name":            user.Name,
			"createdAt":       user.CreatedAt,
			"lastActive":      user.LastActive,
			"language":        user.Preferences.Language,
			"tone":            user.Preferences.Tone,
			"addressingStyle": user.Preferences.AddressingStyle,
		}
		
		fmt.Printf("🔄 Running Neo4j query with params: %+v\n", params)
		
		result, err := tx.Run(query, params)
		if err != nil {
			fmt.Printf("❌ Neo4j query failed: %v\n", err)
			return nil, err
		}
		
		fmt.Printf("✅ Neo4j query executed successfully\n")
		return result.Consume()
	})
	
	if err != nil {
		return "", fmt.Errorf("failed to create user: %v", err)
	}
	
	fmt.Printf("👤 Created new user: %s (ID: %s)\n", user.Name, user.UserID)
	return user.UserID, nil
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

	// Create a new user for the conversation
	fmt.Println("🔄 Creating new user...")
	userID, err := createUser("Shiny")
	if err != nil {
		log.Fatalf("Failed to create user: %v", err)
	}
	fmt.Printf("✅ User created successfully with ID: %s\n", userID)

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
		printMessageNode("human", userInput, client, userID)
		
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
		printMessageNode("ai", chatbotResponse, client, userID)

		messages = append(messages, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleAssistant,
			Content: chatbotResponse,
		})
	}

	if err := scanner.Err(); err != nil {
		log.Printf("Error reading standard input: %v", err)
	}
}
