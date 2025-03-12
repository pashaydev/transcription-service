package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"github.com/gin-gonic/gin"
)

// TranscriptionSegment represents a segment of transcribed text with timestamp
type TranscriptionSegment struct {
	Text      string  `json:"text"`
	StartTime float64 `json:"start_time"` // in seconds
	EndTime   float64 `json:"end_time"`   // in seconds
}

// TranscriptionResponse represents the response from the Python bridge
type TranscriptionResponse struct {
	Error    string                 `json:"error,omitempty"`
	Segments []TranscriptionSegment `json:"segments"`
}

func main() {
	// Set up Gin router
	gin.SetMode(gin.ReleaseMode)
	router := gin.Default()

	// Increase timeout for HTTP server
	server := &http.Server{
		Addr:         ":" + getPort(),
		Handler:      router,
		ReadTimeout:  5 * time.Minute,
		WriteTimeout: 5 * time.Minute,
	}

	// Serve static files
	router.Static("/static", "./static")
	router.StaticFile("/", "./static/index.html")

	// Health check endpoint
	router.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"status": "ok"})
	})

	// API route for transcription
	router.POST("/api/transcribe", func(c *gin.Context) {
		startTime := time.Now()

		// Get the uploaded file
		file, err := c.FormFile("audio")
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "No audio file provided"})
			return
		}

		// Limit file size
		if file.Size > 25*1024*1024 { // 25MB limit
			c.JSON(http.StatusBadRequest, gin.H{"error": "File too large (max 25MB)"})
			return
		}

		// Create temp directory for uploaded files
		tmpDir, err := os.MkdirTemp("", "audio-upload")
		if err != nil {
			log.Printf("Error creating temp dir: %v", err)
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to create temp directory"})
			return
		}
		defer os.RemoveAll(tmpDir)

		// Save the uploaded file
		audioPath := filepath.Join(tmpDir, file.Filename)
		if err := c.SaveUploadedFile(file, audioPath); err != nil {
			log.Printf("Error saving uploaded file: %v", err)
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to save uploaded file"})
			return
		}

		log.Printf("Saved file: %s (size: %.2f MB)", audioPath, float64(file.Size)/(1024*1024))

		// Output path for the transcription
		outputPath := filepath.Join(tmpDir, "output.json")

		// Get the current directory
		currentDir, err := os.Getwd()
		if err != nil {
			log.Printf("Error getting current directory: %v", err)
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to get current directory"})
			return
		}

		// Path to the Python bridge script
		scriptPath := filepath.Join(currentDir, "whisper_bridge.py")

		// Get model size from environment variable or use default
		modelSize := os.Getenv("WHISPER_MODEL")
		if modelSize == "" {
			modelSize = "tiny" // Default to tiny model for speed and memory efficiency
		}

		// Set a timeout context - 3 minutes for processing
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
		defer cancel()

		// Prepare command with the context
		cmd := exec.CommandContext(ctx,
			"python3",
			scriptPath,
			"--input", audioPath,
			"--output", outputPath,
			"--model", modelSize,
		)

		log.Printf("Running transcription with model: %s", modelSize)

		// Run the command and collect output
		output, err := cmd.CombinedOutput()

		// Handle different error cases
		if ctx.Err() == context.DeadlineExceeded {
			log.Printf("Transcription timed out after %v", time.Since(startTime))
			c.JSON(http.StatusRequestTimeout, gin.H{
				"error": "Transcription timed out (3 minutes limit)",
			})
			return
		}

		if err != nil {
			log.Printf("Transcription error after %v: %v", time.Since(startTime), err)
			log.Printf("Command output: %s", string(output))

			// Check if output file exists despite the error
			if _, statErr := os.Stat(outputPath); statErr == nil {
				log.Printf("Output file exists despite error, trying to use it")
			} else {
				c.JSON(http.StatusInternalServerError, gin.H{
					"error":  fmt.Sprintf("Transcription failed: %v", err),
					"output": string(output),
				})
				return
			}
		}

		// Read the output file
		data, err := os.ReadFile(outputPath)
		if err != nil {
			log.Printf("Error reading output file: %v", err)
			c.JSON(http.StatusInternalServerError, gin.H{
				"error":   "Failed to read transcription results",
				"details": err.Error(),
			})
			return
		}

		// Parse the JSON response
		var response TranscriptionResponse
		if err := json.Unmarshal(data, &response); err != nil {
			log.Printf("Error parsing JSON: %v", err)
			c.JSON(http.StatusInternalServerError, gin.H{
				"error":   "Failed to parse transcription output",
				"details": err.Error(),
			})
			return
		}

		// Check if the response contains an error
		if response.Error != "" {
			log.Printf("Error from transcription service: %s", response.Error)
			if len(response.Segments) == 0 {
				c.JSON(http.StatusInternalServerError, gin.H{
					"error": response.Error,
				})
				return
			}
			// If there are segments, we'll still return them with a warning
		}

		// Return the transcription
		duration := time.Since(startTime)
		log.Printf("Transcription completed in %v with %d segments", duration, len(response.Segments))
		c.JSON(http.StatusOK, gin.H{
			"segments":                response.Segments,
			"processing_time_seconds": duration.Seconds(),
		})
	})

	// Start the server
	log.Println("Starting server on port " + getPort() + "...")
	log.Println("Using Whisper model: " + getModelName())
	if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("Failed to start server: %v", err)
	}
}

// getPort gets the port from environment variable or uses default
func getPort() string {
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}
	return port
}

// getModelName gets the configured Whisper model name
func getModelName() string {
	model := os.Getenv("WHISPER_MODEL")
	if model == "" {
		model = "tiny" // Align with the default in the handler
	}
	return model
}
