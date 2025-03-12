package transcriber

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

// TranscriptionSegment represents a segment of transcribed text with timestamp
type TranscriptionSegment struct {
	Text      string  `json:"text"`
	StartTime float64 `json:"start_time"` // in seconds
	EndTime   float64 `json:"end_time"`   // in seconds
}

// Transcriber handles audio transcription
type Transcriber struct {
	ModelPath string
}

// NewTranscriber creates a new transcriber with the given model path
func NewTranscriber(modelPath string) *Transcriber {
	return &Transcriber{
		ModelPath: modelPath,
	}
}

// Transcribe processes an audio file and returns segments with timestamps
func (t *Transcriber) Transcribe(audioPath string) ([]TranscriptionSegment, error) {
	// Create temporary directory for output
	tmpDir, err := os.MkdirTemp("", "whisper-output")
	if err != nil {
		return nil, fmt.Errorf("failed to create temp directory: %w", err)
	}
	defer os.RemoveAll(tmpDir)

	outputPath := filepath.Join(tmpDir, "output.txt")

	// Run whisper.cpp command line tool (assuming it's installed)
	// You may need to adjust this based on your actual whisper setup
	cmd := exec.Command(
		"whisper",
		"-m", t.ModelPath,
		"-f", audioPath,
		"-otxt",
		"-of", outputPath,
	)

	output, err := cmd.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("whisper transcription failed: %w, output: %s", err, string(output))
	}

	// Read the output file
	data, err := os.ReadFile(outputPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read output file: %w", err)
	}

	// Parse the output to get text segments with timestamps
	return parseTranscriptionOutput(string(data))
}

// parseTranscriptionOutput converts Whisper output to structured segments
func parseTranscriptionOutput(output string) ([]TranscriptionSegment, error) {
	var segments []TranscriptionSegment
	// Regular expression to match timestamp pattern [00:00:00 --> 00:00:00]
	re := regexp.MustCompile(`\[(\d{2}:\d{2}:\d{2}) --> (\d{2}:\d{2}:\d{2})\] (.+)`)

	lines := strings.Split(output, "\n")
	for _, line := range lines {
		matches := re.FindStringSubmatch(line)
		if len(matches) >= 4 {
			startTime, err := parseTimestamp(matches[1])
			if err != nil {
				continue
			}

			endTime, err := parseTimestamp(matches[2])
			if err != nil {
				continue
			}

			segment := TranscriptionSegment{
				Text:      matches[3],
				StartTime: startTime,
				EndTime:   endTime,
			}
			segments = append(segments, segment)
		}
	}

	return segments, nil
}

// parseTimestamp converts a timestamp string (HH:MM:SS) to seconds
func parseTimestamp(timestamp string) (float64, error) {
	parts := strings.Split(timestamp, ":")
	if len(parts) != 3 {
		return 0, fmt.Errorf("invalid timestamp format: %s", timestamp)
	}

	hours, err := strconv.Atoi(parts[0])
	if err != nil {
		return 0, err
	}

	minutes, err := strconv.Atoi(parts[1])
	if err != nil {
		return 0, err
	}

	seconds, err := strconv.Atoi(parts[2])
	if err != nil {
		return 0, err
	}

	return float64(hours*3600 + minutes*60 + seconds), nil
}
