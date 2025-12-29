package main

import (
	"bufio"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

type TileData struct {
	Name  string
	Count int64
}

type XYPair struct {
	X     int
	Y     int
	Count int64
}

type LibData struct {
	Name       string
	Count      int64
	OutputPath string
}

type ReadID struct {
	Tile string
	X    int
	Y    int
}

type ReadIDList []ReadID

func (r ReadIDList) Len() int { return len(r) }
func (r ReadIDList) Less(i, j int) bool {
	if r[i].Tile != r[j].Tile {
		return r[i].Tile < r[j].Tile
	}
	if r[i].Y != r[j].Y {
		return r[i].Y < r[j].Y
	}
	return r[i].X < r[j].X
}
func (r ReadIDList) Swap(i, j int) { r[i], r[j] = r[j], r[i] }

var (
	version = "0.1.2"
)

func main() {
	libFile := flag.String("lib", "", "Library data file")
	tileFile := flag.String("tile", "", "Tile data file (default: tile_data.txt in program directory)")
	posFile := flag.String("pos", "", "XY coordinate pairs file (default: xy_comp.txt in program directory)")
	showVersion := flag.Bool("version", false, "Show version information")
	showVersionShort := flag.Bool("v", false, "Show version information (short)")
	flag.Parse()

	// Show version if requested
	if *showVersion || *showVersionShort {
		fmt.Printf("version %s\n", version)
		os.Exit(0)
	}

	// Get program directory
	execPath, err := os.Executable()
	if err != nil {
		// Fallback: use current working directory
		execPath, _ = os.Getwd()
	}
	progDir := filepath.Dir(execPath)

	// Set default values for tile and pos if not provided
	if *tileFile == "" {
		*tileFile = filepath.Join(progDir, "tile_data.txt")
	}
	if *posFile == "" {
		*posFile = filepath.Join(progDir, "xy_comp.txt")
	}

	if *libFile == "" {
		fmt.Fprintf(os.Stderr, "version %s\n\n", version)
		fmt.Fprintf(os.Stderr, "Usage: %s -lib <lib_data.txt> [-tile <tile_data.txt>] [-pos <xy_comp.txt>] -o <output_directory>\n", os.Args[0])
		fmt.Fprintf(os.Stderr, " \tlib_data.txt format: lib_name reads_count output_file_path\n")
		fmt.Fprintf(os.Stderr, "\t-tile: optional, defaults to tile_data.txt in program directory\n")
		fmt.Fprintf(os.Stderr, "\t-pos: optional, defaults to xy_comp.txt in program directory\n")
		fmt.Fprintf(os.Stderr, "\t-o: optional, output directory (default: ./)\n")
		os.Exit(1)
	}

	// Check if output directory exists
	outputDir := flag.String("o", "./", "output directory")
	if _, err := os.Stat(*outputDir); os.IsNotExist(err) {
		fmt.Fprintf(os.Stderr, "Error: output directory %s does not exist\n", *outputDir)
		os.Exit(1)
	}
	// Initialize random seed
	rand.Seed(time.Now().UnixNano())

	// Read input files
	tiles, err := readTileData(*tileFile)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading tile data: %v\n", err)
		os.Exit(1)
	}

	xyPairs, err := readXYData(*posFile)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading XY data: %v\n", err)
		os.Exit(1)
	}

	libs, err := readLibData(*libFile)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading lib data: %v\n", err)
		os.Exit(1)
	}

	// Create output directories for each lib
	for _, lib := range libs {
		if lib.OutputPath == "" {
			fmt.Fprintf(os.Stderr, "Error: output path not specified for library %s\n", lib.Name)
			os.Exit(1)
		}
		// Create directory if it doesn't exist
		outputDir := filepath.Dir(lib.OutputPath)
		if outputDir != "" && outputDir != "." {
			if err := os.MkdirAll(outputDir, 0755); err != nil {
				fmt.Fprintf(os.Stderr, "Error creating output directory for %s: %v\n", lib.Name, err)
				os.Exit(1)
			}
		}
	}

	// Calculate total reads across all libraries
	var totalLibReads int64
	for _, lib := range libs {
		totalLibReads += lib.Count
	}

	// Calculate tile probabilities
	tileProbs := make(map[string]float64)
	var totalTileReads int64
	for _, tile := range tiles {
		totalTileReads += tile.Count
	}
	for _, tile := range tiles {
		tileProbs[tile.Name] = float64(tile.Count) / float64(totalTileReads)
	}

	// Calculate XY pair probabilities
	xyProbs := make([]struct {
		xyPair XYPair
		prob   float64
	}, 0, len(xyPairs))
	var totalXYReads int64
	for _, xy := range xyPairs {
		totalXYReads += xy.Count
	}
	for _, xy := range xyPairs {
		xyProbs = append(xyProbs, struct {
			xyPair XYPair
			prob   float64
		}{
			xyPair: xy,
			prob:   float64(xy.Count) / float64(totalXYReads),
		})
	}

	// Track used X-Y pairs per tile to ensure no duplicates
	tileUsedPairs := make(map[string]map[string]bool) // tile -> "X:Y" -> true
	for _, tile := range tiles {
		tileUsedPairs[tile.Name] = make(map[string]bool)
	}

	// Step 1: Pre-calculate lib assignments for all tiles
	// libTileAssignments: libName -> tileName -> reads count
	fmt.Println("\nPre-calculating lib assignments for all tiles...")
	libTileAssignments := make(map[string]map[string]int64)

	for _, lib := range libs {
		fmt.Printf("Allocating reads for library: %s (%d reads)\n", lib.Name, lib.Count)
		libProportion := float64(lib.Count) / float64(totalLibReads)

		// Calculate target reads per tile (with 1% variance)
		tileTargets := make([]struct {
			tileName string
			target   float64
		}, len(tiles))
		var totalTargeted float64

		for i, tile := range tiles {
			expectedReads := float64(tile.Count) * libProportion
			// Add 1% variance: random between -1% and +1%
			variance := rand.Float64()*0.02 - 0.01 // -1% to +1%
			tileTargets[i] = struct {
				tileName string
				target   float64
			}{
				tileName: tile.Name,
				target:   expectedReads * (1.0 + variance),
			}
			totalTargeted += tileTargets[i].target
		}

		// Normalize to match library count exactly
		if totalTargeted > 0 {
			normalizationFactor := float64(lib.Count) / totalTargeted
			for i := range tileTargets {
				tileTargets[i].target *= normalizationFactor
			}
		}

		// Assign reads to tiles using weighted rounding (largest remainder method)
		tileAssignment := make(map[string]int64)
		remainders := make([]struct {
			tileName  string
			remainder float64
			index     int
		}, len(tiles))

		var totalAssigned int64
		for i, item := range tileTargets {
			assigned := int64(item.target)
			tileAssignment[item.tileName] = assigned
			totalAssigned += assigned
			remainders[i] = struct {
				tileName  string
				remainder float64
				index     int
			}{
				tileName:  item.tileName,
				remainder: item.target - float64(assigned),
				index:     i,
			}
		}

		// Distribute remaining reads to tiles with largest remainders
		remainingReads := lib.Count - totalAssigned
		if remainingReads > 0 {
			sort.Slice(remainders, func(i, j int) bool {
				return remainders[i].remainder > remainders[j].remainder
			})
			for i := 0; i < int(remainingReads) && i < len(remainders); i++ {
				tileAssignment[remainders[i].tileName]++
			}
		}

		libTileAssignments[lib.Name] = tileAssignment
	}

	// Step 2: Process each tile sequentially and assign X-Y pairs
	// Multi-threaded processing with ordered output
	fmt.Println("\nProcessing tiles and generating X-Y pairs...")

	// Create lib name to output path mapping
	libOutputPaths := make(map[string]string)
	for _, lib := range libs {
		libOutputPaths[lib.Name] = lib.OutputPath
	}

	// Open output files for all libs (for real-time writing)
	libWriters := make(map[string]*bufio.Writer)
	libFiles := make(map[string]*os.File)
	for _, lib := range libs {
		outputFile := lib.OutputPath
		file, err := os.Create(outputFile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error creating output file for %s: %v\n", lib.Name, err)
			continue
		}
		libFiles[lib.Name] = file
		libWriters[lib.Name] = bufio.NewWriter(file)
	}

	// Multi-threaded tile processing with ordered output
	numWorkers := 10 // Fixed number of worker threads
	if numWorkers > len(tiles) {
		numWorkers = len(tiles)
	}

	// Channel for tile tasks
	tileTaskChan := make(chan struct {
		index int
		tile  TileData
	}, len(tiles))

	// Channel for tile results (small buffer to reduce memory usage)
	// Changed from len(tiles) to fixed 50 to prevent memory accumulation
	const resultChanBuffer = 50
	tileResultChan := make(chan struct {
		index int
		tile  TileData
		data  map[string][]struct {
			X int
			Y int
		}
		isLastBatch bool // Indicates if this is the last batch for this tile
	}, resultChanBuffer)

	// Batch size for processing large tiles to reduce memory usage
	const batchSize = 100000 // Process 100k reads per batch

	// Worker pool
	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for task := range tileTaskChan {
				// Process this tile
				tile := task.tile
				tileIdx := task.index

				// Collect lib assignments for this tile
				type libAssignment struct {
					libName string
					count   int64
				}
				tileLibAssignments := make([]libAssignment, 0)
				var totalReads int64
				for _, lib := range libs {
					count := libTileAssignments[lib.Name][tile.Name]
					if count > 0 {
						tileLibAssignments = append(tileLibAssignments, libAssignment{
							libName: lib.Name,
							count:   count,
						})
						totalReads += count
					}
				}

				if len(tileLibAssignments) == 0 {
					// Empty tile, send empty result
					tileResultChan <- struct {
						index int
						tile  TileData
						data  map[string][]struct {
							X int
							Y int
						}
						isLastBatch bool
					}{
						index: tileIdx,
						tile:  tile,
						data: make(map[string][]struct {
							X int
							Y int
						}),
						isLastBatch: true,
					}
					continue
				}

				// Shuffle lib assignments to randomize order
				rand.Shuffle(len(tileLibAssignments), func(i, j int) {
					tileLibAssignments[i], tileLibAssignments[j] = tileLibAssignments[j], tileLibAssignments[i]
				})

				// Create a queue of lib assignments (but don't expand all at once for large tiles)
				// For large tiles, we'll process in batches
				if totalReads <= int64(batchSize) {
					// Small tile: process all at once (original logic)
					libQueue := make([]libAssignment, 0, totalReads)
					for _, la := range tileLibAssignments {
						for i := int64(0); i < la.count; i++ {
							libQueue = append(libQueue, libAssignment{libName: la.libName, count: 1})
						}
					}
					// Shuffle the queue to randomize assignment order
					rand.Shuffle(len(libQueue), func(i, j int) {
						libQueue[i], libQueue[j] = libQueue[j], libQueue[i]
					})

					// Select X-Y pairs
					usedPairs := tileUsedPairs[tile.Name]
					tileXYPairs := selectMultipleXYPairsOrdered(xyProbs, usedPairs, len(libQueue))

					if len(tileXYPairs) < len(libQueue) {
						fmt.Fprintf(os.Stderr, "\n  Warning: Only %d XY pairs available for tile %s, need %d\n",
							len(tileXYPairs), tile.Name, len(libQueue))
					}

					// Assign X-Y pairs to libs
					tileResult := make(map[string][]struct {
						X int
						Y int
					})
					for i := 0; i < len(libQueue) && i < len(tileXYPairs); i++ {
						libName := libQueue[i].libName
						xy := tileXYPairs[i]
						tileResult[libName] = append(tileResult[libName], struct {
							X int
							Y int
						}{X: xy.X, Y: xy.Y})
					}

					// Sort each lib's XY pairs by Y, X
					for libName := range tileResult {
						sort.Slice(tileResult[libName], func(i, j int) bool {
							if tileResult[libName][i].Y != tileResult[libName][j].Y {
								return tileResult[libName][i].Y < tileResult[libName][j].Y
							}
							return tileResult[libName][i].X < tileResult[libName][j].X
						})
					}

					// Send result back
					tileResultChan <- struct {
						index int
						tile  TileData
						data  map[string][]struct {
							X int
							Y int
						}
						isLastBatch bool
					}{
						index:       tileIdx,
						tile:        tile,
						data:        tileResult,
						isLastBatch: true,
					}
				} else {
					// Large tile: process in batches
					// Create lib assignment iterator
					type libIterator struct {
						libName   string
						remaining int64
						index     int
					}
					libIterators := make([]libIterator, 0, len(tileLibAssignments))
					for i, la := range tileLibAssignments {
						libIterators = append(libIterators, libIterator{
							libName:   la.libName,
							remaining: la.count,
							index:     i,
						})
					}

					// Get usedPairs for this tile (will be shared across batches)
					usedPairs := tileUsedPairs[tile.Name]

					// Process in batches
					var processedReads int64
					batchNum := 0
					for processedReads < totalReads {
						// Determine batch size
						currentBatchSize := int64(batchSize)
						if totalReads-processedReads < currentBatchSize {
							currentBatchSize = totalReads - processedReads
						}

						// Build lib queue for this batch
						libQueue := make([]libAssignment, 0, currentBatchSize)
						for i := range libIterators {
							for libIterators[i].remaining > 0 && int64(len(libQueue)) < currentBatchSize {
								libQueue = append(libQueue, libAssignment{
									libName: libIterators[i].libName,
									count:   1,
								})
								libIterators[i].remaining--
							}
							if int64(len(libQueue)) >= currentBatchSize {
								break
							}
						}

						// Shuffle the batch queue
						rand.Shuffle(len(libQueue), func(i, j int) {
							libQueue[i], libQueue[j] = libQueue[j], libQueue[i]
						})

						// Select X-Y pairs for this batch
						tileXYPairs := selectMultipleXYPairsOrdered(xyProbs, usedPairs, len(libQueue))

						if len(tileXYPairs) < len(libQueue) {
							fmt.Fprintf(os.Stderr, "\n  Warning: Only %d XY pairs available for tile %s batch %d, need %d\n",
								len(tileXYPairs), tile.Name, batchNum, len(libQueue))
						}

						// Assign X-Y pairs to libs for this batch
						batchResult := make(map[string][]struct {
							X int
							Y int
						})
						for i := 0; i < len(libQueue) && i < len(tileXYPairs); i++ {
							libName := libQueue[i].libName
							xy := tileXYPairs[i]
							batchResult[libName] = append(batchResult[libName], struct {
								X int
								Y int
							}{X: xy.X, Y: xy.Y})
						}

						// Sort each lib's XY pairs by Y, X
						for libName := range batchResult {
							sort.Slice(batchResult[libName], func(i, j int) bool {
								if batchResult[libName][i].Y != batchResult[libName][j].Y {
									return batchResult[libName][i].Y < batchResult[libName][j].Y
								}
								return batchResult[libName][i].X < batchResult[libName][j].X
							})
						}

						// Send batch result
						isLastBatch := processedReads+currentBatchSize >= totalReads
						tileResultChan <- struct {
							index int
							tile  TileData
							data  map[string][]struct {
								X int
								Y int
							}
							isLastBatch bool
						}{
							index:       tileIdx,
							tile:        tile,
							data:        batchResult,
							isLastBatch: isLastBatch,
						}

						processedReads += currentBatchSize
						batchNum++

						// Clear batchResult to free memory immediately
						batchResult = nil
					}

					// Clear usedPairs for this tile after processing (free memory)
					// Note: We keep it during processing to avoid duplicates across batches
					// But we can't clear it here because other batches might still reference it
					// Actually, since we process all batches sequentially, we can clear it after the last batch
					// But we need to track this differently. For now, we'll clear it in the writer after last batch.
				}
			}
		}()
	}

	// Send all tile tasks to workers
	go func() {
		for idx, tile := range tiles {
			tileTaskChan <- struct {
				index int
				tile  TileData
			}{index: idx, tile: tile}
		}
		close(tileTaskChan)
	}()

	// Writer goroutine: collect results in order and write to files
	// Support batch processing: accumulate batches for each tile until isLastBatch
	type tileBatchData struct {
		batches []map[string][]struct {
			X int
			Y int
		}
		isComplete bool
	}
	resultsMap := make(map[int]*tileBatchData)
	var resultsMutex sync.Mutex
	var nextWriteIndex int
	var writerWg sync.WaitGroup

	writerWg.Add(1)
	go func() {
		defer writerWg.Done()
		completed := 0
		totalTiles := len(tiles)
		lastProgressPercent := -1

		for result := range tileResultChan {
			resultsMutex.Lock()

			// Initialize tile batch data if not exists
			if resultsMap[result.index] == nil {
				resultsMap[result.index] = &tileBatchData{
					batches: make([]map[string][]struct {
						X int
						Y int
					}, 0),
					isComplete: false,
				}
			}

			// Append batch data
			resultsMap[result.index].batches = append(resultsMap[result.index].batches, result.data)
			resultsMap[result.index].isComplete = result.isLastBatch

			// Write results in order
			for {
				nextTileData, exists := resultsMap[nextWriteIndex]
				if !exists || !nextTileData.isComplete {
					break
				}

				// Write all batches for this tile to all libs
				for _, batchData := range nextTileData.batches {
					for libName, xyPairs := range batchData {
						writer, exists := libWriters[libName]
						if exists {
							for _, xy := range xyPairs {
								_, err := fmt.Fprintf(writer, "%s:%d:%d\n", tiles[nextWriteIndex].Name, xy.X, xy.Y)
								if err != nil {
									fmt.Fprintf(os.Stderr, "Error writing to %s: %v\n", libName, err)
								}
							}
						}
					}
				}

				// Clear tile data and free memory
				delete(resultsMap, nextWriteIndex)

				// Clear usedPairs for this tile to free memory
				if tileUsedPairs[tiles[nextWriteIndex].Name] != nil {
					tileUsedPairs[tiles[nextWriteIndex].Name] = nil
					delete(tileUsedPairs, tiles[nextWriteIndex].Name)
				}

				nextWriteIndex++
				completed++

				// Progress output
				progressPercent := (completed * 100) / totalTiles
				if progressPercent >= lastProgressPercent+10 || completed%10 == 0 || completed == totalTiles {
					fmt.Fprintf(os.Stderr, "  Processed tile %d/%d (%.1f%%)\n", completed, totalTiles, float64(completed)*100.0/float64(totalTiles))
					lastProgressPercent = progressPercent
				}

				// Flush periodically
				if completed%10 == 0 {
					for _, writer := range libWriters {
						writer.Flush()
					}
				}
			}
			resultsMutex.Unlock()
		}

		// Write any remaining results
		resultsMutex.Lock()
		for i := nextWriteIndex; i < len(tiles); i++ {
			nextTileData, exists := resultsMap[i]
			if exists && nextTileData.isComplete {
				for _, batchData := range nextTileData.batches {
					for libName, xyPairs := range batchData {
						writer, exists := libWriters[libName]
						if exists {
							for _, xy := range xyPairs {
								_, err := fmt.Fprintf(writer, "%s:%d:%d\n", tiles[i].Name, xy.X, xy.Y)
								if err != nil {
									fmt.Fprintf(os.Stderr, "Error writing to %s: %v\n", libName, err)
								}
							}
						}
					}
				}
				delete(resultsMap, i)

				// Clear usedPairs for this tile
				if tileUsedPairs[tiles[i].Name] != nil {
					tileUsedPairs[tiles[i].Name] = nil
					delete(tileUsedPairs, tiles[i].Name)
				}

				completed++

				// Progress output
				progressPercent := (completed * 100) / totalTiles
				if progressPercent >= lastProgressPercent+10 || completed%10 == 0 || completed == totalTiles {
					fmt.Fprintf(os.Stderr, "  Processed tile %d/%d (%.1f%%)\n", completed, totalTiles, float64(completed)*100.0/float64(totalTiles))
					lastProgressPercent = progressPercent
				}
			}
		}
		resultsMutex.Unlock()
	}()

	// Wait for all workers to complete
	wg.Wait()
	close(tileResultChan)

	// Wait for writer to finish
	writerWg.Wait()

	// Final flush and close all files
	fmt.Fprintf(os.Stderr, "\n  Completed processing %d tiles\n", len(tiles))
	fmt.Println("Flushing and closing output files...")
	for libName, writer := range libWriters {
		if err := writer.Flush(); err != nil {
			fmt.Fprintf(os.Stderr, "Error flushing %s: %v\n", libName, err)
		}
		if file, exists := libFiles[libName]; exists {
			file.Close()
		}
	}

	// Count total reads per lib for reporting
	fmt.Println("\nOutput file statistics:")
	for _, lib := range libs {
		var totalReads int64
		for _, tile := range tiles {
			totalReads += libTileAssignments[lib.Name][tile.Name]
		}
		fmt.Printf("  %s: %d read IDs\n", lib.Name, totalReads)
	}

	fmt.Println("Simulation complete!")
}

func readTileData(filename string) ([]TileData, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var tiles []TileData
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 2 {
			continue
		}
		count, err := strconv.ParseInt(fields[1], 10, 64)
		if err != nil {
			return nil, fmt.Errorf("invalid count in tile data: %v", err)
		}
		tiles = append(tiles, TileData{
			Name:  fields[0],
			Count: count,
		})
	}
	return tiles, scanner.Err()
}

func readXYData(filename string) ([]XYPair, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var xyPairs []XYPair
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 3 {
			continue
		}
		x, err := strconv.Atoi(fields[0])
		if err != nil {
			return nil, fmt.Errorf("invalid X coordinate: %v", err)
		}
		y, err := strconv.Atoi(fields[1])
		if err != nil {
			return nil, fmt.Errorf("invalid Y coordinate: %v", err)
		}
		count, err := strconv.ParseInt(fields[2], 10, 64)
		if err != nil {
			return nil, fmt.Errorf("invalid count in XY data: %v", err)
		}
		xyPairs = append(xyPairs, XYPair{
			X:     x,
			Y:     y,
			Count: count,
		})
	}
	return xyPairs, scanner.Err()
}

func readLibData(filename string) ([]LibData, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var libs []LibData
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 3 {
			return nil, fmt.Errorf("invalid lib data format: expected at least 3 columns (lib_name reads_count output_file_path), got %d columns", len(fields))
		}
		// Only use first 3 columns, ignore additional columns if present
		count, err := strconv.ParseInt(fields[1], 10, 64)
		if err != nil {
			return nil, fmt.Errorf("invalid count in lib data: %v", err)
		}
		libs = append(libs, LibData{
			Name:       fields[0],
			Count:      count,
			OutputPath: fields[2],
		})
	}
	return libs, scanner.Err()
}

// selectMultipleXYPairsOrdered selects multiple X-Y pairs based on probability distribution
// while maintaining the Y, X order from xy_comp.txt (which is already sorted by Y, X)
// This ensures that selected pairs are naturally sorted, avoiding the need for final sorting
func selectMultipleXYPairsOrdered(xyProbs []struct {
	xyPair XYPair
	prob   float64
}, usedPairs map[string]bool, count int) []XYPair {
	// Build available pairs list with probabilities (maintaining order)
	type weightedPair struct {
		xyPair XYPair
		weight float64
		index  int // Original index to maintain order
	}

	availablePairs := make([]weightedPair, 0)
	var totalProb float64

	// First pass: collect available pairs and calculate total probability
	// Since xyProbs is already sorted by Y, X, we maintain this order
	for idx, item := range xyProbs {
		pairKey := fmt.Sprintf("%d:%d", item.xyPair.X, item.xyPair.Y)
		if !usedPairs[pairKey] {
			// Apply 1% variance to probability
			variance := rand.Float64()*0.02 - 0.01 // -1% to +1%
			adjustedProb := item.prob * (1.0 + variance)
			availablePairs = append(availablePairs, weightedPair{
				xyPair: item.xyPair,
				weight: adjustedProb,
				index:  idx,
			})
			totalProb += adjustedProb
		}
	}

	if len(availablePairs) == 0 {
		return []XYPair{}
	}

	// Normalize weights
	for i := range availablePairs {
		availablePairs[i].weight /= totalProb
	}

	// Build cumulative distribution
	cumWeights := make([]float64, len(availablePairs))
	cumWeights[0] = availablePairs[0].weight
	for i := 1; i < len(availablePairs); i++ {
		cumWeights[i] = cumWeights[i-1] + availablePairs[i].weight
	}

	// Select multiple pairs using weighted random sampling
	// We'll store pairs with their original indices for sorting
	type selectedPairWithIndex struct {
		xyPair XYPair
		index  int
	}
	selectedPairsWithIndex := make([]selectedPairWithIndex, 0, count)
	selectedIndices := make(map[int]bool)

	for len(selectedPairsWithIndex) < count && len(selectedPairsWithIndex) < len(availablePairs) {
		maxAttempts := 1000
		attempt := 0
		found := false

		for attempt < maxAttempts {
			r := rand.Float64()

			// Binary search for the selected pair
			idx := sort.Search(len(cumWeights), func(i int) bool {
				return cumWeights[i] >= r
			})
			if idx >= len(availablePairs) {
				idx = len(availablePairs) - 1
			}

			// Check if already selected
			if !selectedIndices[idx] {
				pairKey := fmt.Sprintf("%d:%d", availablePairs[idx].xyPair.X, availablePairs[idx].xyPair.Y)
				if !usedPairs[pairKey] {
					selectedPairsWithIndex = append(selectedPairsWithIndex, selectedPairWithIndex{
						xyPair: availablePairs[idx].xyPair,
						index:  availablePairs[idx].index,
					})
					selectedIndices[idx] = true
					usedPairs[pairKey] = true
					found = true
					break
				}
			}
			attempt++
		}

		if !found {
			// Fallback: find first available pair
			foundFallback := false
			for i, pair := range availablePairs {
				if !selectedIndices[i] {
					pairKey := fmt.Sprintf("%d:%d", pair.xyPair.X, pair.xyPair.Y)
					if !usedPairs[pairKey] {
						selectedPairsWithIndex = append(selectedPairsWithIndex, selectedPairWithIndex{
							xyPair: pair.xyPair,
							index:  pair.index,
						})
						selectedIndices[i] = true
						usedPairs[pairKey] = true
						foundFallback = true
						break
					}
				}
			}
			// If no more pairs available, break
			if !foundFallback {
				break
			}
		}
	}

	// Sort selected pairs by original index to maintain Y, X order
	// Since xyProbs is sorted by Y, X, maintaining index order preserves sort
	sort.Slice(selectedPairsWithIndex, func(i, j int) bool {
		return selectedPairsWithIndex[i].index < selectedPairsWithIndex[j].index
	})

	// Extract just the XYPairs
	selectedPairs := make([]XYPair, len(selectedPairsWithIndex))
	for i, sp := range selectedPairsWithIndex {
		selectedPairs[i] = sp.xyPair
	}

	return selectedPairs
}
