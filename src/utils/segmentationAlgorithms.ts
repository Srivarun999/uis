import seedrandom from 'seedrandom';

// Create a seeded random number generator with a fixed seed
const rng = seedrandom('cluster-tendency-2024');

// Add this helper function for deterministic sampling
function deterministicSample<T>(array: T[], sampleSize: number): T[] {
  const result: T[] = [];
  const indices = new Array(array.length).fill(0).map((_, i) => i);
  
  for (let i = 0; i < sampleSize && indices.length > 0; i++) {
    const randomIndex = Math.floor(rng() * indices.length);
    const selectedIndex = indices.splice(randomIndex, 1)[0];
    result.push(array[selectedIndex]);
  }
  
  return result;
}

// Add at the top of the file
type Pixel = [number, number, number];
type RGBArray = number[];

// Add at the top of file after imports
type SegmentationResult = {
  imageData: ImageData;
  labels: number[];
  centroids: number[][];
};

// Helper function for k-means++ initialization
function kmeansppInit(pixels: number[][], k: number, rng: () => number): number[][] {
  const centroids: number[][] = [];
  
  // Choose first centroid randomly
  const firstIdx = Math.floor(rng() * pixels.length);
  centroids.push([...pixels[firstIdx]]);

  // Choose remaining centroids
  for (let i = 1; i < k; i++) {
    const distances = pixels.map(pixel => {
      // Find minimum distance to any existing centroid
      return Math.min(...centroids.map(c => euclideanDistanceSquared(pixel, c)));
    });
    
    // Choose next centroid with probability proportional to distance squared
    const sum = distances.reduce((a, b) => a + b, 0);
    let rand = rng() * sum;
    let idx = 0;
    while (rand > 0 && idx < distances.length) {
      rand -= distances[idx];
      idx++;
    }
    centroids.push([...pixels[idx - 1]]);
  }
  
  return centroids;
}

// K-means clustering implementation with k-means++
export const kMeansSegmentation = (imageData: ImageData, k: number): SegmentationResult => {
  // Validation at the start
  if (!imageData || !imageData.data) {
    throw new Error('Invalid ImageData');
  }
  if (k <= 0 || k > imageData.width * imageData.height) {
    throw new Error('Invalid number of clusters');
  }
  
  const { width, height, data } = imageData;
  const totalPixels = width * height;
  const sampleRate = totalPixels > 10000 ? 8 : 4;
  const pixels: number[][] = [];
  const pixelMap: number[] = [];

  for (let i = 0; i < data.length; i += 4 * sampleRate) {
    const pixelIndex = i / 4;
    pixels.push([data[i], data[i + 1], data[i + 2]]);
    pixelMap.push(pixelIndex);
  }

  let bestCentroids: number[][] = [];
  let bestLabels: number[] = [];
  let bestInertia = Infinity;
  let currentInertia = 0;

  // Run k-means multiple times
  const n_init = 10;
  for (let run = 0; run < n_init; run++) {
    // Initialize centroids using k-means++
    const centroids = kmeansppInit(pixels, k, rng);
    let labels: number[] = new Array(pixels.length).fill(0);
    
    // K-means iterations
    for (let iter = 0; iter < 5; iter++) {
      let changed = false;
      currentInertia = 0;

      // Assign pixels to nearest centroid
      for (let i = 0; i < pixels.length; i++) {
        let minDist = Infinity;
        let closestCluster = 0;

        for (let j = 0; j < k; j++) {
          const dist = euclideanDistanceSquared(pixels[i], centroids[j]);
          if (dist < minDist) {
            minDist = dist;
            closestCluster = j;
          }
        }

        currentInertia += minDist;
        if (labels[i] !== closestCluster) {
          changed = true;
          labels[i] = closestCluster;
        }
      }

      if (!changed) break;

      // Update centroids
      for (let j = 0; j < k; j++) {
        const clusterPixels = pixels.filter((_, i) => labels[i] === j);
        if (clusterPixels.length > 0) {
          centroids[j] = [
            clusterPixels.reduce((sum, p) => sum + p[0], 0) / clusterPixels.length,
            clusterPixels.reduce((sum, p) => sum + p[1], 0) / clusterPixels.length,
            clusterPixels.reduce((sum, p) => sum + p[2], 0) / clusterPixels.length
          ];
        }
      }
    }

    // Keep best result
    if (currentInertia < bestInertia) {
      bestInertia = currentInertia;
      bestCentroids = centroids.map(c => [...c]);
      bestLabels = [...labels];
    }
  }

  // Use best result for final assignment
  const allLabels: number[] = new Array(totalPixels);
  for (let i = 0; i < totalPixels; i++) {
    const pixelIndex = i * 4;
    const pixel = [data[pixelIndex], data[pixelIndex + 1], data[pixelIndex + 2]];
    let closestCluster = 0;
    let minDist = Infinity;

    for (let j = 0; j < k; j++) {
      const dist = euclideanDistance(pixel, bestCentroids[j]);
      if (dist < minDist) {
        minDist = dist;
        closestCluster = j;
      }
    }

    allLabels[i] = closestCluster;
  }

  // Create output image
  const segmented = new ImageData(width, height);
  const clusterColors = generateClusterColors(k);

  for (let i = 0; i < allLabels.length; i++) {
    const cluster = allLabels[i];
    const pixelIndex = i * 4;

    segmented.data[pixelIndex] = clusterColors[cluster][0];
    segmented.data[pixelIndex + 1] = clusterColors[cluster][1];
    segmented.data[pixelIndex + 2] = clusterColors[cluster][2];
    segmented.data[pixelIndex + 3] = 255;
  }

  return { imageData: segmented, labels: allLabels, centroids: bestCentroids };
};

// Sampled DBSCAN clustering implementation (browser-friendly)
export const dbscanSegmentation = (
  imageData: ImageData,
  epsilon: number,
  minSamples: number
): SegmentationResult => {
  // Validation at the start
  if (!imageData || !imageData.data) {
    throw new Error('Invalid ImageData');
  }
  if (epsilon <= 0 || minSamples <= 0) {
    throw new Error('Invalid epsilon or minSamples value');
  }
  
  const { width, height, data } = imageData;
  const totalPixels = width * height;

  // 1. Sample a manageable number of pixels (e.g., 2000)
  const sampleSize = Math.min(2000, totalPixels);
  const sampleIndices = deterministicSample(
    Array.from({length: totalPixels}, (_, i) => i),
    sampleSize
  );
  
  const sampledPixels = sampleIndices.map(i => {
    const idx = i * 4;
    return {
      rgb: [data[idx], data[idx + 1], data[idx + 2]],
      index: i
    };
  });
  
  // 2. Run DBSCAN on the sampled pixels
  const labels: number[] = new Array(sampledPixels.length).fill(-1);
  let clusterId = 0;
  const epsilonScaled = epsilon * 100;

  function getNeighborsDBSCAN(idx: number): number[] {
    const neighbors: number[] = [];
    const p = sampledPixels[idx].rgb;
    for (let i = 0; i < sampledPixels.length; i++) {
      if (i === idx) continue;
      if (euclideanDistance(p, sampledPixels[i].rgb) <= epsilonScaled) {
        neighbors.push(i);
      }
    }
    return neighbors;
  }

  for (let i = 0; i < sampledPixels.length; i++) {
    if (labels[i] !== -1) continue;
    const neighbors = getNeighborsDBSCAN(i);
    if (neighbors.length < minSamples) {
      labels[i] = -2; // noise
      continue;
    }
    labels[i] = clusterId;
    const queue = [...neighbors];
    while (queue.length > 0) {
      const idx = queue.shift()!;
      if (labels[idx] === -2) labels[idx] = clusterId;
      if (labels[idx] !== -1) continue;
      labels[idx] = clusterId;
      const idxNeighbors = getNeighborsDBSCAN(idx);
      if (idxNeighbors.length >= minSamples) {
        for (const n of idxNeighbors) {
          if (!queue.includes(n)) queue.push(n);
        }
      }
    }
    clusterId++;
  }

  // 3. Compute centroids for clusters
  const centroids: number[][] = [];
  for (let c = 0; c < clusterId; c++) {
    const clusterPixels = sampledPixels.filter((_, i) => labels[i] === c);
    if (clusterPixels.length === 0) {
      centroids.push([128, 128, 128]);
      continue;
    }
    centroids.push([
      clusterPixels.reduce((sum, p) => sum + p.rgb[0], 0) / clusterPixels.length,
      clusterPixels.reduce((sum, p) => sum + p.rgb[1], 0) / clusterPixels.length,
      clusterPixels.reduce((sum, p) => sum + p.rgb[2], 0) / clusterPixels.length,
    ]);
  }

  // 4. Assign all pixels in the image to the nearest centroid (or black for noise)
  const allLabels: number[] = new Array(totalPixels).fill(-2);
  for (let i = 0; i < totalPixels; i++) {
    const pixelIndex = i * 4;
    const pixel = [data[pixelIndex], data[pixelIndex + 1], data[pixelIndex + 2]];
    let closestCluster = -2;
    let minDist = Infinity;
    for (let c = 0; c < centroids.length; c++) {
      const dist = euclideanDistance(pixel, centroids[c]);
      if (dist < minDist) {
        minDist = dist;
        closestCluster = c;
      }
    }
    // Only assign to cluster if minDist is within epsilon, else keep as noise
    allLabels[i] = minDist <= epsilonScaled ? closestCluster : -2;
  }

  // 5. Generate output image
  const segmented = new ImageData(width, height);
  const clusterColors = generateClusterColors(Math.max(clusterId, 1));
  for (let i = 0; i < totalPixels; i++) {
    const label = allLabels[i];
    const color = label >= 0 ? clusterColors[label % clusterColors.length] : [0, 0, 0];
    const idx = i * 4;
    segmented.data[idx] = color[0];
    segmented.data[idx + 1] = color[1];
    segmented.data[idx + 2] = color[2];
    segmented.data[idx + 3] = 255;  // Make sure alpha channel is set
  }

  return {
    imageData: segmented,
    labels: allLabels,
    centroids,
  };
};

// Mean Shift clustering implementation
export const meanShiftSegmentation = (imageData: ImageData, bandwidth: number): SegmentationResult => {
  // Validation at the start
  if (!imageData || !imageData.data) {
    throw new Error('Invalid ImageData');
  }
  if (bandwidth <= 0) {
    throw new Error('Invalid bandwidth value');
  }
  
  const { width, height, data } = imageData;
  const totalPixels = width * height;

  // Heavy sampling - use only 1000 pixels for mean shift
  const maxSamples = Math.min(1000, totalPixels);
  const sampleIndices = deterministicSample(
    Array.from({length: totalPixels}, (_, i) => i),
    maxSamples
  );
  
  const pixels: number[][] = sampleIndices.map(i => {
    const idx = i * 4;
    return [data[idx], data[idx + 1], data[idx + 2]];
  });

  const modes: number[][] = [];
  const bandwidthSq = Math.pow(bandwidth * 50, 2);

  // Find modes for a subset of pixels
  const modeSearchLimit = Math.min(50, pixels.length);

  for (let i = 0; i < modeSearchLimit; i++) {
    let currentPoint = [...pixels[i]];

    // Mean shift iterations (limited)
    for (let iter = 0; iter < 3; iter++) {
      const neighbors: number[][] = [];

      for (const pixel of pixels) {
        const dist = euclideanDistanceSquared(pixel, currentPoint);
        if (dist <= bandwidthSq) {
          neighbors.push(pixel);
        }
      }

      if (neighbors.length === 0) break;

      const newPoint = [
        neighbors.reduce((sum, p) => sum + p[0], 0) / neighbors.length,
        neighbors.reduce((sum, p) => sum + p[1], 0) / neighbors.length,
        neighbors.reduce((sum, p) => sum + p[2], 0) / neighbors.length
      ];

      const shift = euclideanDistance(newPoint, currentPoint);
      currentPoint = newPoint;

      if (shift < 5) break;
    }

    // Check if mode already exists
    let found = false;
    for (const mode of modes) {
      if (euclideanDistance(mode, currentPoint) < bandwidth * 30) {
        found = true;
        break;
      }
    }

    if (!found) {
      modes.push(currentPoint);
    }
  }

  // Ensure at least one mode
  if (modes.length === 0) {
    modes.push([128, 128, 128]);
  }

  // Assign all pixels to nearest mode
  const allLabels: number[] = new Array(totalPixels);
  for (let i = 0; i < totalPixels; i++) {
    const pixelIndex = i * 4;
    const pixel = [data[pixelIndex], data[pixelIndex + 1], data[pixelIndex + 2]];
    let closestMode = 0;
    let minDist = Infinity;

    for (let m = 0; m < modes.length; m++) {
      const dist = euclideanDistance(pixel, modes[m]);
      if (dist < minDist) {
        minDist = dist;
        closestMode = m;
      }
    }

    allLabels[i] = closestMode;
  }

  // Create output image
  const segmented = new ImageData(width, height);
  const clusterColors = generateClusterColors(modes.length);

  for (let i = 0; i < allLabels.length; i++) {
    const cluster = allLabels[i];
    const pixelIndex = i * 4;

    segmented.data[pixelIndex] = clusterColors[cluster][0];
    segmented.data[pixelIndex + 1] = clusterColors[cluster][1];
    segmented.data[pixelIndex + 2] = clusterColors[cluster][2];
    segmented.data[pixelIndex + 3] = 255;
  }

  return { imageData: segmented, labels: allLabels, centroids: modes };
};

// Helper functions

function euclideanDistance(p1: number[], p2: number[]): number {
  return Math.sqrt(
    p1.reduce((sum, val, idx) => sum + Math.pow(val - (p2[idx] || 0), 2), 0)
  );
}

function euclideanDistanceSquared(p1: number[], p2: number[]): number {
  return p1.reduce((sum, val, idx) => sum + Math.pow(val - (p2[idx] || 0), 2), 0);
}

function generateClusterColors(numClusters: number): number[][] {
  const colors: number[][] = [];
  for (let i = 0; i < numClusters; i++) {
    const hue = (i * 360) / numClusters;
    const rgb = hslToRgb(hue / 360, 0.8, 0.6);
    colors.push(rgb);
  }
  return colors;
}

function hslToRgb(h: number, s: number, l: number): number[] {
  let r, g, b;
  if (s === 0) {
    r = g = b = l;
  } else {
    const hue2rgb = (p: number, q: number, t: number) => {
      if (t < 0) t += 1;
      if (t > 1) t -= 1;
      if (t < 1 / 6) return p + (q - p) * 6 * t;
      if (t < 1 / 2) return q;
      if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
      return p;
    };
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    r = hue2rgb(p, q, h + 1 / 3);
    g = hue2rgb(p, q, h);
    b = hue2rgb(p, q, h - 1 / 3);
  }
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}


function processImage(imageData: ImageData): boolean {
  try {
    if (!imageData || !imageData.data || 
        imageData.width <= 0 || 
        imageData.height <= 0 || 
        imageData.data.length !== imageData.width * imageData.height * 4) {
      return false;
    }
    return true;
  } catch (error) {
    console.error('Error processing image:', error);
    return false;
  }
}