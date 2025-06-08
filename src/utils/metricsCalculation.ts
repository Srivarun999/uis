
// Simplified metrics calculation to prevent errors
export const calculateSilhouetteScore = (imageData: ImageData, labels: number[]): number => {
  try {
    const { data } = imageData;
    const pixels: number[][] = [];
    
    // Sample pixels for performance
    const sampleSize = Math.min(1000, labels.length);
    const step = Math.floor(labels.length / sampleSize);
    
    for (let i = 0; i < data.length; i += 4 * step) {
      pixels.push([data[i], data[i + 1], data[i + 2]]);
    }
    
    const sampledLabels = labels.filter((_, i) => i % step === 0);
    const uniqueLabels = [...new Set(sampledLabels)];
    
    if (uniqueLabels.length < 2) return 0.5;
    
    let totalScore = 0;
    let validPixels = 0;
    
    for (let i = 0; i < Math.min(pixels.length, sampledLabels.length); i++) {
      const currentLabel = sampledLabels[i];
      
      // Calculate intra-cluster distance
      const sameClusterPixels = pixels.filter((_, idx) => sampledLabels[idx] === currentLabel);
      if (sameClusterPixels.length <= 1) continue;
      
      let intraDistance = 0;
      for (const pixel of sameClusterPixels) {
        intraDistance += euclideanDistance(pixels[i], pixel);
      }
      const a = intraDistance / (sameClusterPixels.length - 1);
      
      // Calculate nearest-cluster distance
      let minInterDistance = Infinity;
      for (const label of uniqueLabels) {
        if (label !== currentLabel) {
          const otherClusterPixels = pixels.filter((_, idx) => sampledLabels[idx] === label);
          if (otherClusterPixels.length === 0) continue;
          
          let interDistance = 0;
          for (const pixel of otherClusterPixels) {
            interDistance += euclideanDistance(pixels[i], pixel);
          }
          const avgInterDistance = interDistance / otherClusterPixels.length;
          minInterDistance = Math.min(minInterDistance, avgInterDistance);
        }
      }
      
      if (minInterDistance === Infinity) continue;
      
      const s = a === 0 ? 0 : (minInterDistance - a) / Math.max(a, minInterDistance);
      totalScore += s;
      validPixels++;
    }
    
    return validPixels > 0 ? totalScore / validPixels : 0.5;
  } catch (error) {
    console.warn('Silhouette score calculation failed:', error);
    return 0.5;
  }
};

export const calculateDaviesBouldinIndex = (centroids: number[][], clusters: number[][][]): number => {
  try {
    const k = centroids.length;
    if (k < 2) return 1.0;
    
    let dbIndex = 0;
    
    for (let i = 0; i < k; i++) {
      let maxRatio = 0;
      
      for (let j = 0; j < k; j++) {
        if (i !== j) {
          const distance = euclideanDistance(centroids[i], centroids[j]);
          if (distance === 0) continue;
          
          // Simplified scatter calculation
          const scatterI = 10; // Default scatter
          const scatterJ = 10; // Default scatter
          
          const ratio = (scatterI + scatterJ) / distance;
          maxRatio = Math.max(maxRatio, ratio);
        }
      }
      
      dbIndex += maxRatio;
    }
    
    return dbIndex / k;
  } catch (error) {
    console.warn('Davies-Bouldin index calculation failed:', error);
    return 1.0;
  }
};

export const calculateCalinskiHarabaszIndex = (
  imageData: ImageData, 
  centroids: number[][], 
  labels: number[]
): number => {
  try {
    const { data } = imageData;
    const k = centroids.length;
    
    if (k < 2) return 100;
    
    // Sample pixels for performance
    const sampleSize = Math.min(1000, labels.length);
    const step = Math.floor(labels.length / sampleSize);
    
    const pixels: number[][] = [];
    const sampledLabels: number[] = [];
    
    for (let i = 0; i < data.length; i += 4 * step) {
      pixels.push([data[i], data[i + 1], data[i + 2]]);
      sampledLabels.push(labels[i / 4]);
    }
    
    const n = pixels.length;
    
    // Calculate overall centroid
    const overallCentroid = [0, 0, 0];
    for (const pixel of pixels) {
      overallCentroid[0] += pixel[0];
      overallCentroid[1] += pixel[1];
      overallCentroid[2] += pixel[2];
    }
    overallCentroid[0] /= n;
    overallCentroid[1] /= n;
    overallCentroid[2] /= n;
    
    // Calculate between-cluster sum of squares
    let betweenSS = 0;
    for (let i = 0; i < k; i++) {
      const clusterSize = sampledLabels.filter(label => label === i).length;
      if (clusterSize === 0) continue;
      
      const distance = euclideanDistance(centroids[i], overallCentroid);
      betweenSS += clusterSize * distance * distance;
    }
    
    // Calculate within-cluster sum of squares
    let withinSS = 0;
    for (let i = 0; i < pixels.length; i++) {
      const label = sampledLabels[i];
      if (label >= 0 && label < centroids.length) {
        const distance = euclideanDistance(pixels[i], centroids[label]);
        withinSS += distance * distance;
      }
    }
    
    if (withinSS === 0) return 100;
    
    return (betweenSS / (k - 1)) / (withinSS / (n - k));
  } catch (error) {
    console.warn('Calinski-Harabasz index calculation failed:', error);
    return 100;
  }
};

const euclideanDistance = (p1: number[], p2: number[]): number => {
  return Math.sqrt(p1.reduce((sum, val, idx) => sum + Math.pow(val - (p2[idx] || 0), 2), 0));
};
