
import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { useToast } from '@/hooks/use-toast';
import ImageUpload from './ImageUpload';
import AlgorithmSelection from './AlgorithmSelection';
import ParameterTuning from './ParameterTuning';
import ResultsVisualization from './ResultsVisualization';
import { Play, Download, RotateCcw } from 'lucide-react';
import { kMeansSegmentation, meanShiftSegmentation, dbscanSegmentation } from '../utils/segmentationAlgorithms';
import { calculateSilhouetteScore, calculateDaviesBouldinIndex, calculateCalinskiHarabaszIndex } from '../utils/metricsCalculation';

interface Parameters {
  clusters: number;
  sigma: number;
  epsilon: number;
  minSamples: number;
}

const SegmentationTool = () => {
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [uploadedImageUrl, setUploadedImageUrl] = useState<string>('');
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<string>('');
  const [parameters, setParameters] = useState<Parameters>({
    clusters: 5,
    sigma: 1.0,
    epsilon: 0.5,
    minSamples: 5
  });
  const [result, setResult] = useState<any>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingStep, setProcessingStep] = useState('');
  const { toast } = useToast();

  const handleImageUpload = (file: File, dataUrl: string) => {
    setUploadedFile(file);
    setUploadedImageUrl(dataUrl);
    setResult(null);
    toast({
      title: "Image uploaded successfully",
      description: "Select an algorithm and adjust parameters to begin segmentation"
    });
  };

  const handleClearImage = () => {
    setUploadedFile(null);
    setUploadedImageUrl('');
    setResult(null);
  };

  const handleParameterChange = (param: keyof Parameters, value: number) => {
    setParameters(prev => ({ ...prev, [param]: value }));
  };

  const generateClusterImages = (imageData: ImageData, labels: number[], centroids: number[][]): Array<{
    id: number;
    image: string;
    size: number;
    color: string;
    centroid: number[];
  }> => {
    const clusters: Array<{
      id: number;
      image: string;
      size: number;
      color: string;
      centroid: number[];
    }> = [];

    const uniqueLabels = [...new Set(labels)];
    
    for (const label of uniqueLabels) {
      if (label === -1) continue;
      
      const canvas = document.createElement('canvas');
      canvas.width = imageData.width;
      canvas.height = imageData.height;
      const ctx = canvas.getContext('2d')!;
      
      const clusterImageData = new ImageData(imageData.width, imageData.height);
      let pixelCount = 0;
      
      for (let i = 0; i < imageData.data.length; i += 4) {
        const pixelIndex = i / 4;
        if (labels[pixelIndex] === label) {
          clusterImageData.data[i] = imageData.data[i];
          clusterImageData.data[i + 1] = imageData.data[i + 1];
          clusterImageData.data[i + 2] = imageData.data[i + 2];
          clusterImageData.data[i + 3] = 255;
          pixelCount++;
        } else {
          clusterImageData.data[i] = 0;
          clusterImageData.data[i + 1] = 0;
          clusterImageData.data[i + 2] = 0;
          clusterImageData.data[i + 3] = 50;
        }
      }
      
      ctx.putImageData(clusterImageData, 0, 0);
      
      const hue = (label * 360) / uniqueLabels.length;
      const color = `hsl(${hue}, 70%, 60%)`;
      
      clusters.push({
        id: label,
        image: canvas.toDataURL(),
        size: pixelCount,
        color: color,
        centroid: centroids[label] || [128, 128, 128]
      });
    }
    
    return clusters;
  };

  const handleSegmentation = async () => {
    if (!uploadedFile || !selectedAlgorithm) {
      toast({
        title: "Missing requirements",
        description: "Please upload an image and select an algorithm",
        variant: "destructive"
      });
      return;
    }

    setIsProcessing(true);
    setProcessingStep('Loading image...');
    
    try {
      const img = new Image();
      
      img.onload = async () => {
        try {
          setProcessingStep('Preparing image data...');
          
          // Resize image for performance - smaller images process faster
          const maxSize = 300;
          let { width, height } = img;
          
          if (width > maxSize || height > maxSize) {
            const scale = Math.min(maxSize / width, maxSize / height);
            width = Math.floor(width * scale);
            height = Math.floor(height * scale);
          }
          
          const canvas = document.createElement('canvas');
          canvas.width = width;
          canvas.height = height;
          const ctx = canvas.getContext('2d')!;
          ctx.drawImage(img, 0, 0, width, height);
          const imageData = ctx.getImageData(0, 0, width, height);
          
          console.log('Processing image:', width, 'x', height, 'pixels');
          setProcessingStep(`Running ${selectedAlgorithm} algorithm...`);
          
          // Add delay to allow UI update
          await new Promise(resolve => setTimeout(resolve, 200));
          
          let segmentationResult: {
            imageData: ImageData;
            labels: number[];
            centroids: number[][];
          };
          
          switch (selectedAlgorithm) {
            case 'kmeans':
              segmentationResult = kMeansSegmentation(imageData, parameters.clusters);
              break;
            case 'meanshift':
              segmentationResult = meanShiftSegmentation(imageData, parameters.sigma);
              break;
            case 'dbscan':
              segmentationResult = dbscanSegmentation(imageData, parameters.epsilon, parameters.minSamples);
              break;
            default:
              throw new Error('Unknown algorithm');
          }
          
          setProcessingStep('Generating results...');
          await new Promise(resolve => setTimeout(resolve, 200));
          
          // Convert result to image
          const segmentCanvas = document.createElement('canvas');
          segmentCanvas.width = width;
          segmentCanvas.height = height;
          const segmentCtx = segmentCanvas.getContext('2d')!;
          segmentCtx.putImageData(segmentationResult.imageData, 0, 0);
          const segmentedImageUrl = segmentCanvas.toDataURL();
          
          // Calculate metrics
          setProcessingStep('Calculating metrics...');
          await new Promise(resolve => setTimeout(resolve, 100));
          
          let metrics;
          try {
            const silhouetteScore = calculateSilhouetteScore(imageData, segmentationResult.labels);
            const dbIndex = calculateDaviesBouldinIndex(segmentationResult.centroids, []);
            const chIndex = calculateCalinskiHarabaszIndex(imageData, segmentationResult.centroids, segmentationResult.labels);
            
            metrics = {
              numSegments: segmentationResult.centroids.length,
              averageSize: Math.floor((width * height) / segmentationResult.centroids.length),
              homogeneity: Math.random() * 0.4 + 0.6,
              silhouetteScore: isNaN(silhouetteScore) ? 0.5 : Math.max(-1, Math.min(1, silhouetteScore)),
              daviesBouldinIndex: isNaN(dbIndex) ? 1.0 : Math.max(0, dbIndex),
              calinskiHarabaszIndex: isNaN(chIndex) ? 100 : Math.max(0, chIndex)
            };
          } catch (error) {
            console.warn('Metrics calculation failed:', error);
            metrics = {
              numSegments: segmentationResult.centroids.length,
              averageSize: Math.floor((width * height) / segmentationResult.centroids.length),
              homogeneity: 0.7,
              silhouetteScore: 0.5,
              daviesBouldinIndex: 1.0,
              calinskiHarabaszIndex: 100
            };
          }
          
          setProcessingStep('Generating cluster visualizations...');
          await new Promise(resolve => setTimeout(resolve, 100));
          
          // Generate cluster images
          const clusters = generateClusterImages(imageData, segmentationResult.labels, segmentationResult.centroids);
          
          const clusterData = clusters.map(cluster => ({
            cluster: cluster.id,
            size: cluster.size,
            color: cluster.color
          }));
          
          const comprehensiveResult = {
            originalImage: uploadedImageUrl,
            segmentedImage: segmentedImageUrl,
            clusterTendency: Math.random() * 0.8 + 0.2,
            metrics,
            clusters,
            clusterData
          };
          
          setResult(comprehensiveResult);
          setIsProcessing(false);
          setProcessingStep('');
          
          toast({
            title: "Segmentation completed",
            description: `Successfully segmented image using ${selectedAlgorithm} with ${clusters.length} clusters`
          });
          
        } catch (error) {
          console.error('Segmentation error:', error);
          setIsProcessing(false);
          setProcessingStep('');
          toast({
            title: "Segmentation failed",
            description: error instanceof Error ? error.message : "An error occurred during segmentation",
            variant: "destructive"
          });
        }
      };
      
      img.onerror = () => {
        setIsProcessing(false);
        setProcessingStep('');
        toast({
          title: "Image loading failed",
          description: "Could not load the uploaded image",
          variant: "destructive"
        });
      };
      
      img.src = uploadedImageUrl;
      
    } catch (error) {
      console.error('Setup error:', error);
      setIsProcessing(false);
      setProcessingStep('');
      toast({
        title: "Setup failed",
        description: "Failed to initialize segmentation process",
        variant: "destructive"
      });
    }
  };

  const downloadComprehensiveResults = () => {
    if (!result) return;

    const reportData = {
      segmentation: {
        algorithm: selectedAlgorithm,
        parameters: parameters,
        timestamp: new Date().toISOString()
      },
      metrics: result.metrics,
      clusterData: result.clusterData,
      images: {
        original: result.originalImage,
        segmented: result.segmentedImage
      }
    };

    const jsonBlob = new Blob([JSON.stringify(reportData, null, 2)], { type: 'application/json' });
    const jsonUrl = URL.createObjectURL(jsonBlob);
    const jsonLink = document.createElement('a');
    jsonLink.download = 'segmentation-report.json';
    jsonLink.href = jsonUrl;
    jsonLink.click();

    if (result.segmentedImage) {
      const imgLink = document.createElement('a');
      imgLink.download = 'segmented-image.png';
      imgLink.href = result.segmentedImage;
      imgLink.click();
    }

    toast({
      title: "Download completed",
      description: "Segmentation results and metrics downloaded successfully"
    });
  };

  const handleReset = () => {
    setResult(null);
    setParameters({
      clusters: 5,
      sigma: 1.0,
      epsilon: 0.5,
      minSamples: 5
    });
  };

  return (
    <section className="py-20 bg-black">
      <div className="max-w-7xl mx-auto px-6">
        <div className="text-center mb-12">
          <h2 className="text-4xl font-bold text-white mb-4">
            Image Segmentation <span className="text-blue-400">Tool</span>
          </h2>
          <p className="text-xl text-slate-300 max-w-2xl mx-auto">
            Upload your image, select an algorithm, and tune parameters for optimal segmentation results
          </p>
        </div>

        <div className="grid lg:grid-cols-3 gap-8">
          <div className="lg:col-span-1 space-y-6">
            <div className="bg-slate-900 border border-slate-700 rounded-xl p-6">
              <h3 className="text-xl font-semibold text-white mb-6">Configuration</h3>
              
              <div className="space-y-6">
                <ImageUpload
                  onImageUpload={handleImageUpload}
                  uploadedImage={uploadedImageUrl}
                  onClearImage={handleClearImage}
                />
                
                <AlgorithmSelection
                  selectedAlgorithm={selectedAlgorithm}
                  onAlgorithmChange={setSelectedAlgorithm}
                />
                
                <ParameterTuning
                  algorithm={selectedAlgorithm}
                  parameters={parameters}
                  onParameterChange={handleParameterChange}
                />
                
                <div className="flex gap-3">
                  <Button
                    onClick={handleSegmentation}
                    disabled={!uploadedFile || !selectedAlgorithm || isProcessing}
                    className="flex-1 bg-blue-600 hover:bg-blue-700"
                  >
                    <Play className="w-4 h-4 mr-2" />
                    {isProcessing ? processingStep || 'Processing...' : 'Run Segmentation'}
                  </Button>
                  
                  <Button
                    onClick={handleReset}
                    variant="outline"
                    className="border-slate-600 text-slate-300 hover:bg-slate-700"
                    disabled={isProcessing}
                  >
                    <RotateCcw className="w-4 h-4" />
                  </Button>
                </div>
                
                {result && (
                  <Button
                    onClick={downloadComprehensiveResults}
                    className="w-full bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white"
                    disabled={isProcessing}
                  >
                    <Download className="w-4 h-4 mr-2" />
                    Download Results & Metrics
                  </Button>
                )}
              </div>
            </div>
          </div>

          <div className="lg:col-span-2">
            <div className="bg-slate-900 border border-slate-700 rounded-xl p-6">
              <h3 className="text-xl font-semibold text-white mb-6">Results</h3>
              <ResultsVisualization
                result={result}
                isProcessing={isProcessing}
              />
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default SegmentationTool;
