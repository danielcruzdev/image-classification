using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ImageClassification.TensorFlow
{
    public static class ConsumeTF
    {
        private static readonly Lazy<PredictionEngine<ImageData, ImagePrediction>> PredictionEngine = new Lazy<PredictionEngine<ImageData, ImagePrediction>>(CreatePredictionEngine);

        public static ImagePrediction Predict(string pathFile)
        {
            var prediction = PredictionEngine.Value.Predict(new ImageData
            {
                ImagePath = pathFile
            });

            return prediction;
        }

        public static PredictionEngine<ImageData, ImagePrediction> CreatePredictionEngine()
        {
            var context = new MLContext();
            var pathArquivoCsv = "C:/Models/TensorFlow/labels.csv";
            var pathImages = "C:/Models/TensorFlow/images";
            var pathModelPB = "C:/Models/TensorFlow/model/tensorflow_inception_graph.pb";

            var data = context.Data.LoadFromTextFile<ImageData>(pathArquivoCsv, separatorChar: ',');

            var pipeline = context.Transforms.Conversion
                .MapValueToKey("LabelKey", "Label")
                .Append(context.Transforms.LoadImages("input", pathImages, nameof(ImageData.ImagePath)))
                .Append(context.Transforms.ResizeImages("input", InceptionSettings.IMageWidth, InceptionSettings.ImageHeight, "input"))
                .Append(context.Transforms.ExtractPixels("input", interleavePixelColors: InceptionSettings.ChannelsList, offsetImage: InceptionSettings.Mean))
                .Append(context.Model.LoadTensorFlowModel(pathModelPB)
                .ScoreTensorFlowModel(new[] { "softmax2_pre_activation" }, new[] { "input" }, addBatchDimensionInput: true))
                .Append(context.MulticlassClassification.Trainers.LbfgsMaximumEntropy("LabelKey", "softmax2_pre_activation"))
                .Append(context.Transforms.Conversion.MapKeyToValue("PredictedLabelValue", "PredictedLabel"));

            var model = pipeline.Fit(data);
            var predict = context.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);
            return predict;
        }
    }
}
