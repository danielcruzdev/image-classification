using Microsoft.ML;

namespace ImageClassification.ML
{
    public class ConsumeModel
    {
        private static readonly Lazy<PredictionEngine<ModelInput, ModelOutput>> PredictionEngine = new Lazy<PredictionEngine<ModelInput, ModelOutput>>(CreatePredictionEngine);

        public static ModelOutput Predict(ModelInput input)
        {
            ModelOutput result = PredictionEngine.Value.Predict(input);
            return result;
        }

        public static PredictionEngine<ModelInput, ModelOutput> CreatePredictionEngine()
        {
            try
            {
                var mlContext = new MLContext(seed: 0);

                ITransformer mlModel = mlContext.Model.Load("C:/Models/ML/ClassificationModel.zip", out var modelInputSchema);
                var predEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(mlModel);
                return predEngine;
            }
            catch (Exception ex)
            {
                throw new Exception(ex.Message);
            }

        }
    }
}
