namespace ImageClassification.Api.Models
{
    public class PredictResult
    {
        public PredictResult(string fileName, string predict, string tecnology, float score)
        {
            FileName = fileName;
            Predict = predict;
            Tecnology = tecnology;
            Score = score;
        }

        public string FileName { get; private set; }
        public string Predict { get; private set; }
        public string Tecnology { get; private set; }
        public float Score { get; private set; }
    }
}
