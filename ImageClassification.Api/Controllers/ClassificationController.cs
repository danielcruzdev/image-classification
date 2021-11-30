using ImageClassification.Api.Models;
using ImageClassification.ML;
using ImageClassification.TensorFlow;
using Microsoft.AspNetCore.Mvc;

namespace ImageClassification.Api.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class ClassificationController : ControllerBase
    {
        [HttpPost]
        public async Task<ActionResult<string>> GetClassificationPhoto([FromForm] List<IFormFile> files)
        {
            var predicts = new List<PredictResult>();
            foreach (var file in files)
            {
                var pathFile = Path.Combine("C:", "ArquivosRecebidos", $"{file.FileName}");
                using var fileStream = new FileStream(pathFile, FileMode.Create);
                await file.CopyToAsync(fileStream);
                fileStream.Close();

                var input = new ModelInput()
                {
                    ImageSource = pathFile,
                };

                var resultML = ConsumeModel.Predict(input);
                var predictionML = resultML.Prediction;
                var scoreML = resultML.Score.Max();
                var predictionTF = "";
                var scoreTF = 0.0f;

                var resultTF = ConsumeTF.Predict(pathFile);
                predictionTF = resultTF.PredictedLabelValue;
                scoreTF = resultTF.Score.Max();

                var predict = scoreML > scoreTF ? FormatPredict(predictionML) : FormatPredict(predictionTF);
                var score = scoreML > scoreTF ? scoreML : scoreTF;
                var tecnology = scoreML > scoreTF ? "ML .NET" : "Tensor Flow";

                var predictResult = new PredictResult(file.FileName, predict, tecnology, score);

                predicts.Add(predictResult);

                System.IO.File.Delete(pathFile);
            }

            return Ok(predicts);
        }

        private static string FormatPredict(string labelValue)
        {
            labelValue = labelValue.ToLower();
            if (labelValue.Contains("bird"))
                return "Bird";
            if (labelValue.Contains("dog"))
                return "Dog";
            if (labelValue.Contains("fish"))
                return "Fish";
            if (labelValue.Contains("horse"))
                return "Horse";
            if (labelValue.Contains("cat"))
                return "Cat";

            return "";
        }
    }
}
