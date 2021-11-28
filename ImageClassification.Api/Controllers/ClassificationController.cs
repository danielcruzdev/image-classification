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
        public async Task<ActionResult<string>> GetClassificationPhoto([FromForm] IFormFile file)
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

            if (scoreML < 0.8)
            {
                var resultTF = ConsumeTF.Predict(pathFile);
                predictionTF = resultTF.PredictedLabelValue;
                scoreTF = resultTF.Score.Max();
            }

            var predictResult = new
            {
                Predict = scoreML > scoreTF ? predictionML : predictionTF,
                Score = scoreML > scoreTF ? scoreML : scoreTF,
                Type = scoreML > scoreTF ? "ML .NET" : "Tensor Flow"
            };

            System.IO.File.Delete(pathFile);
            return Ok(predictResult);
        }

        [HttpPost("ML")]
        public async Task<ActionResult<string>> GetClassificationPhotoML([FromForm] IFormFile file) 
        {
            var pathFile = Path.Combine("C:", "ArquivosRecebidos", $"{file.FileName}");
            using var fileStream = new FileStream(pathFile, FileMode.Create);
            await file.CopyToAsync(fileStream);
            fileStream.Close();

            var input = new ModelInput()
            {
                ImageSource = pathFile,
            };

            var result = ConsumeModel.Predict(input);
            var score = result.Score.Max();

            var predictResult = new
            {
                Predict = result.Prediction,
                Score = score,
                Type = "ML .NET"
            };

            System.IO.File.Delete(pathFile);
            return Ok(predictResult);
        }

        [HttpPost("TF")]
        public async Task<ActionResult<string>> GetClassificationPhotoTF([FromForm] IFormFile file)
        {
            var pathFile = Path.Combine("C:", "ArquivosRecebidos", $"{file.FileName}");
            using var fileStream = new FileStream(pathFile, FileMode.Create);
            await file.CopyToAsync(fileStream);
            fileStream.Close();

            var result = ConsumeTF.Predict(pathFile);
            var score = result.Score.Max();

            var predictResult = new
            {
                Predict = ReturnPredictTensorFlow(result.PredictedLabelValue),
                Score = score,
                Type = "Tensor Flow"
            };

            System.IO.File.Delete(pathFile);
            return Ok(predictResult);
        }

        private static string ReturnPredictTensorFlow(string labelValue)
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
