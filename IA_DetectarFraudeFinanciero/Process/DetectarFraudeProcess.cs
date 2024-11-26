using IA_DetectarFraudeFinanciero.Presentations;
using Microsoft.ML;
using Newtonsoft.Json;
using System.Data;

namespace IA_DetectarFraudeFinanciero.Process;

public class DetectarFraudeProcess
{
    public void GetProcess()
    {
        var mlContext = new MLContext();

        string relativePath = Path.Combine("Data", "transacciones.json");
        string filePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, relativePath);

        if (!File.Exists(filePath))
        {
            Console.WriteLine("Archivo JSON no encontrado.");
        }
        else
        {
            string jsonContent = File.ReadAllText(filePath);

            var objectContent = JsonConvert.DeserializeObject<List<TransformacionData>>(jsonContent)!;

            var lstDFF = new List<TransformacionData>();
            foreach (var item in objectContent)
            {
                lstDFF.Add(item);
            }

            var dataView = mlContext.Data.LoadFromEnumerable<TransformacionData>(lstDFF);

            // Configurar pipeline con KMeans
            var pipeline = mlContext.Transforms
                .Concatenate(
                    "Features",
                    nameof(TransformacionData.Monto),
                    nameof(TransformacionData.Frecuencia),
                    nameof(TransformacionData.TiempoTransaccion))
                .Append(mlContext.Transforms.NormalizeMinMax("Features"))
                .Append(mlContext.Clustering.Trainers.KMeans("Features", numberOfClusters: 2));

            var model = pipeline.Fit(dataView);

            // Predicciones para datos de entrenamiento
            var transformedData = model.Transform(dataView);
            var predictions = mlContext
                .Data
                .CreateEnumerable<DetectaAnomalia>(
                    transformedData, 
                    reuseRowObject: false
                ).ToList();

            // Calcular promedio y desviación estándar del clúster 0
            var cluster0Scores = predictions.Select(p => p.Score![0]).ToArray();
            var averageScore = cluster0Scores.Average();
            var stdDeviation = Math.Sqrt(cluster0Scores.Average(v => Math.Pow(v - averageScore, 2)));

            // Establecer un límite dinámico (ej. 2 desviaciones estándar)
            var threshold = averageScore + 2 * stdDeviation;
            
            // Transacción nueva
            var newTransaction = new TransformacionData 
            { 
                Monto = 2000, 
                Frecuencia = 1, 
                TiempoTransaccion = 15 
            };
            
            var newDataView = mlContext.Data.LoadFromEnumerable(new[] { newTransaction });

            var prediction = model.Transform(newDataView);
            var score = mlContext.Data.CreateEnumerable<DetectaAnomalia>(prediction, reuseRowObject: false).First();

            // Evaluar si es atípico
            var transactionScore = score.Score![0];
            Console.WriteLine("Evaluación de Fraude Financiero");
            Console.WriteLine("*******************************\n");
            Console.WriteLine("Datos de la Transacción Resiente");
            Console.WriteLine("********************************");
            Console.WriteLine($"Monto    $: {newTransaction.Monto}");
            Console.WriteLine($"Frecuencia: {newTransaction.Frecuencia}");
            Console.WriteLine($"Tiempo    : {newTransaction.TiempoTransaccion}\n");
            Console.WriteLine("Resultado del Análisis");
            Console.WriteLine("**********************");
            Console.WriteLine($"Puntaje   : {transactionScore:F2}");
            Console.WriteLine($"Desviación: {threshold:F2}");
            Console.WriteLine("");
            Console.WriteLine(
                transactionScore > threshold ? 
                    "Se ha detectado una anomalía en la transacción." :
                    "La transacción es normal."
                );
        }
    }
}
