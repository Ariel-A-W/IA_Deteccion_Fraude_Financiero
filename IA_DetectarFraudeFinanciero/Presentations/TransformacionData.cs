using Microsoft.ML.Data;

namespace IA_DetectarFraudeFinanciero.Presentations;

public class TransformacionData
{
    [LoadColumn(0)]
    public float Monto { get; set; }

    [LoadColumn(1)]
    public float Frecuencia { get; set; }

    [LoadColumn(2)]
    public float TiempoTransaccion { get; set; }
}
