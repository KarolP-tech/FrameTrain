
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { TrendingUp, Download } from 'lucide-react';

export default function AnalysisPanel() {
  // Mock-Daten für die Analyse
  const mockData = [
    { epoch: 1, loss: 1.8, accuracy: 0.45, val_loss: 1.9, val_accuracy: 0.42 },
    { epoch: 2, loss: 1.5, accuracy: 0.58, val_loss: 1.6, val_accuracy: 0.55 },
    { epoch: 3, loss: 1.2, accuracy: 0.68, val_loss: 1.4, val_accuracy: 0.64 },
    { epoch: 4, loss: 0.9, accuracy: 0.76, val_loss: 1.1, val_accuracy: 0.72 },
    { epoch: 5, loss: 0.7, accuracy: 0.82, val_loss: 0.9, val_accuracy: 0.78 },
    { epoch: 6, loss: 0.5, accuracy: 0.87, val_loss: 0.7, val_accuracy: 0.83 },
    { epoch: 7, loss: 0.4, accuracy: 0.90, val_loss: 0.6, val_accuracy: 0.86 },
    { epoch: 8, loss: 0.3, accuracy: 0.92, val_loss: 0.5, val_accuracy: 0.88 },
  ];

  const exportResults = () => {
    const csv = [
      ['Epoch', 'Loss', 'Accuracy', 'Val Loss', 'Val Accuracy'],
      ...mockData.map(d => [d.epoch, d.loss, d.accuracy, d.val_loss, d.val_accuracy])
    ].map(row => row.join(',')).join('\n');

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'training_results.csv';
    a.click();
  };

  return (
    <div>
      <div className="flex justify-between items-center mb-8">
        <h2 className="text-3xl font-bold text-white">Analyse & Visualisierung</h2>
        <button
          onClick={exportResults}
          className="flex items-center space-x-2 px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white font-semibold rounded-lg hover:from-purple-700 hover:to-pink-700 transition-all"
        >
          <Download className="w-5 h-5" />
          <span>Exportieren</span>
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm font-medium text-gray-300">Finaler Loss</h3>
            <TrendingUp className="w-5 h-5 text-green-400" />
          </div>
          <p className="text-3xl font-bold text-white">0.30</p>
          <p className="text-sm text-green-400 mt-1">-85% seit Start</p>
        </div>

        <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm font-medium text-gray-300">Finale Accuracy</h3>
            <TrendingUp className="w-5 h-5 text-green-400" />
          </div>
          <p className="text-3xl font-bold text-white">92%</p>
          <p className="text-sm text-green-400 mt-1">+104% seit Start</p>
        </div>

        <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm font-medium text-gray-300">Trainingszeit</h3>
            <TrendingUp className="w-5 h-5 text-blue-400" />
          </div>
          <p className="text-3xl font-bold text-white">2.5h</p>
          <p className="text-sm text-gray-400 mt-1">8 Epochs</p>
        </div>
      </div>

      <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20 mb-8">
        <h3 className="text-xl font-bold text-white mb-6">Loss Verlauf</h3>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={mockData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
              <XAxis dataKey="epoch" stroke="#ffffff80" />
              <YAxis stroke="#ffffff80" />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#1e293b', 
                  border: '1px solid #ffffff20',
                  borderRadius: '8px'
                }}
              />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="loss" 
                stroke="#ef4444" 
                strokeWidth={2}
                name="Training Loss"
              />
              <Line 
                type="monotone" 
                dataKey="val_loss" 
                stroke="#f97316" 
                strokeWidth={2}
                strokeDasharray="5 5"
                name="Validation Loss"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
        <h3 className="text-xl font-bold text-white mb-6">Accuracy Verlauf</h3>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={mockData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
              <XAxis dataKey="epoch" stroke="#ffffff80" />
              <YAxis stroke="#ffffff80" />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#1e293b', 
                  border: '1px solid #ffffff20',
                  borderRadius: '8px'
                }}
              />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="accuracy" 
                stroke="#22c55e" 
                strokeWidth={2}
                name="Training Accuracy"
              />
              <Line 
                type="monotone" 
                dataKey="val_accuracy" 
                stroke="#10b981" 
                strokeWidth={2}
                strokeDasharray="5 5"
                name="Validation Accuracy"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
