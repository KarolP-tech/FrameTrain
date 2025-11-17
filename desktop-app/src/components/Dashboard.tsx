import { useState } from 'react';
import Sidebar from './Sidebar';
import ModelManager from './ModelManager';
import TrainingPanel from './TrainingPanel';
import DatasetUpload from './DatasetUpload';
import AnalysisPanel from './AnalysisPanel';
import VersionManager from './VersionManager';
import Settings from './Settings';

interface UserData {
  apiKey: string;
  password: string;
  userId: string;
  email: string;
}

interface DashboardProps {
  userData: UserData;
  onLogout: () => void;
}

type View = 'models' | 'training' | 'dataset' | 'analysis' | 'versions' | 'settings';

export default function Dashboard({ userData, onLogout }: DashboardProps) {
  const [currentView, setCurrentView] = useState<View>('models');
  const [selectedModel, setSelectedModel] = useState<any>(null);

  const renderView = () => {
    switch (currentView) {
      case 'models':
        return <ModelManager onSelectModel={setSelectedModel} />;
      case 'training':
        return <TrainingPanel selectedModel={selectedModel} />;
      case 'dataset':
        return <DatasetUpload />;
      case 'analysis':
        return <AnalysisPanel />;
      case 'versions':
        return <VersionManager selectedModel={selectedModel} />;
      case 'settings':
        return <Settings userData={userData} onLogout={onLogout} />;
      default:
        return <ModelManager onSelectModel={setSelectedModel} />;
    }
  };

  return (
    <div className="flex h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <Sidebar 
        currentView={currentView} 
        onViewChange={setCurrentView}
        userEmail={userData.email}
        onLogout={onLogout}
      />
      
      <main className="flex-1 overflow-auto p-8">
        <div className="max-w-7xl mx-auto">
          {renderView()}
        </div>
      </main>
    </div>
  );
}
