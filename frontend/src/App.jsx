import { useState } from 'react';
import MapView from './components/MapView';
import Sidebar from './components/Sidebar';

function App() {
  const [selectedData, setSelectedData] = useState(null);
  const [mapCenter, setMapCenter] = useState([14.995, 72.47]); // Default center

  return (
    <div className="flex h-screen w-screen bg-gray-100 overflow-hidden">
      {/* LEFT SIDEBAR */}
      <div className="w-1/3 h-full p-4 z-[1000] shadow-2xl bg-white overflow-y-auto relative">
        <Sidebar data={selectedData} onRiverSelect={setMapCenter} />
      </div>
      {/* RIGHT MAP */}
      <div className="w-2/3 h-full relative">
         <MapView onMapClick={setSelectedData} centerPosition={mapCenter} />
      </div>
    </div>
  );
}

export default App;