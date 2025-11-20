import { useState } from 'react';
import MapView from './components/MapView';
import Sidebar from './components/Sidebar';

function App() {
  const [selectedData, setSelectedData] = useState(null);
  // DEFAULT: Start at Netravati Coordinates
  const [mapCenter, setMapCenter] = useState([12.86, 74.85]); 
  // DEFAULT: Start with 'netravati' active
  const [activeRiver, setActiveRiver] = useState("netravati"); 

  const handleRiverChange = (riverId, coords) => {
      setActiveRiver(riverId); // Update logic
      setMapCenter(coords);    // Update visuals (flyTo)
      setSelectedData(null);   // Clear old clicks
  };

  return (
    <div className="flex h-screen w-screen bg-gray-100 overflow-hidden">
      <div className="w-1/3 h-full p-4 z-[1000] shadow-2xl bg-white overflow-y-auto relative">
        <Sidebar 
            data={selectedData} 
            onRiverSelect={handleRiverChange} 
            currentRiver={activeRiver} 
        />
      </div>
      <div className="w-2/3 h-full relative">
         <MapView 
            onMapClick={setSelectedData} 
            centerPosition={mapCenter} 
            selectedRiver={activeRiver} 
         />
      </div>
    </div>
  );
}
export default App;