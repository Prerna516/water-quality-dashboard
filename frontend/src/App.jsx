import { useState } from 'react';
import MapView from './components/MapView';
import Sidebar from './components/Sidebar';

function App() {
  const [selectedData, setSelectedData] = useState(null);
  const [mapCenter, setMapCenter] = useState([12.86, 74.85]); 
  const [activeRiver, setActiveRiver] = useState("netravati"); 

  const handleRiverChange = (riverId, coords) => {
      setActiveRiver(riverId); 
      setMapCenter(coords);    
      setSelectedData(null);   
  };

  return (
    // MAIN CONTAINER: Column on mobile, Row on desktop
    <div className="flex flex-col md:flex-row h-screen w-screen bg-gray-50 overflow-hidden">
      
      {/* SIDEBAR CONTAINER */}
      {/* Mobile: Bottom (Order 2), Height 40% */}
      {/* Desktop: Left (Order 1), Height 100%, Width 33% */}
      <div className="order-2 md:order-1 w-full md:w-1/3 h-[40%] md:h-full p-2 md:p-4 z-[1000] shadow-[0_-4px_10px_rgba(0,0,0,0.1)] md:shadow-xl bg-white overflow-y-auto relative rounded-t-2xl md:rounded-none">
        <Sidebar 
            data={selectedData} 
            onRiverSelect={handleRiverChange} 
            currentRiver={activeRiver} 
        />
      </div>

      {/* MAP CONTAINER */}
      {/* Mobile: Top (Order 1), Height 60% */}
      {/* Desktop: Right (Order 2), Height 100%, Width 66% */}
      <div className="order-1 md:order-2 w-full md:w-2/3 h-[60%] md:h-full relative">
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