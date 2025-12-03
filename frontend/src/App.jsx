import { useState } from 'react';
import MapView from './components/MapView';
import Sidebar from './components/Sidebar';

function App() {
  const [selectedData, setSelectedData] = useState(null);
  const [mapCenter, setMapCenter] = useState([12.86, 74.85]); 
  const [activeRiver, setActiveRiver] = useState("netravati"); 
  const [activeDate, setActiveDate] = useState("");
  
  // NEW: Track active parameter (Default: Salinity)
  const [activeParam, setActiveParam] = useState("salinity"); 

  const handleRiverChange = (riverId, coords) => {
      setActiveRiver(riverId); 
      setMapCenter(coords);    
      setSelectedData(null);
      // Optional: Reset param on river change?
      // setActiveParam("salinity");
  };

  return (
    <div className="flex flex-col md:flex-row h-screen w-screen bg-gray-50 overflow-hidden">
      
      {/* Sidebar Container */}
      <div className="order-2 md:order-1 w-full md:w-1/3 h-[45%] md:h-full p-2 md:p-4 z-[1000] shadow-xl bg-white overflow-y-auto relative rounded-t-2xl md:rounded-none">
        <Sidebar 
            data={selectedData} 
            onRiverSelect={handleRiverChange} 
            currentRiver={activeRiver}
            activeDate={activeDate}
            onDateChange={setActiveDate}
            
            // Pass Parameter Props
            activeParam={activeParam}
            onParamChange={(p) => { setActiveParam(p); setSelectedData(null); }} 
        />
      </div>

      {/* Map Container */}
      <div className="order-1 md:order-2 w-full md:w-2/3 h-[55%] md:h-full relative">
         <MapView 
            onMapClick={setSelectedData} 
            centerPosition={mapCenter} 
            selectedRiver={activeRiver}
            selectedDate={activeDate}
            
            // Pass Parameter Prop
            activeParam={activeParam}
         />
      </div>

    </div>
  );
}
export default App;