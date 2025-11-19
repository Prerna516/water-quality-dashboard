import { MapContainer, TileLayer, Marker, Popup, useMapEvents, Rectangle } from 'react-leaflet'; // Added Rectangle
import 'leaflet/dist/leaflet.css';
import { useEffect, useState } from 'react';
import axios from 'axios';
import L from 'leaflet';

// Fix for default Leaflet marker icons in React
import icon from 'leaflet/dist/images/marker-icon.png';
import iconShadow from 'leaflet/dist/images/marker-shadow.png';
let DefaultIcon = L.icon({
    iconUrl: icon,
    shadowUrl: iconShadow,
    iconAnchor: [12, 41],
    popupAnchor: [1, -34]
});
L.Marker.prototype.options.icon = DefaultIcon;

function ClickHandler({ onDataFound }) {
  useMapEvents({
    // Inside ClickHandler
click: async (e) => {
  const { lat, lng } = e.latlng;
  console.log(`Clicked at: ${lat}, ${lng}`);
  
  try {
    const res = await axios.get(`http://127.0.0.1:8000/api/get-nearest?lat=${lat}&lng=${lng}`);
    
    // LOG THE RAW RESPONSE HERE
    console.log("Full Backend Response:", res); 
    
    if (res.data && res.data.found) { // Check if res.data exists first
       console.log("Data found, sending to parent...");
       onDataFound(res.data);
    } else {
       console.warn("Backend returned success, but 'found' was false/missing:", res.data);
       alert("No data point found near this click.");
    }
  } catch (error) {
    // LOG THE FULL ERROR OBJECT
    console.error("Detailed Axios Error:", error.response ? error.response : error);
    alert("Backend connection failed. Check Console for details.");
  }
},
  });
  return null;
}

function MapUpdater({ center }) {
  const map = useMapEvents({});
  useEffect(() => {
    map.flyTo(center, 13);
  }, [center, map]);
  return null;
}

export default function MapView({ onMapClick, centerPosition }) {
    const [activeMarker, setActiveMarker] = useState(null);
    const [bounds, setBounds] = useState(null); // NEW STATE

    // NEW EFFECT: Fetch bounds when component loads
    useEffect(() => {
        axios.get('http://127.0.0.1:8000/api/bounds')
            .then(res => {
                const b = res.data;
                // Format for Leaflet: [[lat1, lng1], [lat2, lng2]]
                setBounds([[b.min_lat, b.min_lng], [b.max_lat, b.max_lng]]);
            })
            .catch(err => console.error("Could not fetch bounds:", err));
    }, []);

    const handleDataFound = (data) => {
        onMapClick(data);
        setActiveMarker([data.latitude, data.longitude]);
    };

    const darkTiles = "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png";

    return (
        <MapContainer center={centerPosition} zoom={10} style={{ height: '100%', width: '100%' }}>
            <TileLayer attribution='&copy; CartoDB' url={darkTiles} />
            <ClickHandler onDataFound={handleDataFound} />
            <MapUpdater center={centerPosition} />

            {/* NEW: Draw the rectangle if bounds exist */}
            {bounds && (
                <Rectangle
                    bounds={bounds}
                    pathOptions={{ color: '#3b82f6', weight: 2, fillOpacity: 0.1, dashArray: '5, 10' }}
                />
            )}

            {activeMarker && (
                <Marker position={activeMarker}>
                    <Popup>Selected Point</Popup>
                </Marker>
            )}
        </MapContainer>
    );
}