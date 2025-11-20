import { MapContainer, TileLayer, Marker, Popup, useMapEvents, ImageOverlay, Rectangle } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { useEffect, useState } from 'react';
import axios from 'axios';
import L from 'leaflet';

// Fix for default Leaflet marker icons
import icon from 'leaflet/dist/images/marker-icon.png';
import iconShadow from 'leaflet/dist/images/marker-shadow.png';
let DefaultIcon = L.icon({
    iconUrl: icon,
    shadowUrl: iconShadow,
    iconAnchor: [12, 41],
    popupAnchor: [1, -34]
});
L.Marker.prototype.options.icon = DefaultIcon;

// 1. Click Handler Component
function ClickHandler({ onDataFound }) {
  useMapEvents({
    click: async (e) => {
      const { lat, lng } = e.latlng;
      console.log(`Clicked at: ${lat}, ${lng}`);
      
      try {
        // Fetch data for the specific point clicked
        const res = await axios.get(`http://127.0.0.1:8000/api/get-nearest?lat=${lat}&lng=${lng}`);
        
        if (res.data && (res.data.found || res.data.ai_salinity_prediction !== undefined)) {
            // If we have a valid prediction, show it!
           onDataFound(res.data);
        } else {
           alert("No data available for this location.");
        }
      } catch (error) {
        console.error("Backend Error:", error);
      }
    },
  });
  return null;
}

// 2. Map Updater (Centers map when you select a river from sidebar)
function MapUpdater({ center }) {
  const map = useMapEvents({});
  useEffect(() => {
    map.flyTo(center, 12); // Zoom level 12 is good for rivers
  }, [center, map]);
  return null;
}

// 3. MAIN COMPONENT
export default function MapView({ onMapClick, centerPosition }) {
    const [activeMarker, setActiveMarker] = useState(null);
    const [overlayBounds, setOverlayBounds] = useState(null);
    const [heatmapUrl, setHeatmapUrl] = useState(null);

    // Fetch the River Boundaries & Heatmap when component loads
    useEffect(() => {
        // A. Get the bounds (Where to put the image)
        axios.get('http://127.0.0.1:8000/api/bounds')
            .then(res => {
                const b = res.data;
                // Leaflet expects: [[minLat, minLng], [maxLat, maxLng]]
                // Backend sends: min_lat, min_lng, max_lat, max_lng
                const bounds = [[b.min_lat, b.min_lng], [b.max_lat, b.max_lng]];
                setOverlayBounds(bounds);
                
                // B. Set the URL for the heatmap image
                // We add a timestamp to prevent the browser from caching an old image
                setHeatmapUrl(`http://127.0.0.1:8000/api/get-heatmap?t=${Date.now()}`);
            })
            .catch(err => console.error("Error loading map data:", err));
    }, []);

    const handleDataFound = (data) => {
        onMapClick(data);
        setActiveMarker([data.latitude, data.longitude]);
    };

    const darkTiles = "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png";

    return (
        <MapContainer center={centerPosition} zoom={11} style={{ height: '100%', width: '100%' }}>
            <TileLayer attribution='&copy; CartoDB' url={darkTiles} />
            
            <ClickHandler onDataFound={handleDataFound} />
            <MapUpdater center={centerPosition} />

            {/* --- NEW: THE HEATMAP OVERLAY --- */}
            {overlayBounds && heatmapUrl && (
                <ImageOverlay
                    url={heatmapUrl}
                    bounds={overlayBounds}
                    opacity={0.7} // 0.7 = Slightly transparent so you can see the map below
                    zIndex={1000}
                />
            )}

            {/* --- NEW: OPTIONAL BLUE BOX AROUND STUDY AREA --- */}
            {overlayBounds && (
                 <Rectangle
                    bounds={overlayBounds}
                    pathOptions={{ color: '#3b82f6', weight: 1, fillOpacity: 0, dashArray: '5, 5' }}
                />
            )}

            {activeMarker && (
                <Marker position={activeMarker}>
                    <Popup>Salinity Prediction Point</Popup>
                </Marker>
            )}
            {/* ======================================================== */}
        {/* LEGEND: PASTE THIS HERE (Inside MapContainer)          */}
        {/* ======================================================== */}
        <div className="leaflet-bottom leaflet-right">
            <div className="leaflet-control leaflet-bar" style={{ 
                backgroundColor: 'white', 
                padding: '10px', 
                marginBottom: '20px', 
                marginRight: '20px',
                borderRadius: '8px',
                boxShadow: '0 4px 6px rgba(0,0,0,0.3)',
                pointerEvents: 'auto' // Important: Allows clicking/hovering if needed
            }}>
                <div className="text-xs font-bold text-gray-600 mb-1">Salinity (PSU)</div>
                <div className="flex items-center gap-2">
                    <span className="text-xs font-mono font-medium">Low</span>
                    {/* Gradient Bar: Matches 'viridis' colormap */}
                    <div style={{ 
    width: '120px', 
    height: '12px', 
    // CHANGE THIS LINE TO MATCH PLASMA:
    background: 'linear-gradient(to right, #0d0887, #6a00a8, #b12a90, #e16462, #fca636, #f0f921)',
    borderRadius: '2px'
}}></div>
                    <span className="text-xs font-mono font-medium">High</span>
                </div>
            </div>
        </div>

      </MapContainer>
    );
}