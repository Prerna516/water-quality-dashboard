import { MapContainer, TileLayer, Marker, Popup, useMapEvents, ImageOverlay, Rectangle } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { useEffect, useState } from 'react';
import axios from 'axios';
import L from 'leaflet';

// Icon Fix
import icon from 'leaflet/dist/images/marker-icon.png';
import iconShadow from 'leaflet/dist/images/marker-shadow.png';
let DefaultIcon = L.icon({ iconUrl: icon, shadowUrl: iconShadow, iconAnchor: [12, 41], popupAnchor: [1, -34] });
L.Marker.prototype.options.icon = DefaultIcon;

function ClickHandler({ onDataFound, selectedRiver }) {
  useMapEvents({
    click: async (e) => {
      const { lat, lng } = e.latlng;
      try {
        // Fetch nearest point using the SELECTED RIVER
        const res = await axios.get(`http://127.0.0.1:8000/api/get-nearest?lat=${lat}&lng=${lng}&river=${selectedRiver}`);
        if (res.data && res.data.found) {
           onDataFound(res.data);
        } else {
           // Optional: Don't alert if user clicks on empty map, just ignore
           console.log("No data here");
        }
      } catch (error) { console.error("Backend Error:", error); }
    },
  });
  return null;
}

function MapUpdater({ center }) {
  const map = useMapEvents({});
  useEffect(() => {
    map.flyTo(center, 11); // Zoom 11 is perfect for river view
  }, [center, map]);
  return null;
}

export default function MapView({ onMapClick, centerPosition, selectedRiver }) {
    const [activeMarker, setActiveMarker] = useState(null);
    const [overlayBounds, setOverlayBounds] = useState(null);
    const [heatmapUrl, setHeatmapUrl] = useState(null);

    // Fetch bounds whenever the river changes
    useEffect(() => {
        if(!selectedRiver) return;

        axios.get(`http://127.0.0.1:8000/api/bounds?river=${selectedRiver}`)
            .then(res => {
                if(res.data.min_lat) {
                    const b = res.data;
                    setOverlayBounds([[b.min_lat, b.min_lng], [b.max_lat, b.max_lng]]);
                    setHeatmapUrl(`http://127.0.0.1:8000/api/get-heatmap?river=${selectedRiver}&t=${Date.now()}`);
                } else {
                    // If no bounds found (e.g. missing shapefile), clear map
                    setOverlayBounds(null);
                    setHeatmapUrl(null);
                }
            })
            .catch(err => console.error("Error loading map data:", err));
    }, [selectedRiver]);

    const handleDataFound = (data) => {
        onMapClick(data);
        setActiveMarker([data.latitude, data.longitude]);
    };

    // --- FIX FOR DARKNESS: USING LIGHT MAP ---
    const lightTiles = "https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png"; 

    return (
        <MapContainer center={centerPosition} zoom={11} style={{ height: '100%', width: '100%' }}>
            <TileLayer attribution='&copy; CartoDB' url={lightTiles} />
            
            <ClickHandler onDataFound={handleDataFound} selectedRiver={selectedRiver} />
            <MapUpdater center={centerPosition} />

            {overlayBounds && heatmapUrl && (
                <ImageOverlay url={heatmapUrl} bounds={overlayBounds} opacity={0.8} zIndex={1000} />
            )}

            {activeMarker && (
                <Marker position={activeMarker}>
                    <Popup>Salinity: {activeMarker.salinity} PSU</Popup>
                </Marker>
            )}

            {/* Plasma Legend */}
            <div className="leaflet-bottom leaflet-right">
                <div className="leaflet-control leaflet-bar" style={{ 
                    backgroundColor: 'white', padding: '10px', marginBottom: '20px', marginRight: '20px',
                    borderRadius: '8px', boxShadow: '0 4px 6px rgba(0,0,0,0.3)', pointerEvents: 'auto'
                }}>
                    <div className="text-xs font-bold text-gray-600 mb-1">Salinity (PSU)</div>
                    <div className="flex items-center gap-2">
                        <span className="text-xs font-mono font-medium">Low</span>
                        <div style={{ 
                            width: '120px', height: '12px', 
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