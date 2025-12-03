import { MapContainer, TileLayer, Marker, Popup, useMapEvents, ImageOverlay } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { useEffect, useState } from 'react';
import axios from 'axios';
import L from 'leaflet';

import icon from 'leaflet/dist/images/marker-icon.png';
import iconShadow from 'leaflet/dist/images/marker-shadow.png';
let DefaultIcon = L.icon({ iconUrl: icon, shadowUrl: iconShadow, iconAnchor: [12, 41], popupAnchor: [1, -34] });
L.Marker.prototype.options.icon = DefaultIcon;

const BACKEND_URL = "http://127.0.0.1:8000"; 

function ClickHandler({ onDataFound, selectedRiver, selectedDate, activeParam }) {
  useMapEvents({
    click: async (e) => {
      const { lat, lng } = e.latlng;
      try {
        // SEND PARAM TO BACKEND
        const url = `${BACKEND_URL}/api/get-nearest?lat=${lat}&lng=${lng}&river=${selectedRiver}&date=${selectedDate}&param=${activeParam}`;
        const res = await axios.get(url);
        if (res.data && res.data.found) onDataFound(res.data);
      } catch (error) { console.error("Backend Error:", error); }
    },
  });
  return null;
}

function MapUpdater({ center }) {
  const map = useMapEvents({});
  useEffect(() => { map.flyTo(center, 11); }, [center, map]);
  return null;
}

export default function MapView({ onMapClick, centerPosition, selectedRiver, selectedDate, activeParam }) {
    const [activeMarker, setActiveMarker] = useState(null);
    const [overlayBounds, setOverlayBounds] = useState(null);
    const [heatmapUrl, setHeatmapUrl] = useState(null);

    useEffect(() => {
        if(!selectedRiver) return;
        
        axios.get(`${BACKEND_URL}/api/bounds?river=${selectedRiver}`)
            .then(res => {
                if(res.data.min_lat) {
                    const b = res.data;
                    setOverlayBounds([[b.min_lat, b.min_lng], [b.max_lat, b.max_lng]]);
                    
                    // SEND PARAM TO BACKEND FOR HEATMAP
                    const url = `${BACKEND_URL}/api/get-heatmap?river=${selectedRiver}&date=${selectedDate}&param=${activeParam}&t=${Date.now()}`;
                    setHeatmapUrl(url);
                } else {
                    setOverlayBounds(null); setHeatmapUrl(null);
                }
            })
            .catch(err => console.error(err));
    }, [selectedRiver, selectedDate, activeParam]);

    const handleDataFound = (data) => {
        onMapClick(data);
        setActiveMarker([data.latitude, data.longitude]);
    };

    const lightTiles = "https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png"; 

    return (
        <div className="relative h-full w-full">
            <MapContainer center={centerPosition} zoom={11} style={{ height: '100%', width: '100%' }} zoomControl={false}>
                <TileLayer attribution='&copy; CartoDB' url={lightTiles} />
                
                <ClickHandler 
                    onDataFound={handleDataFound} 
                    selectedRiver={selectedRiver} 
                    selectedDate={selectedDate}
                    activeParam={activeParam} 
                />
                <MapUpdater center={centerPosition} />

                {overlayBounds && heatmapUrl && (
                    <ImageOverlay url={heatmapUrl} bounds={overlayBounds} opacity={0.8} zIndex={1000} />
                )}

                {activeMarker && (
                    <Marker position={activeMarker}>
                        <Popup>Prediction Location</Popup>
                    </Marker>
                )}
            </MapContainer>

            {/* LEGEND */}
            <div className="absolute bottom-6 right-2 z-[500] bg-white/90 p-2 rounded-lg shadow-md scale-75 md:scale-100 origin-bottom-right backdrop-blur-sm border border-gray-200">
                <div className="text-[10px] md:text-xs font-bold text-gray-600 mb-1 uppercase tracking-wider">
                    {activeParam} Level
                </div>
                <div className="flex items-center gap-1">
                    <span className="text-[10px] font-mono text-gray-500">Low</span>
                    <div style={{ 
                        width: '100px', height: '8px', 
                        // Dynamic Gradient based on activeParam
                        background: activeParam === 'salinity' 
                            ? 'linear-gradient(to right, #0d0887, #f0f921)' // Plasma
                            : (activeParam === 'chlorophyll' 
                                ? 'linear-gradient(to right, #440154, #5ec962)' // Viridis
                                : 'linear-gradient(to right, #00224e, #ffe945)') // Cividis
                        ,
                        borderRadius: '2px'
                    }}></div>
                    <span className="text-[10px] font-mono text-gray-500">High</span>
                </div>
            </div>
        </div>
    );
}