import React, { useEffect, useState } from 'react';
import axios from 'axios';

// ⚠️ REPLACE WITH YOUR DEPLOYED URL IF DEPLOYING
const BACKEND_URL = "http://127.0.0.1:8000"; 

export default function Sidebar({ data, onRiverSelect, currentRiver, activeDate, onDateChange }) {
    
    const [availableDates, setAvailableDates] = useState([]);

    const rivers = [
        { id: "netravati",  name: "Netravati River",  coords: [12.86, 74.85] },
        { id: "kali",       name: "Kali River",       coords: [14.84, 74.13] },
        { id: "sharavathi", name: "Sharavathi River", coords: [14.28, 74.45] },
        { id: "nandini",    name: "Nandini River",    coords: [13.05, 74.78] },
        { id: "gangavali",  name: "Gangavali River",  coords: [14.58, 74.32] },
    ];

    // Fetch Dates when river changes
    useEffect(() => {
        axios.get(`${BACKEND_URL}/api/dates?river=${currentRiver}`)
            .then(res => {
                setAvailableDates(res.data);
                // Auto-select latest date
                if(res.data.length > 0 && !activeDate) {
                    onDateChange(res.data[res.data.length - 1]);
                }
            })
            .catch(err => console.error("Date fetch error", err));
    }, [currentRiver]);

    // ... (Keep your getSalinityStatus function here) ...
    const getSalinityStatus = (val) => {
        if (val < 0.5) return { text: "Fresh Water", desc: "Safe range.", color: "text-green-200", bg: "bg-green-600", icon: "💧" };
        if (val < 30) return { text: "Brackish Water", desc: "Mix of fresh & sea.", color: "text-yellow-200", bg: "bg-yellow-600", icon: "⚠️" };
        return { text: "Sea Water", desc: "High salt content.", color: "text-red-200", bg: "bg-red-600", icon: "🌊" };
    };
    
    const value = data?.ai_salinity_prediction || 0;
    const status = getSalinityStatus(value);

    return (
        <div className="flex flex-col h-full gap-3 md:gap-4 text-gray-800 font-sans">
            <div className="border-b-2 border-blue-600 pb-2 md:pb-4 flex justify-between items-end">
                <div>
                    <h1 className="text-xl md:text-3xl font-extrabold text-blue-900 tracking-tight">AQUA MONITOR</h1>
                    <p className="text-[10px] md:text-sm text-blue-600 font-bold uppercase tracking-wider">AI Salinity Prediction</p>
                </div>
            </div>

            <div className="bg-blue-50 p-2 md:p-4 rounded-xl border border-blue-100 shadow-sm flex flex-col gap-3">
                 <div>
                     <label className="text-[10px] md:text-xs font-bold text-blue-800 uppercase mb-1 block">River Basin</label>
                     <select 
                        className="w-full p-2 text-sm bg-white border border-blue-200 rounded-lg outline-none"
                        value={currentRiver} 
                        onChange={(e) => {
                            const r = rivers.find(riv => riv.id === e.target.value);
                            if(r) onRiverSelect(e.target.value, r.coords);
                        }}
                     >
                        {rivers.map(r => <option key={r.id} value={r.id}>{r.name}</option>)}
                     </select>
                 </div>

                 {/* NEW DATE PICKER */}
                 <div>
                     <label className="text-[10px] md:text-xs font-bold text-blue-800 uppercase mb-1 block">Date</label>
                     <select 
                        className="w-full p-2 text-sm bg-white border border-blue-200 rounded-lg outline-none"
                        value={activeDate} 
                        onChange={(e) => onDateChange(e.target.value)}
                        disabled={availableDates.length === 0}
                     >
                        {availableDates.length === 0 && <option>No dates available</option>}
                        {availableDates.map(d => <option key={d} value={d}>{d}</option>)}
                     </select>
                 </div>
            </div>

            {/* Results Section (Same as before) */}
            <div className="flex-grow overflow-y-auto">
                {!data ? (
                    <div className="h-40 md:h-64 flex flex-col items-center justify-center text-center p-4 bg-gray-50 border-2 border-dashed border-gray-300 rounded-xl">
                        <p className="text-gray-400 font-medium text-xs md:text-sm">
                            Tap the map to analyze water quality.
                        </p>
                    </div>
                ) : (
                    <div className="space-y-4 animate-fade-in mt-1">
                        <div className="flex justify-between items-center bg-gray-100 p-2 rounded-lg text-[10px] md:text-xs font-mono text-gray-600 border border-gray-200">
                             <span>LAT: {data.latitude?.toFixed(4)}</span>
                             <span>LON: {data.longitude?.toFixed(4)}</span>
                        </div>

                        <div className={`relative overflow-hidden ${status.bg} text-white p-4 md:p-6 rounded-2xl shadow-lg transform transition-all`}>
                            <div className="flex justify-between items-start">
                                <span className="relative z-10 text-[10px] md:text-xs uppercase font-bold opacity-80 tracking-widest border-b border-white/20 pb-1">
                                    Prediction
                                </span>
                                <span className="text-xl md:text-2xl">{status.icon}</span>
                            </div>
                            
                            <div className="relative z-10 flex items-baseline mt-2 md:mt-4">
                                <span className="text-4xl md:text-6xl font-extrabold tracking-tight">
                                    {value.toFixed(2)}
                                </span>
                                <span className="ml-2 text-sm md:text-xl font-medium opacity-80">PSU</span>
                            </div>
                            
                            <div className={`mt-1 md:mt-2 font-bold text-sm md:text-lg ${status.color}`}>
                                {status.text}
                            </div>
                            
                            <div className="mt-4 flex items-center gap-2">
                                <div className="h-1.5 w-1.5 rounded-full bg-white animate-pulse"></div>
                                <span className="text-[10px] md:text-xs font-medium text-white/80">AI Confidence: High</span>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}