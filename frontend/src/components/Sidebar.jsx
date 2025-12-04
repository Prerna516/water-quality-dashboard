import React, { useEffect, useState } from 'react';
import axios from 'axios';

// Use Localhost for now
const BACKEND_URL = "http://127.0.0.1:8000"; 

export default function Sidebar({ 
    data, 
    onRiverSelect, 
    currentRiver, 
    activeDate, 
    onDateChange,
    activeParam,     
    onParamChange    
}) {
    
    const [availableDates, setAvailableDates] = useState([]);

    const rivers = [
        { id: "netravati",  name: "Netravati River",  coords: [12.86, 74.85] },
        { id: "kali",       name: "Kali River",       coords: [14.84, 74.13] },
        { id: "sharavathi", name: "Sharavathi River", coords: [14.28, 74.45] },
        { id: "nandini",    name: "Nandini River",    coords: [13.05, 74.78] },
        { id: "gangavali",  name: "Gangavali River",  coords: [14.58, 74.32] },
    ];

    // Fetch Dates
    useEffect(() => {
        axios.get(`${BACKEND_URL}/api/dates?river=${currentRiver}`)
            .then(res => {
                setAvailableDates(res.data);
                if(res.data.length > 0 && (!activeDate || !res.data.includes(activeDate))) {
                    onDateChange(res.data[res.data.length - 1]);
                }
            })
            .catch(err => console.error(err));
    }, [currentRiver]);

    // --- DYNAMIC DISPLAY LOGIC ---
    const getStatus = (val, param) => {
        // SALINITY
        if (param === "salinity") {
            if (val < 0.5) return { text: "Fresh Water", color: "bg-blue-600", icon: "💧", unit: "PSU" };
            if (val < 25)  return { text: "Brackish", color: "bg-purple-600", icon: "⚠️", unit: "PSU" };
            return { text: "Saline", color: "bg-yellow-500", icon: "🌊", unit: "PSU" };
        }
        // TURBIDITY
        if (param === "turbidity") {
            if (val < 5) return { text: "Clear", color: "bg-cyan-600", icon: "💎", unit: "NTU" };
            return { text: "Turbid", color: "bg-amber-600", icon: "🌫️", unit: "NTU" };
        }
        // CHLOROPHYLL
        if (param === "chlorophyll") {
            if (val < 5) return { text: "Low Algae", color: "bg-teal-600", icon: "🌿", unit: "mg/m³" };
            return { text: "High Algae", color: "bg-green-700", icon: "🦠", unit: "mg/m³" };
        }
        return { text: "--", color: "bg-gray-500", icon: "?" };
    };

    const displayValue = data?.value || 0;
    const status = getStatus(displayValue, activeParam);

    return (
        <div className="flex flex-col h-full gap-3 md:gap-4 text-gray-800 font-sans">
            
            {/* Header */}
            <div className="border-b-2 border-blue-600 pb-2 md:pb-4">
                <h1 className="text-xl md:text-3xl font-extrabold text-blue-900 tracking-tight">AQUA MONITOR</h1>
                <p className="text-[10px] md:text-sm text-blue-600 font-bold uppercase tracking-wider">Multi-Parameter AI</p>
            </div>

            {/* Controls */}
            <div className="bg-blue-50 p-3 rounded-xl border border-blue-100 shadow-sm flex flex-col gap-3">
                 
                 {/* River Selector */}
                 <div>
                     <label className="text-[10px] md:text-xs font-bold text-blue-800 uppercase mb-1 block">River Basin</label>
                     <select 
                        className="w-full p-2 text-sm bg-white border border-blue-200 rounded-lg outline-none cursor-pointer"
                        value={currentRiver} 
                        onChange={(e) => {
                            const r = rivers.find(riv => riv.id === e.target.value);
                            if(r) onRiverSelect(e.target.value, r.coords);
                        }}
                     >
                        {rivers.map(r => <option key={r.id} value={r.id}>{r.name}</option>)}
                     </select>
                 </div>

                 {/* PARAMETER BUTTONS */}
                 <div>
                     <label className="text-[10px] md:text-xs font-bold text-blue-800 uppercase mb-1 block">Parameter</label>
                     <div className="flex gap-2">
                         {['salinity', 'turbidity', 'chlorophyll'].map(p => (
                             <button
                                key={p}
                                onClick={() => onParamChange(p)}
                                className={`flex-1 py-2 text-[10px] md:text-xs font-bold rounded-lg uppercase transition-all border shadow-sm
                                    ${activeParam === p 
                                        ? 'bg-blue-600 text-white border-blue-600' 
                                        : 'bg-white text-gray-600 border-gray-300 hover:bg-blue-50'
                                    }`}
                             >
                                {p.slice(0, 4)}.
                             </button>
                         ))}
                     </div>
                 </div>

                 {/* Date Selector */}
                 <div>
                     <label className="text-[10px] md:text-xs font-bold text-blue-800 uppercase mb-1 block">Date</label>
                     <select 
                        className="w-full p-2 text-sm bg-white border border-blue-200 rounded-lg outline-none cursor-pointer"
                        value={activeDate} 
                        onChange={(e) => onDateChange(e.target.value)}
                        disabled={availableDates.length === 0}
                     >
                        {availableDates.length === 0 && <option>No Data Available</option>}
                        {availableDates.map(d => <option key={d} value={d}>{d}</option>)}
                     </select>
                 </div>
            </div>

            {/* Results */}
            <div className="flex-grow overflow-y-auto">
                {!data ? (
                    <div className="h-40 md:h-64 flex flex-col items-center justify-center text-center p-4 bg-gray-50 border-2 border-dashed border-gray-300 rounded-xl">
                        <p className="text-gray-400 font-medium text-xs md:text-sm">
                            Select parameter & click map.
                        </p>
                    </div>
                ) : (
                    <div className="space-y-4 animate-fade-in mt-1">
                        <div className="flex justify-between items-center bg-gray-100 p-2 rounded-lg text-[10px] md:text-xs font-mono text-gray-600 border border-gray-200">
                             <span>LAT: {data.latitude?.toFixed(4)}</span>
                             <span>LON: {data.longitude?.toFixed(4)}</span>
                        </div>

                        {/* HERO CARD */}
                        <div className={`relative overflow-hidden ${status.color} text-white p-5 rounded-2xl shadow-lg transition-all duration-500`}>
                            <div className="flex justify-between items-start">
                                <span className="text-xs uppercase font-bold opacity-80 tracking-widest border-b border-white/20 pb-1">
                                    {activeParam}
                                </span>
                                <span className="text-3xl">{status.icon}</span>
                            </div>
                            
                            <div className="flex items-baseline mt-3">
                                <span className="text-5xl font-extrabold tracking-tight">
                                    {displayValue.toFixed(2)}
                                </span>
                                <span className="ml-2 text-lg font-medium opacity-80">{status.unit}</span>
                            </div>
                            
                            <div className="mt-2 font-bold text-lg">{status.text}</div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}