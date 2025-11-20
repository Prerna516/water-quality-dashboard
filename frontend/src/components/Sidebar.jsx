import React from 'react';

export default function Sidebar({ data, onRiverSelect, currentRiver }) {
    
    // DEFINING THE 3 RIVERS HERE
    const rivers = [
        { id: "netravati",  name: "Netravati River",  coords: [12.86, 74.85] },
        { id: "kali",       name: "Kali River",       coords: [14.84, 74.13] },
        { id: "sharavathi", name: "Sharavathi River", coords: [14.28, 74.45] },
    ];

    return (
        <div className="flex flex-col h-full gap-4 text-gray-800">
            <div className="border-b-2 border-blue-500 pb-4">
                <h1 className="text-3xl font-extrabold text-blue-900 tracking-tight">AQUA MONITOR</h1>
                <p className="text-sm text-blue-600 font-medium uppercase tracking-wider">Salinity Prediction Model</p>
            </div>

            {/* RIVER SELECTION DROPDOWN */}
            <div className="bg-blue-50 p-4 rounded-xl border border-blue-100">
                 <label className="text-xs font-bold text-blue-800 uppercase mb-2 block">Select River Basin</label>
                 <select 
                    className="w-full p-3 bg-white border border-blue-200 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none font-medium"
                    value={currentRiver} // Shows current selection
                    onChange={(e) => {
                        const selectedId = e.target.value;
                        const r = rivers.find(riv => riv.id === selectedId);
                        if(r) onRiverSelect(selectedId, r.coords);
                    }}
                 >
                    {rivers.map(r => <option key={r.id} value={r.id}>{r.name}</option>)}
                 </select>
            </div>

            <div className="flex-grow">
                {!data ? (
                    <div className="h-64 flex items-center justify-center text-center p-6 bg-gray-50 border-2 border-dashed border-gray-300 rounded-xl">
                        <p className="text-gray-500 font-medium">
                            Click on the water to predict salinity levels.
                        </p>
                    </div>
                ) : (
                    <div className="space-y-4 animate-fade-in">
                        <div className="bg-gray-800 text-white p-4 rounded-xl">
                             <p className="text-xs text-gray-400 uppercase tracking-wider mb-1">Selected Coordinates</p>
                             <p className="font-mono text-lg">
                                 {data.latitude?.toFixed(4)} N, {data.longitude?.toFixed(4)} E
                             </p>
                        </div>

                        <div className="mt-6">
                            <h2 className="text-xl font-bold text-gray-800 mb-3">Analysis Result</h2>
                            <div className="bg-blue-600 text-white p-6 rounded-2xl shadow-lg shadow-blue-200">
                                <span className="text-sm uppercase font-bold opacity-80 tracking-wider">Predicted Salinity</span>
                                <div className="flex items-baseline mt-2">
                                    <span className="text-5xl font-extrabold">
                                        {(data.ai_salinity_prediction || 0).toFixed(3)}
                                    </span>
                                    <span className="text-xl font-medium ml-2 opacity-90">PSU</span>
                                </div>
                                <div className="mt-4 text-sm bg-blue-700 inline-block px-3 py-1 rounded-full">
                                    AI Confidence: High
                                </div>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}