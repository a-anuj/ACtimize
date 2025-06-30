import { useState } from 'react';
import './MainBody.css'; 
import axios from 'axios';


function App() {
  const [hours, setHours] = useState(6);
  const [tempSet, setTempSet] = useState(24);
  const [roomSize, setRoomSize] = useState(120);
  const [outsideTemp, setOutsideTemp] = useState(35);
  const [acType, setAcType] = useState(1.5);
  const [response, setResponse] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();

    const prompt = `I use my AC for ${hours} hours a day at ${tempSet} degrees. My room is ${roomSize} sq ft, the outside temperature is ${outsideTemp}°C, and it's a ${acType} ton AC. Can you tell me the electricity cost and suggest improvements?`;

    try {
      const res = await axios.post('http://localhost:5000/predict', { prompt });
      setResponse(res.data.response);
    } catch (err) {
      console.error('API error:', err);
      setResponse('Something went wrong. Try again.');
    }
  };

  return (
    <div className="App">
      <h1>💡 AC Electricity Cost Predictor</h1>
      <form onSubmit={handleSubmit}>
        <label>Hours AC runs/day:
          <input type="number" value={hours} onChange={(e) => setHours(e.target.value)} />
        </label>
        <label>Temperature set:
          <select value={tempSet} onChange={(e) => setTempSet(e.target.value)}>
            {[22, 23, 24, 25, 26].map(temp => (
              <option key={temp} value={temp}>{temp}°C</option>
            ))}
          </select>
        </label>
        <label>Room size (sq ft):
          <input type="number" value={roomSize} onChange={(e) => setRoomSize(e.target.value)} />
        </label>
        <label>Outside temperature:
          <input type="number" value={outsideTemp} onChange={(e) => setOutsideTemp(e.target.value)} />
        </label>
        <label>AC Type (ton):
          <select value={acType} onChange={(e) => setAcType(e.target.value)}>
            {[1.0, 1.5, 2.0].map(type => (
              <option key={type} value={type}>{type} ton</option>
            ))}
          </select>
        </label>
        <button type="submit">🔍 Predict</button>
      </form>

      {response && (
        <div>
          <br/>
            <div dangerouslySetInnerHTML={{ __html: response }} />
        </div>
        )}
    </div>
  );
}

export default App;
