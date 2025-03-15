// pages/index.js
import { useState } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from "recharts";

export default function Home() {
  const [transaction, setTransaction] = useState("");
  const [result, setResult] = useState(null);
  const [trendData, setTrendData] = useState([]);

  // Call REST API to detect fraud
  const detectFraud = async () => {
    try {
      const res = await fetch(
        `http://localhost:8002/detect_fraud?transaction=${encodeURIComponent(
          transaction
        )}`
      );
      const json = await res.json();
      setResult(json);
      setTrendData((prev) => [
        ...prev,
        { transaction: transaction, fraud_score: json.fraud_score },
      ]);
    } catch (err) {
      console.error("Error calling API:", err);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <h1 className="text-3xl font-bold text-center mb-8">
        Fraud Detection Dashboard
      </h1>
      <div className="max-w-xl mx-auto">
        <input
          type="text"
          placeholder="Enter transaction text..."
          className="w-full p-3 border rounded"
          value={transaction}
          onChange={(e) => setTransaction(e.target.value)}
        />
        <button
          onClick={detectFraud}
          className="mt-4 w-full bg-blue-500 text-white p-3 rounded hover:bg-blue-600"
        >
          Detect Fraud
        </button>
        {result && (
          <div className="mt-6 p-4 bg-white rounded shadow">
            <p>
              <strong>Transaction:</strong> {result.transaction}
            </p>
            <p>
              <strong>Fraud Score:</strong> {result.fraud_score.toFixed(2)}
            </p>
            <p>
              <strong>Status:</strong> {result.is_fraud ? "Fraudulent" : "Normal"}
            </p>
          </div>
        )}
      </div>
      <div className="mt-12 max-w-4xl mx-auto">
        <h2 className="text-2xl font-semibold mb-4">Fraud Score Trend</h2>
        <LineChart width={800} height={300} data={trendData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="transaction" hide />
          <YAxis domain={[0, 1]} />
          <Tooltip />
          <Line type="monotone" dataKey="fraud_score" stroke="#8884d8" />
        </LineChart>
      </div>
    </div>
  );
}
