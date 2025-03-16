// pages/index.js
import { useState, useEffect } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from "recharts";

// ClientOnly component renders its children only on the client
function ClientOnly({ children }) {
  const [mounted, setMounted] = useState(false);
  useEffect(() => {
    setMounted(true);
  }, []);
  return mounted ? children : null;
}

export default function Home() {
  const [transaction, setTransaction] = useState("");
  const [result, setResult] = useState(null);
  const [trendData, setTrendData] = useState([]);

  // Call REST API to detect fraud
  const detectFraud = async () => {
    try {
      const API_URL = process.env.NEXT_PUBLIC_API_URL;
      const res = await fetch(
        `${API_URL}/detect_fraud?transaction=${encodeURIComponent(
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
    <div className="min-h-screen p-6 bg-gray-100">
      <h1 className="text-3xl font-bold text-center mb-4">
        Fraud Detection Dashboard
      </h1>
      <div className="max-w-xl mx-auto">
        <input
          type="text"
          placeholder="Enter transaction text..."
          className="w-full p-3 border rounded mb-4"
          value={transaction}
          onChange={(e) => setTransaction(e.target.value)}
        />
        <button
          onClick={detectFraud}
          className="w-full bg-blue-500 text-white p-3 rounded hover:bg-blue-600"
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
              <strong>Status:</strong>{" "}
              {result.is_fraud ? "Fraudulent" : "Normal"}
            </p>
          </div>
        )}
      </div>
      <div className="mt-12 max-w-4xl mx-auto">
        <h2 className="text-2xl font-semibold mb-4 text-center">
          Fraud Score Trend
        </h2>
        <ClientOnly>
          <LineChart width={800} height={300} data={trendData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="transaction" hide />
            <YAxis domain={[0, 1]} />
            <Tooltip />
            <Line type="monotone" dataKey="fraud_score" stroke="#8884d8" />
          </LineChart>
        </ClientOnly>
      </div>
    </div>
  );
}