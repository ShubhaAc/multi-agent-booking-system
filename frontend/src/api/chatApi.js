import { API_BASE_URL } from "../config";

export async function sendMessage(message, senderId) {
  const res = await fetch(`${API_BASE_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      sender_id: senderId,
      vector_store: "chroma",
    }),
  });
  if (!res.ok) {
    throw new Error(`Chat request failed: ${res.status}`);
  }
  const data = await res.json();
  return data.response;
}

export async function resetConversation(senderId) {
  const res = await fetch(`${API_BASE_URL}/reset/${senderId}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    throw new Error(`Reset request failed: ${res.status}`);
  }
  return res.json();
}