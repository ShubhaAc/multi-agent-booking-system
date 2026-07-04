import { useState, useCallback } from "react";
import { v4 as uuidv4 } from "uuid";
import { sendMessage, resetConversation } from "../api/chatApi";

const SENDER_ID_KEY = "brightsmile_sender_id";

function getOrCreateSenderId() {
  let id = localStorage.getItem(SENDER_ID_KEY);
  if (!id) {
    id = uuidv4();
    localStorage.setItem(SENDER_ID_KEY, id);
  }
  return id;
}

export function useChat() {
  const [senderId, setSenderId] = useState(getOrCreateSenderId);
  const [messages, setMessages] = useState([]);
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState(null);

  const send = useCallback(
    async (text) => {
      const trimmed = text.trim();
      if (!trimmed || isSending) return;

      setError(null);
      setMessages((prev) => [...prev, { role: "user", content: trimmed }]);
      setIsSending(true);

      try {
        const reply = await sendMessage(trimmed, senderId);
        setMessages((prev) => [...prev, { role: "assistant", content: reply }]);
      } catch (err) {
        setError("Something went wrong reaching the assistant. Please try again.");
        console.error(err);
      } finally {
        setIsSending(false);
      }
    },
    [senderId, isSending]
  );

  const startNewConversation = useCallback(async () => {
    try {
      await resetConversation(senderId);
    } catch (err) {
      console.error("Reset failed:", err);
    }
    const newId = uuidv4();
    localStorage.setItem(SENDER_ID_KEY, newId);
    setSenderId(newId);
    setMessages([]);
    setError(null);
  }, [senderId]);

  return { messages, isSending, error, send, startNewConversation, senderId };
}