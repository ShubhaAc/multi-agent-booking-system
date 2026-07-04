import { useEffect, useRef } from "react";
import MessageBubble from "./MessageBubble";
import TypingIndicator from "./TypingIndicator";

export default function ChatWindow({ messages, isSending }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isSending]);

  return (
    <div className="chat-window">
      {messages.length === 0 && (
        <div className="empty-state">
          👋 Hi! Ask me to book, cancel, or reschedule an appointment, or ask about our
          services.
        </div>
      )}
      {messages.map((msg, i) => (
        <MessageBubble key={i} role={msg.role} content={msg.content} />
      ))}
      {isSending && <TypingIndicator />}
      <div ref={bottomRef} />
    </div>
  );
}