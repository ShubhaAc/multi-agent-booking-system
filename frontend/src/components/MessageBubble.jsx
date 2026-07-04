import { useState } from "react";

export default function MessageBubble({ role, content }) {
  const isUser = role === "user";
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <div className={`message-row ${isUser ? "message-row-user" : "message-row-assistant"}`}>
      <div className={`avatar ${isUser ? "avatar-user" : "avatar-assistant"}`}>
        {isUser ? "U" : "B"}
      </div>
      <div className="message-content">
        <div className={`message-bubble ${isUser ? "message-bubble-user" : "message-bubble-assistant"}`}>
          {content}
        </div>
        {!isUser && (
          <button className="copy-button" onClick={handleCopy}>
            {copied ? "Copied" : "Copy"}
          </button>
        )}
      </div>
    </div>
  );
}