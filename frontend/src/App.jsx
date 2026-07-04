import { useState, useEffect } from "react";
import { useChat } from "./hooks/useChat";
import ChatWindow from "./components/ChatWindow";
import MessageInput from "./components/MessageInput";
import "./App.css";

export default function App() {
  const { messages, isSending, error, send, startNewConversation } = useChat();
  const [theme, setTheme] = useState("light");

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  return (
    <div className="app-shell">
      <header className="app-header">
        <h1>BrightSmile Dental Assistant</h1>
        <div className="header-actions">
          <button
            className="icon-button"
            onClick={() => setTheme(theme === "light" ? "dark" : "light")}
          >
            {theme === "light" ? "🌙" : "☀️"}
          </button>
          <button className="new-chat-button" onClick={startNewConversation}>
            New conversation
          </button>
        </div>
      </header>

      <ChatWindow messages={messages} isSending={isSending} />

      {error && <div className="error-banner">{error}</div>}

      <MessageInput onSend={send} disabled={isSending} />
    </div>
  );
}