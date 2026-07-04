import { useState, useRef, useEffect } from "react";

export default function MessageInput({ onSend, disabled }) {
  const [value, setValue] = useState("");
  const textareaRef = useRef(null);

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }, [value]);

  const submit = () => {
    if (!value.trim() || disabled) return;
    onSend(value);
    setValue("");
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <div className="input-area">
      <div className="message-input-form">
        <textarea
          ref={textareaRef}
          className="message-input"
          placeholder="Message BrightSmile Assistant..."
          value={value}
          rows={1}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled}
        />
        <button
          className="send-button"
          type="button"
          onClick={submit}
          disabled={disabled || !value.trim()}
          aria-label="Send message"
        >
          ➤
        </button>
      </div>
      <div className="hint-text">Enter to send · Shift+Enter for new line</div>
    </div>
  );
}