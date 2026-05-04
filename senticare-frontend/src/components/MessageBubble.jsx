import React from "react";

export default function MessageBubble({ sender, text }) {

  return (
    <div className={sender === "bot" ? "bot-bubble" : "user-bubble"}>
      {text}
    </div>
  );

}