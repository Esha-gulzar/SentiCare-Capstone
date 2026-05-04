import React, { useState } from "react";
import ScreeningQuestions from "./ScreeningQuestions";
import FeatureQuestions from "./FeatureQuestions";
import CBTResponse from "./CBTResponse";

export default function ChatContainer() {

  const [stage, setStage] = useState("screening");
  const [screeningAnswers, setScreeningAnswers] = useState({});
  const [featureAnswers, setFeatureAnswers] = useState({});
  const [response, setResponse] = useState(null);

  return (
    <div className="chat-container">

      <div className="header">
        SentiCare AI 🧠
      </div>

      {stage === "screening" && (
        <ScreeningQuestions
          setStage={setStage}
          screeningAnswers={screeningAnswers}
          setScreeningAnswers={setScreeningAnswers}
        />
      )}

      {stage === "features" && (
        <FeatureQuestions
          screeningAnswers={screeningAnswers}
          featureAnswers={featureAnswers}
          setFeatureAnswers={setFeatureAnswers}
          setResponse={setResponse}
          setStage={setStage}
        />
      )}

      {stage === "response" && (
        <CBTResponse response={response} />
      )}

    </div>
  );
}