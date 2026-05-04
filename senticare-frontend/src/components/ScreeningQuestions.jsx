import React, { useEffect, useState } from "react";
import axios from "axios";

export default function ScreeningQuestions({
  setStage,
  screeningAnswers,
  setScreeningAnswers
}) {

  const [questions, setQuestions] = useState([]);

  useEffect(() => {

    axios.get("http://localhost:5000/questions/screening")
      .then(res => setQuestions(res.data.questions))

  }, []);

  function handleChange(id, value){

    setScreeningAnswers({
      ...screeningAnswers,
      [id]: Number(value)
    });

  }

  return (

    <div>

      <h3>Mental Health Check</h3>

      {questions.map(q => (

        <div key={q.id} className="question">

          <p>{q.question}</p>

          {[0,1,2,3].map(val => (

            <label key={val}>

              <input
                type="radio"
                name={q.id}
                value={val}
                onChange={() => handleChange(q.id,val)}
              />

              {val}

            </label>

          ))}

        </div>

      ))}

      <button onClick={()=>setStage("features")}>
        Next
      </button>

    </div>

  );

}