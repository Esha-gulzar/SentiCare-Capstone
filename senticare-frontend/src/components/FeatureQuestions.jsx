import React, { useEffect, useState } from "react";
import { generateResponse } from "../services/api";

export default function FeatureQuestions({

  screeningAnswers,
  featureAnswers,
  setFeatureAnswers,
  setResponse,
  setStage

}) {

  const [questions,setQuestions] = useState({});

  useEffect(()=>{

    fetch("http://localhost:5000/questions/anxiety")
      .then(res=>res.json())
      .then(data=>setQuestions(data))

  },[])

  function handleChange(key,value){

    setFeatureAnswers({
      ...featureAnswers,
      [key]: value
    });

  }

  async function submit(){

    const res = await generateResponse({

      screening_responses: screeningAnswers,
      feature_responses: featureAnswers

    });

    setResponse(res.data);
    setStage("response");

  }

  return(

    <div>

      <h3>Tell us more</h3>

      {Object.keys(questions).map(key => {

        const q = questions[key];

        return(

          <div key={key}>

            <p>{q.question}</p>

            {q.input_type === "number" && (
              <input
                type="number"
                onChange={e=>handleChange(key,e.target.value)}
              />
            )}

            {q.input_type === "slider" && (
              <input
                type="range"
                min={q.min}
                max={q.max}
                onChange={e=>handleChange(key,e.target.value)}
              />
            )}

            {q.input_type === "select" && (
              <select onChange={e=>handleChange(key,e.target.value)}>

                {q.options.map(o=>(
                  <option key={o}>{o}</option>
                ))}

              </select>
            )}

          </div>

        )

      })}

      <button onClick={submit}>
        Submit
      </button>

    </div>

  )

}