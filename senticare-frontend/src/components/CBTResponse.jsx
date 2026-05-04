import React from "react";

export default function CBTResponse({response}){

  if(!response) return null;

  return(

    <div className="bot-message">

      <h3>Support Message</h3>

      <p><b>Validation:</b> {response.validation}</p>

      <p><b>Steps:</b></p>

      <ul>
        {response.intervention_steps.map((s,i)=>(
          <li key={i}>{s}</li>
        ))}
      </ul>

      <p><b>Grounding:</b> {response.grounding_statement}</p>

      <p><b>Reflection:</b></p>

      <ul>
        {response.guided_questions.map((q,i)=>(
          <li key={i}>{q}</li>
        ))}
      </ul>

    </div>

  )

}