import axios from "axios";

const API = axios.create({
  baseURL: "http://localhost:5000"
});

export const getScreeningQuestions = () =>
  API.get("/questions/screening");

export const getFeatureQuestions = (condition) =>
  API.get(`/questions/${condition}`);

export const generateResponse = (data) =>
  API.post("/generate_response", data);