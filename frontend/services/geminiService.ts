import { GoogleGenAI, Type } from "@google/genai";
import { DetectionResult } from "../types";

// Initialize the client
// API Key is guaranteed to be in process.env.API_KEY per system instructions
const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

export const checkFacePresence = async (base64Image: string): Promise<DetectionResult> => {
  try {
    // Strip the data:image/jpeg;base64, prefix if present
    const cleanBase64 = base64Image.replace(/^data:image\/(png|jpeg|webp);base64,/, "");

    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: {
        parts: [
          {
            inlineData: {
              mimeType: 'image/jpeg',
              data: cleanBase64
            }
          },
          {
            text: "Analyze this image strictly. Is there a human face clearly visible? Respond with a JSON object."
          }
        ]
      },
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            faceDetected: {
              type: Type.BOOLEAN,
              description: "True if a human face is clearly visible in the image, false otherwise."
            },
          },
          required: ["faceDetected"]
        }
      }
    });

    const text = response.text;
    if (!text) {
      throw new Error("No response from Gemini");
    }

    const result = JSON.parse(text) as DetectionResult;
    return result;

  } catch (error) {
    console.error("Gemini analysis failed:", error);
    // Fail safe to false to avoid false positives in safety systems
    return { faceDetected: false };
  }
};