import re
import json
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum
import openai
from openai import OpenAI
import os

class ThoughtType(Enum):
    CASE_ANALYSIS = "case_analysis"
    ELEMENT_MATCHING = "element_matching"
    CONFIDENCE_ASSESSMENT = "confidence_assessment"
    FINAL_PREDICTION = "final_prediction"

@dataclass
class Thought:
    content: str
    thought_type: ThoughtType
    confidence: float
    supporting_evidence: List[str]
    predicted_statutes: List[str]

class TreeOfThoughtsLegalPredictor:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        """
        Initialize the predictor with OpenAI API
        
        Args:
            api_key: OpenAI API key (if None, will try to get from environment)
            model: OpenAI model to use (default: gpt-4)
        """
        # Fixed: Use proper environment variable name or the provided API key
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError(
                "OpenAI API key must be provided either:\n"
                "1. As api_key parameter: TreeOfThoughtsLegalPredictor(api_key='your-key')\n"
                "2. As environment variable: export OPENAI_API_KEY='your-key'\n"
                "3. In a .env file: OPENAI_API_KEY=your-key"
            )
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        
        self.statutes = {
            "Indian Penal Code 498A": "Whoever, being the husband or the relative of the husband of a woman, subjects such woman to cruelty shall be punished with imprisonment for a term which may extend to three years and shall also be liable to fine.",
            "Indian Penal Code 506": "Whoever commits the offence of criminal intimidation shall be punished with imprisonment of either description for a term which may extend to two years, or with fine, or with both",
            "Indian Penal Code 147": "Whoever is guilty of rioting, shall be punished with imprisonment of either description for a term which may extend to two years, or with fine, or with both.",
            "Indian Penal Code 201": "Whoever, knowing or having reason to believe that an offence has been committed, causes any evidence of the commission of that offence to disappear, with the intention of screening the offender from legal punishment, or with that intention gives any information respecting the offence which he knows or believes to be false;",
            "Indian Penal Code 302": "Whoever commits murder shall be punished with death, or imprisonment for life, and shall also be liable to fine.",
            "Indian Penal Code 376": "Whoever, except in the cases provided for in sub-section (2), commits rape, shall be punished with rigorous imprisonment of either description for a term which shall not be less than ten years, but which may extend to imprisonment for life, and shall also be liable to fine.",
            "Indian Penal Code 420": "Whoever cheats and thereby dishonestly induces the person deceived to deliver any property to any person, or to make, alter or destroy the whole or any part of a valuable security, or anything which is signed or sealed, and which is capable of being converted into a valuable security, shall be punished with imprisonment of either description for a term which may extend to seven years, and shall also be liable to fine."
        }
    
    def _call_openai(self, prompt: str, temperature: float = 0.3) -> str:
        """Make a call to OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert legal analyst specializing in Indian Penal Code. Provide precise, factual analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=4000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return ""
    
    def generate_case_analysis_thoughts(self, case_text: str) -> List[Thought]:
        """Generate initial thoughts analyzing the case using OpenAI"""
        thoughts = []
        
        # Extract key facts using OpenAI
        facts_prompt = f"""
        Analyze the following legal case and extract the key facts. Focus on:
        1. Actions taken by parties
        2. Relationships between parties
        3. Evidence mentioned
        4. Outcomes or consequences
        5. Intent or motive
        
        Case: {case_text}
        
        Provide a concise list of key facts in bullet points.
        """
        
        facts_response = self._call_openai(facts_prompt)
        key_facts = [fact.strip('- ').strip() for fact in facts_response.split('\n') if fact.strip().startswith('-')]
        
        thoughts.append(Thought(
            content=f"Key facts identified: {'; '.join(key_facts)}",
            thought_type=ThoughtType.CASE_ANALYSIS,
            confidence=0.9,
            supporting_evidence=key_facts,
            predicted_statutes=[]
        ))
        
        # Identify parties using OpenAI
        parties_prompt = f"""
        Identify all parties involved in the following legal case. Categorize them as:
        - Accused/Defendant
        - Victim/Complainant
        - Witness
        - Other parties
        
        Case: {case_text}
        
        List each party type found in the case.
        """
        
        parties_response = self._call_openai(parties_prompt)
        parties = [party.strip('- ').strip() for party in parties_response.split('\n') if party.strip()]
        
        thoughts.append(Thought(
            content=f"Parties involved: {'; '.join(parties)}",
            thought_type=ThoughtType.CASE_ANALYSIS,
            confidence=0.8,
            supporting_evidence=parties,
            predicted_statutes=[]
        ))
        
        # Identify criminal acts using OpenAI
        acts_prompt = f"""
        Identify potential criminal acts in the following case. Focus on:
        - Physical violence or assault
        - Threats or intimidation
        - Sexual offenses
        - Property crimes
        - Evidence tampering
        - Murder or homicide
        - Domestic violence
        
        Case: {case_text}
        
        List the criminal acts you identify.
        """
        
        acts_response = self._call_openai(acts_prompt)
        criminal_acts = [act.strip('- ').strip() for act in acts_response.split('\n') if act.strip()]
        
        thoughts.append(Thought(
            content=f"Potential criminal acts: {'; '.join(criminal_acts)}",
            thought_type=ThoughtType.CASE_ANALYSIS,
            confidence=0.9,
            supporting_evidence=criminal_acts,
            predicted_statutes=[]
        ))
        
        return thoughts
    
    def generate_element_matching_thoughts(self, case_text: str) -> List[Thought]:
        """Generate thoughts matching case elements to statutes using OpenAI"""
        thoughts = []
        
        statutes_text = "\n".join([f"{code}: {desc}" for code, desc in self.statutes.items()])
        
        matching_prompt = f"""
        You are analyzing a legal case to determine which Indian Penal Code statutes might apply.
        
        Available statutes:
        {statutes_text}
        
        Case to analyze: {case_text}
        
        For each statute that might apply, provide:
        1. The statute name (e.g., "Indian Penal Code 302")
        2. Confidence score (0.0 to 1.0) based on how well the case facts match the statute
        3. Supporting evidence from the case
        
        Only include statutes with confidence > 0.5. Format your response as:
        STATUTE: [statute name]
        CONFIDENCE: [score]
        EVIDENCE: [supporting evidence]
        ---
        
        If no statutes apply with confidence > 0.5, respond with "NO APPLICABLE STATUTES"
        """
        
        response = self._call_openai(matching_prompt, temperature=0.1)
        
        if "NO APPLICABLE STATUTES" in response:
            return thoughts
        
        # Parse the response
        statute_blocks = response.split('---')
        
        for block in statute_blocks:
            if not block.strip():
                continue
                
            lines = [line.strip() for line in block.strip().split('\n') if line.strip()]
            statute_name = ""
            confidence = 0.0
            evidence = ""
            
            for line in lines:
                if line.startswith('STATUTE:'):
                    statute_name = line.replace('STATUTE:', '').strip()
                elif line.startswith('CONFIDENCE:'):
                    try:
                        confidence = float(line.replace('CONFIDENCE:', '').strip())
                    except:
                        confidence = 0.0
                elif line.startswith('EVIDENCE:'):
                    evidence = line.replace('EVIDENCE:', '').strip()
            
            if statute_name and confidence > 0.2:
                thoughts.append(Thought(
                    content=f"Elements found for {statute_name}: {evidence}",
                    thought_type=ThoughtType.ELEMENT_MATCHING,
                    confidence=confidence,
                    supporting_evidence=[evidence],
                    predicted_statutes=[statute_name]
                ))
        
        return thoughts
    
    def generate_confidence_assessment_thoughts(self, element_thoughts: List[Thought]) -> List[Thought]:
        """Generate thoughts assessing confidence in statute predictions using OpenAI"""
        thoughts = []
        
        if not element_thoughts:
            return thoughts
        
        # Group by statute
        statute_info = {}
        for thought in element_thoughts:
            if thought.predicted_statutes:
                statute = thought.predicted_statutes[0]
                if statute not in statute_info:
                    statute_info[statute] = {
                        'confidences': [],
                        'evidence': []
                    }
                statute_info[statute]['confidences'].append(thought.confidence)
                statute_info[statute]['evidence'].extend(thought.supporting_evidence)
        
        # Use OpenAI to assess overall confidence
        for statute, info in statute_info.items():
            avg_confidence = sum(info['confidences']) / len(info['confidences'])
            evidence_text = '; '.join(info['evidence'])
            
            assessment_prompt = f"""
            Assess the overall confidence for applying {statute} to this case.
            
            Statute: {self.statutes.get(statute, "Unknown statute")}
            
            Evidence from case: {evidence_text}
            
            Initial confidence scores: {info['confidences']}
            
            Provide a final confidence assessment (0.0 to 1.0) considering:
            1. Strength of evidence
            2. Completeness of elements
            3. Legal precedent alignment
            
            Respond with only the confidence score (e.g., 0.75)
            """
            
            confidence_response = self._call_openai(assessment_prompt, temperature=0.1)
            
            try:
                final_confidence = float(confidence_response.strip())
            except:
                final_confidence = avg_confidence
            
            thoughts.append(Thought(
                content=f"Confidence assessment for {statute}: {final_confidence:.2f}",
                thought_type=ThoughtType.CONFIDENCE_ASSESSMENT,
                confidence=final_confidence,
                supporting_evidence=info['evidence'],
                predicted_statutes=[statute]
            ))
        
        return thoughts
    
    def generate_final_prediction_thought(self, confidence_thoughts: List[Thought], threshold: float = 0.3) -> Thought:
        """Generate final prediction using OpenAI for final reasoning"""
        if not confidence_thoughts:
            return Thought(
                content="Final prediction: []",
                thought_type=ThoughtType.FINAL_PREDICTION,
                confidence=0.0,
                supporting_evidence=["No statutes meet confidence threshold"],
                predicted_statutes=[]
            )
        
        # Prepare data for final decision
        statute_data = []
        for thought in confidence_thoughts:
            if thought.confidence >= threshold and thought.predicted_statutes:
                statute_data.append({
                    'statute': thought.predicted_statutes[0],
                    'confidence': thought.confidence,
                    'evidence': thought.supporting_evidence
                })
        
        if not statute_data:
            return Thought(
                content="Final prediction: []",
                thought_type=ThoughtType.FINAL_PREDICTION,
                confidence=0.0,
                supporting_evidence=["No statutes meet confidence threshold"],
                predicted_statutes=[]
            )
        
        # Use OpenAI for final ranking and selection
        statute_info = "\n".join([
            f"- {data['statute']}: Confidence {data['confidence']:.2f}, Evidence: {'; '.join(data['evidence'])}"
            for data in statute_data
        ])
        
        final_prompt = f"""
        Based on the following statute analysis, provide the final ranked list of applicable statutes.
        Only include statutes you are confident about (>= {threshold} confidence).
        
        Analyzed statutes:
        {statute_info}
        
        Provide the final list in this exact format:
        ["statute1", "statute2", "statute3"]
        
        If no statutes are applicable, respond with: []
        
        Consider legal precedent and strength of evidence in your ranking.
        """
        
        final_response = self._call_openai(final_prompt, temperature=0.1)
        
        try:
            # Extract list from response
            import ast
            final_statutes = ast.literal_eval(final_response.strip())
            if not isinstance(final_statutes, list):
                final_statutes = []
        except:
            # Fallback to original logic
            final_statutes = [data['statute'] for data in sorted(statute_data, key=lambda x: x['confidence'], reverse=True)]
        
        max_confidence = max([data['confidence'] for data in statute_data]) if statute_data else 0.0
        
        return Thought(
            content=f"Final prediction: {final_statutes}",
            thought_type=ThoughtType.FINAL_PREDICTION,
            confidence=max_confidence,
            supporting_evidence=[f"High confidence match for {s}" for s in final_statutes],
            predicted_statutes=final_statutes
        )
    
    def predict_statutes(self, case_text: str, confidence_threshold: float = 0.3, verbose: bool = False) -> List[str]:
        """Main method to predict statutes using Tree-of-Thoughts approach with OpenAI"""
        all_thoughts = []
        
        if verbose:
            print("Using OpenAI model:", self.model)
            print("=" * 50)
        
        # Step 1: Case Analysis
        if verbose:
            print("=== CASE ANALYSIS THOUGHTS (OpenAI-powered) ===")
        case_analysis_thoughts = self.generate_case_analysis_thoughts(case_text)
        all_thoughts.extend(case_analysis_thoughts)
        
        if verbose:
            for thought in case_analysis_thoughts:
                print(f"- {thought.content}")
            print()
        
        # Step 2: Element Matching
        if verbose:
            print("=== ELEMENT MATCHING THOUGHTS (OpenAI-powered) ===")
        element_matching_thoughts = self.generate_element_matching_thoughts(case_text)
        all_thoughts.extend(element_matching_thoughts)
        
        if verbose:
            for thought in element_matching_thoughts:
                print(f"- {thought.content} (Confidence: {thought.confidence:.2f})")
            print()
        
        # Step 3: Confidence Assessment
        if verbose:
            print("=== CONFIDENCE ASSESSMENT THOUGHTS (OpenAI-powered) ===")
        confidence_assessment_thoughts = self.generate_confidence_assessment_thoughts(element_matching_thoughts)
        all_thoughts.extend(confidence_assessment_thoughts)
        
        if verbose:
            for thought in confidence_assessment_thoughts:
                print(f"- {thought.content}")
            print()
        
        # Step 4: Final Prediction
        final_thought = self.generate_final_prediction_thought(confidence_assessment_thoughts, confidence_threshold)
        all_thoughts.append(final_thought)
        
        if verbose:
            print("=== FINAL PREDICTION (OpenAI-powered) ===")
            print(f"- {final_thought.content}")
            print()
        
        return final_thought.predicted_statutes

# Example usage and testing
def doc_prediction(gr_path):
    json_2=open(gr_path)
    jgr=json.load(json_2)
    docs=list(jgr.keys())
    fact_list=[]
    for doc in docs:
        fact_list.append(jgr[doc]['fact'])
    return fact_list

def main():
    # Option 1: Pass API key directly
    predictor = TreeOfThoughtsLegalPredictor(
        api_key="",
        model="gpt-4"  # or "gpt-4" for better quality
    )
    
    # Option 2: Set environment variable (recommended)
    # os.environ["OPENAI_API_KEY"] = "your-api-key"
    # predictor = TreeOfThoughtsLegalPredictor(model="gpt-3.5-turbo")
    sections=['147', '420', '201','506', '302', '376', '498A']
    test_cases=[]
    for section in sections:
        dir_2 = f"/home/subinay/Documents/data/statute_prediction/statute_dataset/statute_dataset-main/Test_doc/test_collection_IPC_{section}.json"
        test_cases.append(doc_prediction(dir_2))
    print(len(test_cases))
    test_cases= [item for sublist in test_cases for item in sublist]
    
    
    
    
    print("TREE-OF-THOUGHTS LEGAL STATUTE PREDICTOR (OpenAI-Powered)")
    print("=" * 60)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\nTEST CASE {i}:")
        print("-" * 30)
        print(f"Case: {case.strip()}")
        print()
        
        try:
            predicted_statutes = predictor.predict_statutes(case, verbose=True)
            print(f"FINAL RESULT: {predicted_statutes}")
        except Exception as e:
            print(f"Error processing case: {e}")
        
        print("=" * 60)

if __name__ == "__main__":
    main()
