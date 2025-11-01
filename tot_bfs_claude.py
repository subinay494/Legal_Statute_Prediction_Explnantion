import re
import json
from typing import List, Dict, Tuple, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import openai
from openai import OpenAI
import os
import copy
from collections import deque
from anthropic import Anthropic


class ThoughtType(Enum):
    ROOT = "root"
    FACT_EXTRACTION = "fact_extraction"
    LEGAL_ELEMENT_ANALYSIS = "legal_element_analysis"
    STATUTE_MATCHING = "statute_matching"
    CONFIDENCE_EVALUATION = "confidence_evaluation"
    FINAL_DECISION = "final_decision"

@dataclass
class SearchState:
    """Represents a state in the BFS search tree"""
    depth: int
    thought_content: str
    thought_type: ThoughtType
    confidence_score: float
    extracted_facts: List[str] = field(default_factory=list)
    identified_elements: Dict[str, List[str]] = field(default_factory=dict)
    statute_matches: Dict[str, float] = field(default_factory=dict)
    supporting_evidence: List[str] = field(default_factory=list)
    path_history: List[str] = field(default_factory=list)
    is_terminal: bool = False
    children: List['SearchState'] = field(default_factory=list)
    
    def add_child(self, child_state: 'SearchState'):
        """Add a child state to this state"""
        self.children.append(child_state)
        child_state.path_history = self.path_history + [self.thought_content]

class TOTBFSLegalPredictor:
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-20250514"):
        """
        Initialize the TOT-BFS predictor with Claude API
        
        Args:
            api_key: Anthropic API key (if None, will try to get from environment)
            model: Claude model to use (default: claude-3-5-sonnet-20241022)
        """
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        
        if not api_key:
            raise ValueError(
                "Anthropic API key must be provided either:\n"
                "1. As api_key parameter: TOTBFSLegalPredictor(api_key='your-key')\n"
                "2. As environment variable: export ANTHROPIC_API_KEY='your-key'\n"
                "3. In a .env file: ANTHROPIC_API_KEY=your-key"
            )
        
        self.client = Anthropic(api_key=api_key)
        self.model = model
        
        # Maximum depth for BFS - INCREASED FROM 5 TO 7
        self.max_depth = 7
        
        # Minimum confidence threshold for continuing a path
        self.confidence_threshold = 0.4
        
        # Maximum number of children per state
        self.max_children = 3
        
        # Maximum number of nodes to explore per level (BFS specific)
        self.max_nodes_per_level = 10
        
        # Best solution found so far
        self.best_solution: Optional[SearchState] = None
        self.best_score: float = 0.0
        
        # All terminal solutions found
        self.all_solutions: List[SearchState] = []
        
        # NEW: Collect high-quality non-terminal solutions as backup
        self.backup_solutions: List[SearchState] = []
        
        # Search statistics
        self.nodes_explored = 0
        self.levels_explored = 0
        
        # BFS queue
        self.search_queue: deque = deque()
        
        # Your existing statutes dictionary here...
        self.statutes = {
            "Indian Penal Code 498A": "Whoever, being the husband or the relative of the husband of a woman, subjects such woman to cruelty shall be punished with imprisonment for a term which may extend to three years and shall also be liable to fine.",
            "Indian Penal Code 506": "Whoever commits the offence of criminal intimidation shall be punished with imprisonment of either description for a term which may extend to two years, or with fine, or with both",
            "Indian Penal Code 147": "Whoever is guilty of rioting, shall be punished with imprisonment of either description for a term which may extend to two years, or with fine, or with both.",
            "Indian Penal Code 201": "Whoever, knowing or having reason to believe that an offence has been committed, causes any evidence of the commission of that offence to disappear, with the intention of screening the offender from legal punishment, or with that intention gives any information respecting the offence which he knows or believes to be false;",
            "Indian Penal Code 302": "Whoever commits murder shall be punished with death, or imprisonment for life, and shall also be liable to fine.",
            "Indian Penal Code 376": "Whoever, except in the cases provided for in sub-section (2), commits rape, shall be punished with rigorous imprisonment of either description for a term which shall not be less than ten years, but which may extend to imprisonment for life, and shall also be liable to fine.",
            "Indian Penal Code 420": "Whoever cheats and thereby dishonestly induces the person deceived to deliver any property to any person, or to make, alter or destroy the whole or any part of a valuable security, or anything which is signed or sealed, and which is capable of being converted into a valuable security, shall be punished with imprisonment of either description for a term which may extend to seven years, and shall also be liable to fine.",
            "Indian Penal Code 34": "When a criminal act is done by several persons in furtherance of the common intention of all, each of such persons is liable for that act in the same manner as if it were done by him alone."
        }
    
    def _call_claude(self, prompt: str, temperature: float = 0.1) -> str:
        """Make a call to Claude API"""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=3000,
                temperature=temperature,
                system="You are an expert legal analyst specializing in Indian Penal Code. Use systematic reasoning and provide precise analysis.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text.strip()
        except Exception as e:
            print(f"Error calling Claude API: {e}")
            return ""
    
    def evaluate_state(self, state: SearchState) -> float:
        """
        Evaluate the quality of a search state
        Returns a score between 0.0 and 1.0
        """
        if state.thought_type == ThoughtType.ROOT:
            return 0.5
        
        # Base score from confidence
        score = state.confidence_score
        
        # Bonus for having multiple types of evidence
        if len(state.supporting_evidence) > 2:
            score += 0.1
        
        # Bonus for depth (more detailed analysis)
        depth_bonus = min(state.depth * 0.05, 0.15)
        score += depth_bonus
        
        # Penalty for very low confidence paths
        if state.confidence_score < 0.3:
            score *= 0.5
        
        # Bonus for terminal states with good statute matches
        if state.is_terminal and state.statute_matches:
            avg_match_confidence = sum(state.statute_matches.values()) / len(state.statute_matches)
            score += avg_match_confidence * 0.2
        
        return min(score, 1.0)
    
    # NEW: Enhanced terminal condition check
    def is_terminal_node(self, state: SearchState) -> bool:
        """
        Enhanced terminal condition check
        """
        # Original terminal flag
        if state.is_terminal:
            return True
            
        # NEW: Consider high-confidence confidence evaluation nodes as potential terminals
        if (state.thought_type == ThoughtType.CONFIDENCE_EVALUATION and 
            state.confidence_score >= 0.7 and 
            state.statute_matches):
            return True
            
        # NEW: Consider high-quality statute matching nodes at sufficient depth
        if (state.thought_type == ThoughtType.STATUTE_MATCHING and 
            state.depth >= 4 and 
            state.confidence_score >= 0.8 and 
            state.statute_matches):
            return True
            
        return False
    
    def should_prune(self, state: SearchState) -> bool:
        """
        Decide whether to prune this branch of the search tree
        """
        # Prune if confidence is too low
        if state.confidence_score < self.confidence_threshold:
            return True
        
        # Prune if we've reached maximum depth
        if state.depth >= self.max_depth:
            return True
        
        # In BFS, we're less aggressive about pruning based on best score
        # since we explore breadth-wise and may find better solutions at the same level
        if self.best_score > 0 and self.evaluate_state(state) < self.best_score * 0.4:
            return True
        
        return False
    
    def generate_children_states(self, parent_state: SearchState, case_text: str) -> List[SearchState]:
        """Generate child states from a parent state using OpenAI"""
        children = []
        
        if parent_state.thought_type == ThoughtType.ROOT:
            # Generate fact extraction children
            children.extend(self._generate_fact_extraction_states(parent_state, case_text))
        
        elif parent_state.thought_type == ThoughtType.FACT_EXTRACTION:
            # Generate legal element analysis children
            children.extend(self._generate_legal_element_states(parent_state, case_text))
        
        elif parent_state.thought_type == ThoughtType.LEGAL_ELEMENT_ANALYSIS:
            # Generate statute matching children
            children.extend(self._generate_statute_matching_states(parent_state, case_text))
        
        elif parent_state.thought_type == ThoughtType.STATUTE_MATCHING:
            # Generate confidence evaluation children
            children.extend(self._generate_confidence_evaluation_states(parent_state, case_text))
        
        elif parent_state.thought_type == ThoughtType.CONFIDENCE_EVALUATION:
            # Generate final decision (terminal state)
            children.extend(self._generate_final_decision_states(parent_state, case_text))
        
        # Add children to parent and limit number
        for child in children[:self.max_children]:
            parent_state.add_child(child)
        
        return children[:self.max_children]
    
    def _generate_fact_extraction_states(self, parent: SearchState, case_text: str) -> List[SearchState]:
        """Generate fact extraction states"""
        children = []
        
        # Different approaches to fact extraction
        approaches = [
            "Extract chronological sequence of events",
            "Identify key actors and their relationships",
            "Focus on evidence and criminal acts"
        ]
        
        for i, approach in enumerate(approaches):
            prompt = f"""
            Using the approach: "{approach}"
            
            Analyze this legal case: {case_text}
            
            Extract the most important facts following this approach.
            Provide:
            1. List of key facts (max 5)
            2. Confidence in fact extraction (0.0-1.0)
            3. Supporting evidence from text
            
            Format:
            FACTS: [fact1, fact2, ...]
            CONFIDENCE: [score]
            EVIDENCE: [evidence1, evidence2, ...]
            """
            
            response = self._call_claude(prompt, temperature=0.2 + i*0.1)
            
            # Parse response
            facts, confidence, evidence = self._parse_fact_response(response)
            
            if facts:
                child = SearchState(
                    depth=parent.depth + 1,
                    thought_content=f"Fact extraction using {approach}: {facts}",
                    thought_type=ThoughtType.FACT_EXTRACTION,
                    confidence_score=confidence,
                    extracted_facts=facts,
                    supporting_evidence=evidence
                )
                children.append(child)
        
        return children
    
    def _generate_legal_element_states(self, parent: SearchState, case_text: str) -> List[SearchState]:
        """Generate legal element analysis states"""
        children = []
        
        # Focus on different legal aspects
        aspects = [
            "Criminal intent and mens rea",
            "Physical actions and actus reus", 
            "Relationships and contexts"
        ]
        
        for aspect in aspects:
            prompt = f"""
            Focus on: {aspect}
            
            Case facts: {parent.extracted_facts}
            Case text: {case_text}
            
            Analyze the legal elements present in this case.
            
            Available statutes to consider:
            {list(self.statutes.keys())}
            
            For each relevant statute, identify:
            1. Which legal elements are present
            2. Strength of evidence for each element
            3. Overall confidence
            
            Format:
            STATUTE: [statute name]
            ELEMENTS: [element1, element2, ...]
            CONFIDENCE: [score]
            ---
            """
            
            response = self._call_claude(prompt, temperature=0.3)
            
            # Parse response
            elements_dict, avg_confidence = self._parse_elements_response(response)
            
            if elements_dict:
                child = SearchState(
                    depth=parent.depth + 1,
                    thought_content=f"Legal elements analysis focusing on {aspect}",
                    thought_type=ThoughtType.LEGAL_ELEMENT_ANALYSIS,
                    confidence_score=avg_confidence,
                    extracted_facts=parent.extracted_facts,
                    identified_elements=elements_dict,
                    supporting_evidence=parent.supporting_evidence
                )
                children.append(child)
        
        return children
    
    def _generate_statute_matching_states(self, parent: SearchState, case_text: str) -> List[SearchState]:
        """Generate statute matching states"""
        children = []
        
        # Different matching strategies
        strategies = [
            "Strict element-by-element matching",
            "Contextual and precedent-based matching"
        ]
        
        for strategy in strategies:
            prompt = f"""
            Using strategy: {strategy}
            
            Case elements identified: {parent.identified_elements}
            Case facts: {parent.extracted_facts}
            
            Available statutes:
            {json.dumps(self.statutes, indent=2)}
            
            Match the case to applicable statutes using this strategy.
            
            For each potential statute match:
            1. Statute name
            2. Match confidence (0.0-1.0)  
            3. Reasoning for match
            
            Format:
            STATUTE: [name]
            CONFIDENCE: [score]
            REASONING: [explanation]
            ---
            """
            
            response = self._call_claude(prompt, temperature=0.2)
            
            # Parse response
            statute_matches, reasoning = self._parse_statute_matches(response)
            
            if statute_matches:
                avg_confidence = sum(statute_matches.values()) / len(statute_matches)
                
                child = SearchState(
                    depth=parent.depth + 1,
                    thought_content=f"Statute matching using {strategy}",
                    thought_type=ThoughtType.STATUTE_MATCHING,
                    confidence_score=avg_confidence,
                    extracted_facts=parent.extracted_facts,
                    identified_elements=parent.identified_elements,
                    statute_matches=statute_matches,
                    supporting_evidence=parent.supporting_evidence + reasoning
                )
                children.append(child)
        
        return children
    
    def _generate_confidence_evaluation_states(self, parent: SearchState, case_text: str) -> List[SearchState]:
        """Generate confidence evaluation states"""
        children = []
        
        if not parent.statute_matches:
            return children
        
        # Evaluate confidence for each statute match
        for statute, confidence in parent.statute_matches.items():
            prompt = f"""
            Evaluate the confidence for applying {statute} to this case.
            
            Statute definition: {self.statutes.get(statute, "Unknown")}
            
            Case facts: {parent.extracted_facts}
            Legal elements: {parent.identified_elements.get(statute, [])}
            Initial confidence: {confidence}
            
            Provide:
            1. Final confidence score (0.0-1.0)
            2. Key strengths of the match
            3. Potential weaknesses
            4. Recommendation (APPLY/REJECT/APPLY with caution)
            
            Format:
            CONFIDENCE: [score]
            STRENGTHS: [strength1, strength2, ...]
            WEAKNESSES: [weakness1, weakness2, ...]
            RECOMMENDATION: [APPLY/REJECT/APPLY with caution]
            """
            
            response = self._call_claude(prompt, temperature=0.1)
            
            # Parse response
            final_confidence, strengths, weaknesses, recommendation = self._parse_confidence_evaluation(response)
            
            child = SearchState(
                depth=parent.depth + 1,
                thought_content=f"Confidence evaluation for {statute}: {recommendation}",
                thought_type=ThoughtType.CONFIDENCE_EVALUATION,
                confidence_score=final_confidence,
                extracted_facts=parent.extracted_facts,
                identified_elements=parent.identified_elements,
                statute_matches={statute: final_confidence},
                supporting_evidence=parent.supporting_evidence + strengths
            )
            children.append(child)
        
        return children
    
    def _generate_final_decision_states(self, parent: SearchState, case_text: str) -> List[SearchState]:
        """Generate final decision (terminal) states"""
        children = []
        
        prompt = f"""
        Make the final decision on statute applicability.
        
        Path taken: {' -> '.join(parent.path_history + [parent.thought_content])}
        
        Statute being evaluated: {list(parent.statute_matches.keys())}
        Confidence scores: {parent.statute_matches}
        Supporting evidence: {parent.supporting_evidence}
        
        Provide final decision:
        1. List of applicable statutes (if confidence >= 0.5)
        2. Overall confidence in decision
        3. Final reasoning
        
        Format:
        APPLICABLE_STATUTES: [statute1, statute2, ...]
        OVERALL_CONFIDENCE: [score]
        REASONING: [explanation]
        """
        
        response = self._call_claude(prompt, temperature=0.1)
        
        # Parse response
        applicable_statutes, overall_confidence, reasoning = self._parse_final_decision(response)
        
        final_matches = {statute: parent.statute_matches.get(statute, 0.0) 
                        for statute in applicable_statutes}
        
        child = SearchState(
            depth=parent.depth + 1,
            thought_content=f"Final decision: {applicable_statutes}",
            thought_type=ThoughtType.FINAL_DECISION,
            confidence_score=overall_confidence,
            extracted_facts=parent.extracted_facts,
            identified_elements=parent.identified_elements,
            statute_matches=final_matches,
            supporting_evidence=parent.supporting_evidence + [reasoning],
            is_terminal=True
        )
        
        children.append(child)
        return children
    
    def breadth_first_search(self, case_text: str, verbose: bool = False) -> List[str]:
        """
        Perform breadth-first search to find the best statute predictions
        """
        # Initialize search
        self.best_solution = None
        self.best_score = 0.0
        self.all_solutions = []
        self.backup_solutions = []  # NEW
        self.nodes_explored = 0
        self.levels_explored = 0
        self.search_queue = deque()
        
        # Create root state
        root_state = SearchState(
            depth=0,
            thought_content="Starting legal analysis",
            thought_type=ThoughtType.ROOT,
            confidence_score=1.0
        )
        
        # Initialize queue with root
        self.search_queue.append(root_state)
        
        # Start BFS
        self._bfs_iterative(case_text, verbose)
        
        if verbose:
            print(f"\nSEARCH STATISTICS:")
            print(f"Nodes explored: {self.nodes_explored}")
            print(f"Levels explored: {self.levels_explored}")
            print(f"Terminal solutions found: {len(self.all_solutions)}")
            print(f"Backup solutions found: {len(self.backup_solutions)}")
            print(f"Best score found: {self.best_score:.3f}")
        
        # Return best solution or consensus from multiple solutions
        return self._get_best_prediction()
    
    def _bfs_iterative(self, case_text: str, verbose: bool = False):
        """
        Iterative BFS implementation using a queue
        """
        current_level = 0
        
        while self.search_queue:
            # Process all nodes at current level
            level_size = len(self.search_queue)
            nodes_at_level = 0
            
            if verbose:
                print(f"\n{'='*50}")
                print(f"EXPLORING LEVEL {current_level} ({level_size} nodes)")
                print(f"{'='*50}")
            
            # Process nodes level by level
            for _ in range(min(level_size, self.max_nodes_per_level)):
                if not self.search_queue:
                    break
                    
                current_state = self.search_queue.popleft()
                nodes_at_level += 1
                self.nodes_explored += 1
                
                if verbose:
                    print(f"Node {nodes_at_level}: {current_state.thought_content} "
                          f"(Score: {self.evaluate_state(current_state):.3f}, "
                          f"Confidence: {current_state.confidence_score:.3f})")
                
                # NEW: Check if this is a terminal state using enhanced logic
                if self.is_terminal_node(current_state):
                    self._process_terminal_state(current_state, verbose)
                    continue
                
                # NEW: Collect high-quality non-terminal states as backup solutions
                if (current_state.statute_matches and 
                    current_state.confidence_score >= 0.6 and 
                    self.evaluate_state(current_state) >= 0.7):
                    self.backup_solutions.append(current_state)
                    if verbose:
                        print(f"  Added as backup solution")
                
                # Check if we should prune this branch
                if self.should_prune(current_state):
                    if verbose:
                        print(f"  [PRUNED - Low confidence or max depth reached]")
                    continue
                
                # Generate and enqueue children
                children = self.generate_children_states(current_state, case_text)
                
                # Add children to queue for next level exploration
                for child in children:
                    if not self.should_prune(child):
                        self.search_queue.append(child)
                
                if verbose and children:
                    print(f"  Generated {len(children)} children for next level")
            
            # Clear remaining nodes at this level if we hit the limit
            remaining_at_level = level_size - nodes_at_level
            for _ in range(remaining_at_level):
                if self.search_queue:
                    pruned_state = self.search_queue.popleft()
                    if verbose:
                        print(f"  [LEVEL LIMIT REACHED - Pruning: {pruned_state.thought_content}]")
            
            current_level += 1
            self.levels_explored = current_level
            
            # Early termination conditions
            if current_level >= self.max_depth:
                if verbose:
                    print(f"\nMax depth ({self.max_depth}) reached. Terminating search.")
                break
            
            # If we have found good solutions and queue is getting large, we can stop
            if len(self.all_solutions) >= 3 and len(self.search_queue) > 20:
                if verbose:
                    print(f"\nFound sufficient solutions ({len(self.all_solutions)}). Terminating search.")
                break
    
    def _process_terminal_state(self, state: SearchState, verbose: bool = False):
        """Process a terminal state and update best solutions"""
        score = self.evaluate_state(state)
        self.all_solutions.append(state)
        
        if score > self.best_score:
            self.best_score = score
            self.best_solution = copy.deepcopy(state)
            
            if verbose:
                print(f"  *** NEW BEST SOLUTION: {score:.3f} ***")
                print(f"  Statutes: {list(state.statute_matches.keys())}")
        elif verbose:
            print(f"  Terminal state score: {score:.3f}")
    
    def _get_best_prediction(self) -> List[str]:
        """
        Enhanced method to get the best prediction from all found solutions
        """
        # Strategy 1: Use the single best terminal solution
        if self.all_solutions and self.best_solution and self.best_solution.statute_matches:
            return list(self.best_solution.statute_matches.keys())
        
        # Strategy 2: If no terminal solutions, use best backup solutions
        if not self.all_solutions and self.backup_solutions:
            # Sort backup solutions by quality score
            sorted_backups = sorted(self.backup_solutions, 
                                  key=lambda x: self.evaluate_state(x), 
                                  reverse=True)
            
            best_backup = sorted_backups[0]
            if best_backup.statute_matches:
                # Filter statutes with confidence >= 0.5
                applicable_statutes = [statute for statute, conf in best_backup.statute_matches.items() 
                                     if conf >= 0.5]
                return applicable_statutes
        
        # Strategy 3: Consensus approach from multiple solutions
        all_candidate_solutions = self.all_solutions + self.backup_solutions
        if len(all_candidate_solutions) > 1:
            return self._get_consensus_prediction(all_candidate_solutions)
        
        # Strategy 4: If nothing found, try relaxed search from queue
        if not self.all_solutions and not self.backup_solutions and self.search_queue:
            return self._extract_from_remaining_queue()
        
        return []
    
    def _get_consensus_prediction(self, solutions: List[SearchState]) -> List[str]:
        """
        Get consensus prediction from multiple solutions
        """
        statute_votes = {}
        statute_confidences = {}
        
        # Count votes and average confidences for each statute
        for solution in solutions:
            if solution.statute_matches:
                for statute, confidence in solution.statute_matches.items():
                    if confidence >= 0.5:  # Only count confident matches
                        statute_votes[statute] = statute_votes.get(statute, 0) + 1
                        if statute not in statute_confidences:
                            statute_confidences[statute] = []
                        statute_confidences[statute].append(confidence)
        
        # Calculate average confidence for each statute
        statute_avg_conf = {statute: sum(confs)/len(confs) 
                           for statute, confs in statute_confidences.items()}
        
        # Return statutes that appear in at least 30% of solutions or have high confidence
        min_votes = max(1, len(solutions) * 0.3)
        consensus_statutes = []
        
        for statute, votes in statute_votes.items():
            avg_conf = statute_avg_conf.get(statute, 0)
            if votes >= min_votes or avg_conf >= 0.75:
                consensus_statutes.append(statute)
        
        return consensus_statutes
    
    def _extract_from_remaining_queue(self) -> List[str]:
        """
        NEW: Extract potential solutions from remaining queue nodes
        """
        remaining_nodes = list(self.search_queue)
        potential_solutions = []
        
        for node in remaining_nodes:
            if (node.statute_matches and 
                node.confidence_score >= 0.6 and 
                any(conf >= 0.5 for conf in node.statute_matches.values())):
                potential_solutions.append(node)
        
        if potential_solutions:
            # Sort by evaluation score and take the best
            best_remaining = max(potential_solutions, key=lambda x: self.evaluate_state(x))
            applicable_statutes = [statute for statute, conf in best_remaining.statute_matches.items() 
                                 if conf >= 0.5]
            return applicable_statutes
        
        return []
    
    # Helper parsing methods (same as before)
    def _parse_fact_response(self, response: str) -> Tuple[List[str], float, List[str]]:
        """Parse fact extraction response"""
        facts, confidence, evidence = [], 0.5, []
        
        for line in response.split('\n'):
            if line.startswith('FACTS:'):
                facts_str = line.replace('FACTS:', '').strip()
                try:
                    facts = eval(facts_str) if facts_str.startswith('[') else facts_str.split(', ')
                except:
                    facts = [facts_str]
            elif line.startswith('CONFIDENCE:'):
                try:
                    confidence = float(line.replace('CONFIDENCE:', '').strip())
                except:
                    confidence = 0.5
            elif line.startswith('EVIDENCE:'):
                evidence_str = line.replace('EVIDENCE:', '').strip()
                try:
                    evidence = eval(evidence_str) if evidence_str.startswith('[') else evidence_str.split(', ')
                except:
                    evidence = [evidence_str]
        
        return facts, confidence, evidence
    
    def _parse_elements_response(self, response: str) -> Tuple[Dict[str, List[str]], float]:
        """Parse legal elements response"""
        elements_dict = {}
        confidences = []
        
        sections = response.split('---')
        for section in sections:
            statute, elements, confidence = "", [], 0.0
            
            for line in section.split('\n'):
                if line.startswith('STATUTE:'):
                    statute = line.replace('STATUTE:', '').strip()
                elif line.startswith('ELEMENTS:'):
                    elements_str = line.replace('ELEMENTS:', '').strip()
                    try:
                        elements = eval(elements_str) if elements_str.startswith('[') else elements_str.split(', ')
                    except:
                        elements = [elements_str]
                elif line.startswith('CONFIDENCE:'):
                    try:
                        confidence = float(line.replace('CONFIDENCE:', '').strip())
                    except:
                        confidence = 0.5
            
            if statute and elements:
                elements_dict[statute] = elements
                confidences.append(confidence)
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        return elements_dict, avg_confidence
    
    def _parse_statute_matches(self, response: str) -> Tuple[Dict[str, float], List[str]]:
        """Parse statute matching response"""
        matches = {}
        reasoning = []
        
        sections = response.split('---')
        for section in sections:
            statute, confidence, reason = "", 0.0, ""
            
            for line in section.split('\n'):
                if line.startswith('STATUTE:'):
                    statute = line.replace('STATUTE:', '').strip()
                elif line.startswith('CONFIDENCE:'):
                    try:
                        confidence = float(line.replace('CONFIDENCE:', '').strip())
                    except:
                        confidence = 0.5
                elif line.startswith('REASONING:'):
                    reason = line.replace('REASONING:', '').strip()
            
            if statute and confidence > 0:
                matches[statute] = confidence
                if reason:
                    reasoning.append(reason)
        
        return matches, reasoning
    
    def _parse_confidence_evaluation(self, response: str) -> Tuple[float, List[str], List[str], str]:
        """Parse confidence evaluation response"""
        confidence, strengths, weaknesses, recommendation = 0.5, [], [], "REJECT"
        
        for line in response.split('\n'):
            if line.startswith('CONFIDENCE:'):
                try:
                    confidence = float(line.replace('CONFIDENCE:', '').strip())
                except:
                    confidence = 0.5
            elif line.startswith('STRENGTHS:'):
                strengths_str = line.replace('STRENGTHS:', '').strip()
                try:
                    strengths = eval(strengths_str) if strengths_str.startswith('[') else strengths_str.split(', ')
                except:
                    strengths = [strengths_str]
            elif line.startswith('WEAKNESSES:'):
                weaknesses_str = line.replace('WEAKNESSES:', '').strip()
                try:
                    weaknesses = eval(weaknesses_str) if weaknesses_str.startswith('[') else weaknesses_str.split(', ')
                except:
                    weaknesses = [weaknesses_str]
            elif line.startswith('RECOMMENDATION:'):
                recommendation = line.replace('RECOMMENDATION:', '').strip()
        
        return confidence, strengths, weaknesses, recommendation
    
    def _parse_final_decision(self, response: str) -> Tuple[List[str], float, str]:
        """Parse final decision response"""
        statutes, confidence, reasoning = [], 0.5, ""
        
        for line in response.split('\n'):
            if line.startswith('APPLICABLE_STATUTES:'):
                statutes_str = line.replace('APPLICABLE_STATUTES:', '').strip()
                try:
                    statutes = eval(statutes_str) if statutes_str.startswith('[') else statutes_str.split(', ')
                except:
                    statutes = [statutes_str] if statutes_str else []
            elif line.startswith('OVERALL_CONFIDENCE:'):
                try:
                    confidence = float(line.replace('OVERALL_CONFIDENCE:', '').strip())
                except:
                    confidence = 0.5
            elif line.startswith('REASONING:'):
                reasoning = line.replace('REASONING:', '').strip()
        
        return statutes, confidence, reasoning
    
    def predict_statutes(self, case_text: str, confidence_threshold: float = 0.4, verbose: bool = False) -> List[str]:
        """Main method to predict statutes using TOT-BFS approach"""
        if verbose:
            print("TREE-OF-THOUGHTS BREADTH-FIRST SEARCH LEGAL PREDICTOR")
            print("=" * 60)
            print(f"Model: {self.model}")
            print(f"Max Depth: {self.max_depth}")
            print(f"Confidence Threshold: {confidence_threshold}")
            print(f"Max Children per Node: {self.max_children}")
            print(f"Max Nodes per Level: {self.max_nodes_per_level}")
            print("=" * 60)
        
        self.confidence_threshold = confidence_threshold
        return self.breadth_first_search(case_text, verbose)

# Example usage function
def doc_prediction(gr_path):
    json_2 = open(gr_path)
    jgr = json.load(json_2)
    docs = list(jgr.keys())
    fact_list = {}
    for doc in docs:
        fact_list[doc]=jgr[doc]['fact']
    return fact_list

def main():
    prediction_section={}
    # Initialize predictor
    predictor = TOTBFSLegalPredictor(
        api_key="",  # Replace with your API key
        model="claude-sonnet-4-20250514"
    )
    
    # Load test cases
    sections = ['147', '420', '201', '506', '302', '376', '498A']
    for section in sections:
        print(section)
        prediction_doc={}
        dir_2 = f"/home/subinay/Documents/data/statute_prediction/statute_dataset/statute_dataset-main/Test_doc/test_collection_IPC_{section}.json"
        test_cases=doc_prediction(dir_2)
        list_docs=test_cases.keys()
        for i, doc in enumerate(list_docs):
            print(doc)
            case=test_cases[doc]
            print(f"\n{'='*80}")
            print(f"TEST CASE {i}:")
            print(f"{'='*80}")
            print(f"Case: {case.strip()}")
            print()
        
            try:
                predicted_statutes = predictor.predict_statutes(case, verbose=True)
                print(f"\nFINAL RESULT: {predicted_statutes}")
                prediction_doc[doc]=predicted_statutes
            except Exception as e:
                print(f"Error processing case: {e}")
        
            print('='*80)
        prediction_section[section]=prediction_doc
    with open("predicted_statutes_bfs_claude_3.json","w") as f1:
        json.dump(prediction_section,f1)
main()


