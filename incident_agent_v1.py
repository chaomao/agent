"""
Incident Response AI Agent - Day 1 Implementation
This implements the RAG pattern discussed: Retrieval-Augmented Generation
- Brain 1: Memory (Vector Database with ChromaDB)
- Brain 2: Reasoning (LLM with LangChain)
- Brain 3: Observation (Context Analysis)
"""

import os
import json
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime

# LangChain imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Vector DB
import chromadb
from chromadb.config import Settings


@dataclass
class IncidentReport:
    """Structured incident report"""
    id: str
    title: str
    severity: str
    symptoms: str
    investigation_steps: List[Dict[str, str]]
    root_cause: str
    resolution: str
    timestamp: str
    metadata: Dict[str, str]
    
    def to_text(self) -> str:
        """Convert incident to searchable text"""
        steps_text = "\n".join([
            f"[{step['timestamp']}] {step['action']}: {step['finding']}"
            for step in self.investigation_steps
        ])
        
        return f"""
INCIDENT: {self.title}
SEVERITY: {self.severity}
DATE: {self.timestamp}

SYMPTOMS:
{self.symptoms}

INVESTIGATION:
{steps_text}

ROOT CAUSE:
{self.root_cause}

RESOLUTION:
{self.resolution}
"""


class IncidentDatabase:
    """Brain 1: Memory - Manages historical incidents with vector search"""
    
    def __init__(self, collection_name: str = "incidents"):
        # Initialize ChromaDB (local vector database)
        self.client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            allow_reset=True
        ))
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        # Initialize embeddings (convert text to vectors)
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"  # Fast and good quality
        )
    
    def add_incident(self, incident: IncidentReport):
        """Add incident to vector database"""
        text = incident.to_text()
        
        # Generate embedding
        embedding = self.embeddings.embed_query(text)
        
        # Store in ChromaDB
        self.collection.add(
            ids=[incident.id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[{
                "title": incident.title,
                "severity": incident.severity,
                "root_cause": incident.root_cause,
                "timestamp": incident.timestamp
            }]
        )
        
        print(f"✓ Added incident {incident.id} to database")
    
    def search_similar(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for similar incidents using vector similarity"""
        # Convert query to vector
        query_embedding = self.embeddings.embed_query(query)
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Format results
        similar_incidents = []
        if results['ids']:
            for i in range(len(results['ids'][0])):
                similar_incidents.append({
                    'id': results['ids'][0][i],
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity_score': 1 - results['distances'][0][i]  # Convert distance to similarity
                })
        
        return similar_incidents
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        count = self.collection.count()
        return {
            "total_incidents": count,
            "collection_name": self.collection.name
        }


class IncidentAnalyzer:
    """Brain 2: Reasoning - Uses LLM to analyze and suggest"""
    
    def __init__(self, model: str = "claude-3-5-sonnet-20241022"):
        """
        Initialize with LLM
        Using Claude for best reasoning capability (as discussed)
        """
        self.llm = ChatAnthropic(
            model=model,
            temperature=0.3,  # Low temperature for consistent, focused responses
            max_tokens=4096
        )
        
        # Prompt template for incident analysis
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert DevOps engineer helping to troubleshoot production incidents.

Your role:
- Analyze incident symptoms and logs
- Reference similar historical incidents
- Suggest specific, actionable next steps
- Explain your reasoning clearly

Key principles:
1. Always reference similar incidents when available
2. Provide specific commands or checks, not vague advice
3. Rank suggestions by likelihood based on patterns
4. Consider safety - warn about risky actions
5. Be concise but thorough"""),
            
            ("user", """Current Incident:
{current_incident}

Similar Historical Incidents:
{similar_incidents}

Task: Analyze this incident and provide:
1. SYMPTOM ANALYSIS: What the symptoms tell us
2. POTENTIAL ROOT CAUSES: Ranked by likelihood (reference similar incidents)
3. RECOMMENDED NEXT STEPS: Specific actions to take (with commands if applicable)
4. SAFETY CONSIDERATIONS: Any risks to be aware of

Be specific and actionable. Reference incident IDs when drawing from historical data.""")
        ])
        
        # Prompt for hypothesis generation
        self.hypothesis_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at root cause analysis for production incidents."),
            ("user", """Based on these symptoms and investigation progress:

{context}

Generate 3-5 potential root cause hypotheses, ranked by likelihood.
For each hypothesis:
- Probability estimate (%)
- Supporting evidence
- How to verify
- Similar incidents (if any)

Format as a structured list.""")
        ])
    
    def analyze_incident(
        self, 
        current_incident: str, 
        similar_incidents: List[Dict]
    ) -> str:
        """
        Core RAG function: Retrieves similar incidents, LLM reasons about them
        """
        # Format similar incidents for LLM context
        similar_context = self._format_similar_incidents(similar_incidents)
        
        # Create analysis chain
        chain = (
            self.analysis_prompt 
            | self.llm 
            | StrOutputParser()
        )
        
        # Run analysis
        response = chain.invoke({
            "current_incident": current_incident,
            "similar_incidents": similar_context
        })
        
        return response
    
    def generate_hypotheses(self, context: str) -> str:
        """Generate root cause hypotheses"""
        chain = (
            self.hypothesis_prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain.invoke({"context": context})
    
    def _format_similar_incidents(self, incidents: List[Dict]) -> str:
        """Format similar incidents for LLM context"""
        if not incidents:
            return "No similar incidents found in database."
        
        formatted = []
        for idx, inc in enumerate(incidents, 1):
            similarity_pct = inc['similarity_score'] * 100
            formatted.append(
                f"--- Similar Incident #{idx} (Similarity: {similarity_pct:.1f}%) ---\n"
                f"ID: {inc['id']}\n"
                f"Title: {inc['metadata']['title']}\n"
                f"Root Cause: {inc['metadata']['root_cause']}\n\n"
                f"Details:\n{inc['document'][:800]}...\n"  # Truncate for context limits
            )
        
        return "\n\n".join(formatted)


class ContextAnalyzer:
    """Brain 3: Observation - Analyzes current incident context"""
    
    @staticmethod
    def extract_key_info(incident_text: str) -> Dict:
        """Extract structured information from incident text"""
        info = {
            "has_error_messages": False,
            "has_timestamps": False,
            "has_commands": False,
            "severity_indicators": [],
            "service_mentions": []
        }
        
        lines = incident_text.lower().split('\n')
        
        # Detect error patterns
        error_keywords = ['error', 'failed', 'exception', 'timeout', 'refused']
        if any(keyword in incident_text.lower() for keyword in error_keywords):
            info["has_error_messages"] = True
        
        # Detect timestamps (basic pattern)
        if any(char.isdigit() and ':' in line for line in lines):
            info["has_timestamps"] = True
        
        # Detect commands
        command_indicators = ['kubectl', 'docker', 'curl', 'grep', 'tail', '$', '#']
        if any(indicator in incident_text for indicator in command_indicators):
            info["has_commands"] = True
        
        # Severity indicators
        if any(word in incident_text.lower() for word in ['critical', 'down', 'outage']):
            info["severity_indicators"].append("HIGH")
        if any(word in incident_text.lower() for word in ['degraded', 'slow', 'intermittent']):
            info["severity_indicators"].append("MEDIUM")
        
        # Service mentions
        services = ['api', 'database', 'auth', 'login', 'payment', 'cache', 'redis', 'postgres']
        for service in services:
            if service in incident_text.lower():
                info["service_mentions"].append(service)
        
        return info


class IncidentAgent:
    """Main AI Agent - Orchestrates all components"""
    
    def __init__(self):
        print("🤖 Initializing Incident Response AI Agent...")
        
        # Initialize all three brains
        self.database = IncidentDatabase()
        self.analyzer = IncidentAnalyzer()
        self.context = ContextAnalyzer()
        
        print("✓ Agent initialized successfully\n")
    
    def load_sample_data(self):
        """Load sample historical incidents (for demo)"""
        print("📚 Loading sample historical incidents...")
        
        # Sample incident 1: JWT Authentication Issue
        incident1 = IncidentReport(
            id="INC-2024-1247",
            title="Mobile App Login Failure - JWT Validation Error",
            severity="P1-Critical",
            symptoms="Mobile users unable to login. Error: 'Authentication failed'. Web login works fine. 85% failure rate.",
            investigation_steps=[
                {
                    "timestamp": "14:23",
                    "action": "Checked API logs",
                    "finding": "JWT validation failed - token signature invalid errors"
                },
                {
                    "timestamp": "14:35",
                    "action": "Reviewed recent deployments",
                    "finding": "Mobile API v3.2.1 deployed at 14:10 UTC with JWT secret rotation"
                },
                {
                    "timestamp": "14:42",
                    "action": "Compared configurations",
                    "finding": "Mobile clients using JWT_SECRET_V2, server validating with V3"
                }
            ],
            root_cause="Configuration mismatch: JWT secret rotation updated server config but mobile clients still using old secret version",
            resolution="Rolled back mobile-api-service to v3.2.0. Login success rate returned to 99.8%. Created ticket for proper secret rotation procedure.",
            timestamp="2024-10-15T14:23:00Z",
            metadata={
                "affected_users": "6847",
                "mttr_minutes": "165",
                "service": "authentication"
            }
        )
        
        # Sample incident 2: Database Connection Pool Exhaustion
        incident2 = IncidentReport(
            id="INC-2024-0892",
            title="API Timeouts - Database Connection Pool Exhausted",
            severity="P1-Critical",
            symptoms="API endpoints returning 504 timeouts. Response times > 30s. Started after traffic spike.",
            investigation_steps=[
                {
                    "timestamp": "09:15",
                    "action": "Checked API metrics",
                    "finding": "Request latency p99 > 30s, normally < 200ms"
                },
                {
                    "timestamp": "09:22",
                    "action": "Checked database connections",
                    "finding": "Connection pool at 100/100 (max), 500+ queued requests"
                },
                {
                    "timestamp": "09:28",
                    "action": "Analyzed slow queries",
                    "finding": "Multiple long-running queries (>10s) blocking connections"
                }
            ],
            root_cause="Connection pool max size (100) too small for current traffic. Long-running queries holding connections.",
            resolution="Increased connection pool to 200. Added query timeout (5s). Optimized slow queries. Added connection pool monitoring.",
            timestamp="2024-09-22T09:15:00Z",
            metadata={
                "affected_requests": "12500",
                "mttr_minutes": "85",
                "service": "database"
            }
        )
        
        # Sample incident 3: Memory Leak in Node Service
        incident3 = IncidentReport(
            id="INC-2024-0445",
            title="Node.js Service Crashing - Memory Leak",
            severity="P2-High",
            symptoms="Payment service pods restarting every 2-3 hours. OOMKilled errors in logs.",
            investigation_steps=[
                {
                    "timestamp": "16:30",
                    "action": "Checked pod status",
                    "finding": "Pods restarting with reason: OOMKilled. Memory usage pattern shows steady increase"
                },
                {
                    "timestamp": "16:45",
                    "action": "Analyzed heap dumps",
                    "finding": "Heap size growing from 200MB to 2GB over 2 hours"
                },
                {
                    "timestamp": "17:00",
                    "action": "Reviewed recent code changes",
                    "finding": "New caching layer added, no TTL or size limits"
                }
            ],
            root_cause="In-memory cache with no eviction policy. Cache growing unbounded causing memory leak.",
            resolution="Added TTL (1 hour) and max size (10000 items) to cache. Deployed fix. Memory usage stable at ~300MB.",
            timestamp="2024-08-10T16:30:00Z",
            metadata={
                "affected_transactions": "~2000",
                "mttr_minutes": "120",
                "service": "payment"
            }
        )
        
        # Add to database
        for incident in [incident1, incident2, incident3]:
            self.database.add_incident(incident)
        
        print(f"✓ Loaded {len([incident1, incident2, incident3])} sample incidents\n")
    
    def process_incident(self, incident_input: str) -> Dict:
        """
        Main function: Process new incident and provide analysis
        This is what devs will call during incidents
        """
        print("=" * 60)
        print("🔍 PROCESSING NEW INCIDENT")
        print("=" * 60)
        
        # Step 1: Analyze context (Brain 3)
        print("\n[Step 1] Analyzing incident context...")
        context_info = self.context.extract_key_info(incident_input)
        print(f"  - Has error messages: {context_info['has_error_messages']}")
        print(f"  - Services mentioned: {context_info['service_mentions']}")
        print(f"  - Severity indicators: {context_info['severity_indicators']}")
        
        # Step 2: Search for similar incidents (Brain 1)
        print("\n[Step 2] Searching for similar historical incidents...")
        similar_incidents = self.database.search_similar(incident_input, top_k=3)
        
        if similar_incidents:
            print(f"  ✓ Found {len(similar_incidents)} similar incidents:")
            for inc in similar_incidents:
                similarity_pct = inc['similarity_score'] * 100
                print(f"    • {inc['id']}: {inc['metadata']['title']} (similarity: {similarity_pct:.1f}%)")
        else:
            print("  ⚠ No similar incidents found (database may be empty)")
        
        # Step 3: Generate analysis (Brain 2)
        print("\n[Step 3] Generating AI analysis...")
        analysis = self.analyzer.analyze_incident(incident_input, similar_incidents)
        
        # Step 4: Compile response
        response = {
            "timestamp": datetime.now().isoformat(),
            "context_analysis": context_info,
            "similar_incidents_found": len(similar_incidents),
            "similar_incidents": [
                {
                    "id": inc['id'],
                    "title": inc['metadata']['title'],
                    "similarity": f"{inc['similarity_score']*100:.1f}%"
                }
                for inc in similar_incidents
            ],
            "ai_analysis": analysis
        }
        
        print("\n✓ Analysis complete!\n")
        
        return response
    
    def display_response(self, response: Dict):
        """Pretty print the agent's response"""
        print("=" * 60)
        print("🤖 AI AGENT RESPONSE")
        print("=" * 60)
        
        print(f"\n⏰ Timestamp: {response['timestamp']}")
        
        print(f"\n📊 Context Analysis:")
        context = response['context_analysis']
        if context['service_mentions']:
            print(f"  • Services: {', '.join(context['service_mentions'])}")
        if context['severity_indicators']:
            print(f"  • Severity: {', '.join(context['severity_indicators'])}")
        
        if response['similar_incidents_found'] > 0:
            print(f"\n🔗 Similar Historical Incidents ({response['similar_incidents_found']} found):")
            for inc in response['similar_incidents']:
                print(f"  • [{inc['id']}] {inc['title']}")
                print(f"    Similarity: {inc['similarity']}")
        else:
            print("\n⚠️  No similar incidents found in database")
        
        print("\n" + "=" * 60)
        print("💡 DETAILED ANALYSIS")
        print("=" * 60)
        print(response['ai_analysis'])
        print("\n" + "=" * 60)
    
    def get_stats(self):
        """Get agent statistics"""
        return self.database.get_stats()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def main():
    """Demo the agent with a sample incident"""
    
    # Set API keys (you need to set these in your environment)
    # os.environ["OPENAI_API_KEY"] = "your-key-here"
    # os.environ["ANTHROPIC_API_KEY"] = "your-key-here"
    
    print("\n" + "=" * 60)
    print("  INCIDENT RESPONSE AI AGENT - DAY 1 IMPLEMENTATION")
    print("=" * 60 + "\n")
    
    # Initialize agent
    agent = IncidentAgent()
    
    # Load sample historical data
    agent.load_sample_data()
    
    # Sample new incident (dev pastes this)
    new_incident = """
PRODUCTION INCIDENT - Login Issues

Time: 2024-10-28 15:45 UTC
Severity: Critical

SYMPTOMS:
- Users reporting "Authentication failed" errors on mobile app
- Login button shows loading spinner then error message
- Approximately 70% of login attempts failing
- Web application login works fine
- Issue started around 15:30 UTC

ERROR LOGS:
2024-10-28 15:31:45 ERROR [auth-service] JWT validation failed: signature verification failed
2024-10-28 15:31:46 ERROR [auth-service] JWT validation failed: signature verification failed
2024-10-28 15:31:47 ERROR [auth-service] JWT validation failed: signature verification failed

RECENT CHANGES:
- Mobile API deployment v4.1.0 at 15:25 UTC
- Updated authentication middleware
- No database migrations

METRICS:
- Error rate: 68% (normal: <1%)
- Affected users: ~3,000 in 15 minutes
- API response time: normal (200ms)

QUESTION: What should I check first? Is this related to the deployment?
"""
    
    print("\n" + "=" * 60)
    print("📝 NEW INCIDENT SUBMITTED")
    print("=" * 60)
    print(new_incident)
    
    # Process the incident
    response = agent.process_incident(new_incident)
    
    # Display the agent's response
    agent.display_response(response)
    
    # Show stats
    print("\n📊 Agent Statistics:")
    stats = agent.get_stats()
    print(f"  • Total incidents in database: {stats['total_incidents']}")
    print(f"  • Collection: {stats['collection_name']}")
    print("\n✅ Demo complete!\n")


if __name__ == "__main__":
    main()