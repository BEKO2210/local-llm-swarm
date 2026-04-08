'use client';

import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Send, Loader2, Brain, ChevronDown, ChevronUp, Zap } from 'lucide-react';
import MarkdownRenderer from './MarkdownRenderer';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  isStreaming?: boolean;
}

interface ThinkingStep {
  agent: string;
  status: 'pending' | 'active' | 'completed';
  timestamp: Date;
}

interface ChatInterfaceProps {
  conversationId?: string;
  onConversationCreated?: (id: string) => void;
}

const API_BASE = 'http://127.0.0.1:8000';

export default function ChatInterface({ 
  conversationId,
  onConversationCreated 
}: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [deepThinking, setDeepThinking] = useState(false);
  const [showThinking, setShowThinking] = useState(true);
  const [thinkingSteps, setThinkingSteps] = useState<ThinkingStep[]>([]);
  const [currentAgent, setCurrentAgent] = useState<string>('');
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  
  // Use ref for streaming content to avoid closure staleness
  const streamingContentRef = useRef('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = textareaRef.current.scrollHeight + 'px';
    }
  }, [input]);

  // Load conversation history when conversationId changes
  useEffect(() => {
    if (conversationId) {
      loadConversationHistory(conversationId);
    } else {
      setMessages([]);
    }
  }, [conversationId]);

  const loadConversationHistory = async (convId: string) => {
    try {
      setIsLoadingHistory(true);
      const response = await fetch(`${API_BASE}/api/chat/conversations/${convId}`);
      if (!response.ok) {
        throw new Error('Failed to load conversation');
      }
      const data = await response.json();
      
      if (data.messages && Array.isArray(data.messages)) {
        const loadedMessages: Message[] = data.messages.map((msg: any) => ({
          id: msg.id || Date.now().toString() + Math.random(),
          role: msg.role,
          content: msg.content,
          isStreaming: false,
        }));
        setMessages(loadedMessages);
      }
    } catch (error) {
      console.error('Error loading conversation:', error);
    } finally {
      setIsLoadingHistory(false);
    }
  };

  const initializeThinkingSteps = (isDeep: boolean) => {
    const steps: ThinkingStep[] = [
      { agent: 'Planner', status: 'pending', timestamp: new Date() },
      { agent: 'Executor', status: 'pending', timestamp: new Date() },
      { agent: 'Critic', status: 'pending', timestamp: new Date() },
    ];
    if (isDeep) {
      steps.push({ agent: 'Heavy Critic (Gemma 26B)', status: 'pending', timestamp: new Date() });
    }
    setThinkingSteps(steps);
  };

  const updateThinkingStep = useCallback((agent: string, status: ThinkingStep['status']) => {
    setThinkingSteps(prev => 
      prev.map(step => 
        step.agent === agent ? { ...step, status, timestamp: new Date() } : step
      )
    );
    setCurrentAgent(status === 'active' ? agent : '');
  }, []);

  const handleSubmit = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input.trim(),
    };

    // Add user message
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);
    initializeThinkingSteps(deepThinking);

    // Create assistant message placeholder
    const assistantMessageId = (Date.now() + 1).toString();
    streamingContentRef.current = '';
    
    setMessages(prev => [...prev, {
      id: assistantMessageId,
      role: 'assistant',
      content: '',
      isStreaming: true,
    }]);

    abortControllerRef.current = new AbortController();
    let receivedConversationId = conversationId;

    try {
      console.log('[Chat] Sending request:', { prompt: userMessage.content, conversation_id: conversationId });
      
      const response = await fetch(`${API_BASE}/api/chat/swarm`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: userMessage.content,
          conversation_id: conversationId || undefined,
          deep_thinking: deepThinking,
          temperature: 0.7,
        }),
        signal: abortControllerRef.current.signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      console.log('[Chat] Response received, starting stream read');

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('No response body');
      }

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        
        if (done) {
          console.log('[Chat] Stream complete');
          break;
        }

        // Decode chunk and add to buffer
        buffer += decoder.decode(value, { stream: true });
        
        // Process complete lines from buffer
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // Keep incomplete line in buffer

        for (const line of lines) {
          const trimmedLine = line.trim();
          if (!trimmedLine) continue;

          // SSE format: "data: <content>"
          if (trimmedLine.startsWith('data:')) {
            const data = trimmedLine.slice(5).trim();
            
            if (!data) continue;

            console.log('[Chat] SSE data received:', data.substring(0, 50));

            // Handle conversation ID
            if (data.startsWith('[CONVERSATION_ID:')) {
              const convId = data.slice(17, -1);
              receivedConversationId = convId;
              console.log('[Chat] Received conversation ID:', convId);
              if (onConversationCreated && !conversationId) {
                onConversationCreated(convId);
              }
              continue;
            }

            // Handle completion
            if (data === '[DONE]') {
              console.log('[Chat] Stream done');
              setIsLoading(false);
              setCurrentAgent('');
              setThinkingSteps(prev => 
                prev.map(step => ({ ...step, status: 'completed' }))
              );
              continue;
            }

            // Handle errors
            if (data.startsWith('[ERROR]')) {
              console.error('[Chat] Stream error:', data);
              setMessages(prev => 
                prev.map(msg => 
                  msg.id === assistantMessageId 
                    ? { ...msg, content: 'Error: ' + data.slice(7), isStreaming: false }
                    : msg
                )
              );
              continue;
            }

            // Unescape content
            const content = data
              .replace(/\\n/g, '\n')
              .replace(/\\r/g, '\r')
              .replace(/\\t/g, '\t');

            // Update streaming content
            streamingContentRef.current += content;
            
            // Update message in state
            setMessages(prev => {
              const updated = prev.map(msg => 
                msg.id === assistantMessageId 
                  ? { ...msg, content: streamingContentRef.current }
                  : msg
              );
              return updated;
            });

            // Update thinking steps based on content length (approximate)
            const len = streamingContentRef.current.length;
            if (len < 50) {
              updateThinkingStep('Planner', 'active');
            } else if (len < 200) {
              updateThinkingStep('Planner', 'completed');
              updateThinkingStep('Executor', 'active');
            } else if (len < 500) {
              updateThinkingStep('Executor', 'completed');
              updateThinkingStep('Critic', 'active');
            } else if (deepThinking && len > 1000) {
              updateThinkingStep('Critic', 'completed');
              updateThinkingStep('Heavy Critic (Gemma 26B)', 'active');
            }
          }
        }
      }

      // Process any remaining buffer
      if (buffer.trim()) {
        const line = buffer.trim();
        if (line.startsWith('data:')) {
          const data = line.slice(5).trim();
          if (data && !data.startsWith('[')) {
            streamingContentRef.current += data.replace(/\\n/g, '\n');
            setMessages(prev => 
              prev.map(msg => 
                msg.id === assistantMessageId 
                  ? { ...msg, content: streamingContentRef.current }
                  : msg
              )
            );
          }
        }
      }

    } catch (error) {
      console.error('[Chat] Error:', error);
      if (error instanceof Error && error.name === 'AbortError') {
        console.log('[Chat] Request aborted');
      } else {
        setMessages(prev => 
          prev.map(msg => 
            msg.id === assistantMessageId 
              ? { ...msg, content: 'Error: Failed to get response. Please try again.', isStreaming: false }
              : msg
          )
        );
      }
    } finally {
      setIsLoading(false);
      setCurrentAgent('');
      setThinkingSteps(prev => 
        prev.map(step => ({ ...step, status: 'completed' }))
      );
      setMessages(prev => 
        prev.map(msg => 
          msg.id === assistantMessageId 
            ? { ...msg, isStreaming: false }
            : msg
        )
      );
      streamingContentRef.current = '';
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const getStatusIcon = (status: ThinkingStep['status']) => {
    switch (status) {
      case 'completed':
        return <div className="w-2 h-2 bg-green-500 rounded-full" />;
      case 'active':
        return <Loader2 size={14} className="animate-spin text-blue-400" />;
      default:
        return <div className="w-2 h-2 bg-gray-600 rounded-full" />;
    }
  };

  return (
    <div className="flex flex-col h-full bg-gray-950">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-gray-800 bg-gray-900/50">
        <div>
          <h1 className="text-xl font-semibold text-white">Swarm Chat</h1>
          <p className="text-sm text-gray-500">
            {conversationId ? `Conversation: ${conversationId.slice(0, 12)}...` : 'New Conversation'}
          </p>
        </div>

        <div className="flex items-center gap-4">
          {/* Deep Thinking Toggle */}
          <button
            onClick={() => setDeepThinking(!deepThinking)}
            disabled={isLoading}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
              deepThinking 
                ? 'bg-purple-600/20 text-purple-300 border border-purple-500/50' 
                : 'bg-gray-800 text-gray-400 border border-gray-700 hover:bg-gray-700'
            }`}
          >
            {deepThinking ? <Brain size={18} /> : <Zap size={18} />}
            <span className="text-sm font-medium">
              {deepThinking ? 'Deep' : 'Fast'}
            </span>
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {isLoadingHistory ? (
          <div className="flex flex-col items-center justify-center h-full text-gray-500">
            <Loader2 size={48} className="animate-spin mb-4" />
            <p>Loading conversation...</p>
          </div>
        ) : messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-gray-500">
            <Brain size={64} className="mb-4 opacity-30" />
            <p className="text-lg font-medium">Welcome to Local LLM Swarm</p>
            <p className="text-sm mt-2">Start a conversation with the multi-agent system</p>
            <div className="flex gap-4 mt-6 text-xs">
              <span className="px-3 py-1 bg-gray-800 rounded-full">Planner → Strategy</span>
              <span className="px-3 py-1 bg-gray-800 rounded-full">Executor → Draft</span>
              <span className="px-3 py-1 bg-gray-800 rounded-full">Critic → Polish</span>
            </div>
          </div>
        ) : (
          messages.map((message, index) => (
            <div
              key={message.id}
              className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-3xl px-6 py-4 rounded-2xl ${
                  message.role === 'user'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-800 text-gray-100'
                }`}
              >
                {message.role === 'assistant' ? (
                  <div className="min-h-[20px]">
                    {message.content ? (
                      <MarkdownRenderer content={message.content} />
                    ) : message.isStreaming ? (
                      <div className="flex items-center gap-2 text-gray-400 py-2">
                        <Loader2 size={16} className="animate-spin" />
                        <span className="text-sm">Thinking...</span>
                      </div>
                    ) : null}
                  </div>
                ) : (
                  <p className="whitespace-pre-wrap">{message.content}</p>
                )}
                
                {/* Streaming indicator */}
                {message.isStreaming && message.content.length > 0 && (
                  <div className="mt-3 flex items-center gap-2 text-xs text-gray-500 border-t border-gray-700 pt-2">
                    <div className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-pulse" />
                    <span>Streaming... ({message.content.length} chars)</span>
                  </div>
                )}
              </div>
            </div>
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Thinking Process Accordion */}
      {isLoading && thinkingSteps.length > 0 && (
        <div className="mx-6 mb-4 border border-gray-700 rounded-lg bg-gray-900/50 overflow-hidden">
          <button
            onClick={() => setShowThinking(!showThinking)}
            className="w-full flex items-center justify-between px-4 py-3 text-sm font-medium text-gray-300 hover:bg-gray-800/50 transition-colors"
          >
            <div className="flex items-center gap-2">
              <Brain size={16} className="text-purple-400" />
              <span>Thinking Process</span>
              {currentAgent && (
                <span className="text-xs text-blue-400 ml-2">
                  ({currentAgent}...)
                </span>
              )}
            </div>
            {showThinking ? <ChevronDown size={16} /> : <ChevronUp size={16} />}
          </button>
          
          {showThinking && (
            <div className="px-4 pb-4 space-y-2">
              {thinkingSteps.map((step, index) => (
                <div
                  key={step.agent}
                  className={`flex items-center gap-3 p-2 rounded-lg transition-colors ${
                    step.status === 'active' ? 'bg-blue-500/10' : ''
                  }`}
                >
                  <div className="flex-shrink-0">
                    {getStatusIcon(step.status)}
                  </div>
                  <div className="flex-1">
                    <p className={`text-sm ${
                      step.status === 'active' ? 'text-blue-400 font-medium' : 
                      step.status === 'completed' ? 'text-gray-300' : 'text-gray-500'
                    }`}>
                      {index + 1}. {step.agent}
                    </p>
                  </div>
                  <div className="text-xs text-gray-600">
                    {step.status === 'completed' ? 'Done' : 
                     step.status === 'active' ? 'Running...' : 'Waiting'}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Input */}
      <div className="border-t border-gray-800 p-6 bg-gray-900/50">
        <form onSubmit={handleSubmit} className="relative">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type your message... (Shift+Enter for new line)"
            disabled={isLoading}
            className="w-full bg-gray-800 text-white placeholder-gray-500 rounded-xl pl-4 pr-14 py-4 min-h-[56px] max-h-48 resize-none focus:outline-none focus:ring-2 focus:ring-blue-500/50 border border-gray-700"
            rows={1}
          />
          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            className="absolute right-3 bottom-3 p-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 disabled:text-gray-500 text-white rounded-lg transition-colors"
          >
            {isLoading ? (
              <Loader2 size={20} className="animate-spin" />
            ) : (
              <Send size={20} />
            )}
          </button>
        </form>
        <p className="text-xs text-gray-500 mt-2 text-center">
          Press Enter to send • Shift+Enter for new line
        </p>
      </div>
    </div>
  );
}
