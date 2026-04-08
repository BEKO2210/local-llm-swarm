'use client';

import React, { useState, useEffect } from 'react';
import { MessageSquare, Plus, ChevronRight, Loader2 } from 'lucide-react';

interface Conversation {
  id: string;
  title: string;
  last_updated: string;
  message_count: number;
}

interface SidebarProps {
  onSelectConversation: (id: string) => void;
  onNewConversation: () => void;
  activeConversationId?: string;
}

export default function Sidebar({ 
  onSelectConversation, 
  onNewConversation,
  activeConversationId 
}: SidebarProps) {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchConversations = async () => {
    try {
      setLoading(true);
      const response = await fetch('http://127.0.0.1:8000/api/chat/conversations');
      if (!response.ok) {
        throw new Error('Failed to fetch conversations');
      }
      const data = await response.json();
      setConversations(data.conversations || []);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      console.error('Error fetching conversations:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchConversations();
    // Refresh every 30 seconds
    const interval = setInterval(fetchConversations, 30000);
    return () => clearInterval(interval);
  }, []);

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffDays = Math.floor((now.getTime() - date.getTime()) / (1000 * 60 * 60 * 24));
    
    if (diffDays === 0) {
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } else if (diffDays === 1) {
      return 'Yesterday';
    } else if (diffDays < 7) {
      return date.toLocaleDateString([], { weekday: 'short' });
    } else {
      return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
    }
  };

  return (
    <div className="w-80 bg-gray-900 border-r border-gray-800 flex flex-col h-full">
      {/* Header */}
      <div className="p-4 border-b border-gray-800">
        <button
          onClick={onNewConversation}
          className="w-full flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-700 text-white py-3 px-4 rounded-lg transition-colors font-medium"
        >
          <Plus size={20} />
          New Conversation
        </button>
      </div>

      {/* Conversations List */}
      <div className="flex-1 overflow-y-auto">
        {loading && conversations.length === 0 ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="animate-spin text-gray-500" size={24} />
          </div>
        ) : error ? (
          <div className="p-4 text-center">
            <p className="text-red-400 text-sm">{error}</p>
            <button
              onClick={fetchConversations}
              className="mt-2 text-blue-400 hover:text-blue-300 text-sm"
            >
              Retry
            </button>
          </div>
        ) : conversations.length === 0 ? (
          <div className="p-4 text-center text-gray-500">
            <MessageSquare size={48} className="mx-auto mb-3 opacity-50" />
            <p className="text-sm">No conversations yet</p>
            <p className="text-xs mt-1">Start a new chat to begin</p>
          </div>
        ) : (
          <div className="py-2">
            {conversations.map((conv) => (
              <button
                key={conv.id}
                onClick={() => onSelectConversation(conv.id)}
                className={`w-full text-left p-3 mx-2 rounded-lg transition-colors group ${
                  activeConversationId === conv.id
                    ? 'bg-gray-800 border-l-4 border-blue-500'
                    : 'hover:bg-gray-800/50 border-l-4 border-transparent'
                }`}
                style={{ width: 'calc(100% - 16px)' }}
              >
                <div className="flex items-start gap-3">
                  <MessageSquare 
                    size={18} 
                    className={`mt-0.5 flex-shrink-0 ${
                      activeConversationId === conv.id ? 'text-blue-400' : 'text-gray-500'
                    }`} 
                  />
                  <div className="flex-1 min-w-0">
                    <p className={`text-sm font-medium truncate ${
                      activeConversationId === conv.id ? 'text-white' : 'text-gray-300'
                    }`}>
                      {conv.title}
                    </p>
                    <div className="flex items-center justify-between mt-1">
                      <span className="text-xs text-gray-500">
                        {conv.message_count} messages
                      </span>
                      <span className="text-xs text-gray-500">
                        {formatDate(conv.last_updated)}
                      </span>
                    </div>
                  </div>
                  <ChevronRight 
                    size={16} 
                    className={`flex-shrink-0 transition-opacity ${
                      activeConversationId === conv.id ? 'opacity-100 text-blue-400' : 'opacity-0 group-hover:opacity-50'
                    }`} 
                  />
                </div>
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-gray-800">
        <div className="flex items-center gap-2 text-xs text-gray-500">
          <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
          <span>System Online</span>
        </div>
      </div>
    </div>
  );
}
