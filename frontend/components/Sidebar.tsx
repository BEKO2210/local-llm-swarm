'use client';

import React, { useState, useEffect } from 'react';
import { MessageSquare, Plus, ChevronRight, Loader2, Trash2, AlertCircle } from 'lucide-react';

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
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);

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

  const handleDelete = async (e: React.MouseEvent, convId: string) => {
    e.stopPropagation();
    
    if (confirmDelete === convId) {
      // Actually delete
      try {
        setDeletingId(convId);
        const response = await fetch(`http://127.0.0.1:8000/api/chat/conversations/${convId}`, {
          method: 'DELETE',
        });
        
        if (!response.ok) {
          throw new Error('Failed to delete');
        }
        
        // Remove from list
        setConversations(prev => prev.filter(c => c.id !== convId));
        
        // If this was the active conversation, go to new conversation
        if (activeConversationId === convId) {
          onNewConversation();
        }
      } catch (err) {
        console.error('Error deleting conversation:', err);
      } finally {
        setDeletingId(null);
        setConfirmDelete(null);
      }
    } else {
      // Show confirmation
      setConfirmDelete(convId);
      // Auto-hide confirmation after 3 seconds
      setTimeout(() => setConfirmDelete(prev => prev === convId ? null : prev), 3000);
    }
  };

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
            <AlertCircle size={48} className="mx-auto mb-3 text-red-500 opacity-70" />
            <p className="text-red-400 text-sm">{error}</p>
            <button
              onClick={fetchConversations}
              className="mt-3 px-4 py-2 bg-gray-800 hover:bg-gray-700 text-blue-400 hover:text-blue-300 rounded-lg text-sm transition-colors"
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
              <div
                key={conv.id}
                className={`group mx-2 rounded-lg transition-all ${
                  activeConversationId === conv.id
                    ? 'bg-gray-800 border-l-4 border-blue-500'
                    : 'hover:bg-gray-800/50 border-l-4 border-transparent'
                }`}
              >
                <button
                  onClick={() => onSelectConversation(conv.id)}
                  className="w-full text-left p-3 flex items-start gap-3"
                >
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
                  <div className="flex items-center gap-1">
                    {confirmDelete === conv.id ? (
                      <button
                        onClick={(e) => handleDelete(e, conv.id)}
                        className="p-1.5 bg-red-600 hover:bg-red-700 text-white rounded transition-colors animate-pulse"
                        title="Click again to confirm delete"
                      >
                        <AlertCircle size={14} />
                      </button>
                    ) : deletingId === conv.id ? (
                      <Loader2 size={16} className="animate-spin text-gray-400" />
                    ) : (
                      <button
                        onClick={(e) => handleDelete(e, conv.id)}
                        className="p-1.5 text-gray-600 hover:text-red-400 hover:bg-red-500/10 rounded opacity-0 group-hover:opacity-100 transition-all"
                        title="Delete conversation"
                      >
                        <Trash2 size={14} />
                      </button>
                    )}
                    <ChevronRight 
                      size={16} 
                      className={`flex-shrink-0 transition-opacity text-gray-500 ${
                        activeConversationId === conv.id ? 'opacity-100' : 'opacity-0 group-hover:opacity-50'
                      }`} 
                    />
                  </div>
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-gray-800">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-xs text-gray-500">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
            <span>System Online</span>
          </div>
          {conversations.length > 0 && (
            <span className="text-xs text-gray-600">
              {conversations.length} conversation{conversations.length !== 1 ? 's' : ''}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
