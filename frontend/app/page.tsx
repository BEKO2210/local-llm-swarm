'use client';

import React, { useState } from 'react';
import Sidebar from '@/components/Sidebar';
import ChatInterface from '@/components/ChatInterface';

export default function Home() {
  const [activeConversationId, setActiveConversationId] = useState<string | undefined>();

  const handleNewConversation = () => {
    setActiveConversationId(undefined);
  };

  const handleSelectConversation = (id: string) => {
    setActiveConversationId(id);
  };

  const handleConversationCreated = (id: string) => {
    setActiveConversationId(id);
  };

  return (
    <main className="flex h-screen bg-gray-950 text-white overflow-hidden">
      <Sidebar
        onSelectConversation={handleSelectConversation}
        onNewConversation={handleNewConversation}
        activeConversationId={activeConversationId}
      />
      <div className="flex-1">
        <ChatInterface
          conversationId={activeConversationId}
          onConversationCreated={handleConversationCreated}
        />
      </div>
    </main>
  );
}
