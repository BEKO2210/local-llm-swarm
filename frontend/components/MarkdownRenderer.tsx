'use client';

import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import remarkGfm from 'remark-gfm';
import rehypeKatex from 'rehype-katex';
import { Components } from 'react-markdown';

interface MarkdownRendererProps {
  content: string;
  className?: string;
}

const components: Components = {
  h1: ({ children }) => <h1 className="text-2xl font-bold mt-6 mb-3 text-white">{children}</h1>,
  h2: ({ children }) => <h2 className="text-xl font-bold mt-5 mb-2 text-white">{children}</h2>,
  h3: ({ children }) => <h3 className="text-lg font-bold mt-4 mb-2 text-gray-100">{children}</h3>,
  p: ({ children }) => <p className="mb-4 leading-relaxed text-gray-300">{children}</p>,
  ul: ({ children }) => <ul className="list-disc pl-6 mb-4 space-y-1">{children}</ul>,
  ol: ({ children }) => <ol className="list-decimal pl-6 mb-4 space-y-1">{children}</ol>,
  li: ({ children }) => <li className="text-gray-300">{children}</li>,
  code: ({ className, children }) => {
    const isInline = !className;
    if (isInline) {
      return (
        <code className="bg-gray-700 text-pink-300 px-1.5 py-0.5 rounded text-sm font-mono">
          {children}
        </code>
      );
    }
    return (
      <pre className="bg-gray-800 rounded-lg p-4 overflow-x-auto my-4">
        <code className={`${className} text-sm font-mono text-gray-200`}>
          {children}
        </code>
      </pre>
    );
  },
  blockquote: ({ children }) => (
    <blockquote className="border-l-4 border-blue-500 pl-4 my-4 italic text-gray-400">
      {children}
    </blockquote>
  ),
  a: ({ href, children }) => (
    <a href={href} className="text-blue-400 hover:text-blue-300 underline" target="_blank" rel="noopener noreferrer">
      {children}
    </a>
  ),
  table: ({ children }) => (
    <div className="overflow-x-auto my-4">
      <table className="min-w-full border border-gray-700">{children}</table>
    </div>
  ),
  thead: ({ children }) => <thead className="bg-gray-800">{children}</thead>,
  tbody: ({ children }) => <tbody className="bg-gray-900">{children}</tbody>,
  tr: ({ children }) => <tr className="border-b border-gray-700">{children}</tr>,
  th: ({ children }) => <th className="px-4 py-2 text-left text-gray-200 font-semibold">{children}</th>,
  td: ({ children }) => <td className="px-4 py-2 text-gray-300">{children}</td>,
  hr: () => <hr className="my-6 border-gray-700" />,
};

export default function MarkdownRenderer({ content, className = '' }: MarkdownRendererProps) {
  return (
    <div className={`markdown-body ${className}`}>
      <ReactMarkdown
        remarkPlugins={[remarkMath, remarkGfm]}
        rehypePlugins={[rehypeKatex]}
        components={components}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
