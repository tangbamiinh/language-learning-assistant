'use client';

import { useEffect, useState } from 'react';
import {
  ControlBar,
  RoomAudioRenderer,
  useSession,
  SessionProvider,
  useAgent,
  BarVisualizer,
} from '@livekit/components-react';
import { TokenSource } from 'livekit-client';
import '@livekit/components-styles';

const tokenSource = TokenSource.endpoint('/api/livekit-token');

export function LiveKitVoiceRoom({
  onDisconnect,
}: {
  onDisconnect: () => void;
}) {
  const session = useSession(tokenSource, {
    agentName: 'chinese-tutor',
    roomName: `chat-${Date.now()}`,
  });

  const [isConnecting, setIsConnecting] = useState(true);

  useEffect(() => {
    session.start();
    setIsConnecting(false);
    return () => {
      session.end();
      onDisconnect();
    };
  }, [session, onDisconnect]);

  return (
    <SessionProvider session={session}>
      <div className="flex flex-col items-center justify-center gap-4 py-4">
        {isConnecting ? (
          <div className="flex items-center gap-2 text-sm text-gray-500">
            <span className="w-2 h-2 bg-violet-400 rounded-full animate-pulse" />
            Connecting to voice agent...
          </div>
        ) : (
          <AgentStatusView />
        )}

        <ControlBar
          controls={{ microphone: true, camera: false, screenShare: false }}
          className="lk-controlbar"
        />

        <RoomAudioRenderer />
      </div>
    </SessionProvider>
  );
}

function AgentStatusView() {
  const agent = useAgent();

  const agentStateLabel = getAgentStateLabel(agent.state);
  const agentStateColor = getAgentStateColor(agent.state);

  return (
    <div className="flex flex-col items-center gap-4 w-full max-w-md">
      {/* Agent avatar */}
      <div className="relative">
        <div
          className={`w-20 h-20 rounded-full bg-gradient-to-br from-violet-500 to-indigo-600 flex items-center justify-center text-white font-bold text-lg shadow-lg transition-all ${
            agent.state === 'speaking' ? 'scale-110 shadow-xl' : ''
          }`}
        >
          小明
        </div>
        <div
          className={`absolute -bottom-1 -right-1 w-5 h-5 rounded-full border-2 border-white ${agentStateColor}`}
        />
      </div>

      {/* Agent state */}
      <p className="text-sm font-medium text-gray-600">{agentStateLabel}</p>

      {/* Agent audio visualizer */}
      {agent.microphoneTrack && (
        <div className="w-full h-16 flex items-center justify-center">
          <BarVisualizer
            track={agent.microphoneTrack}
            state={agent.state}
            barCount={24}
            className="w-full"
          />
        </div>
      )}
    </div>
  );
}

function getAgentStateLabel(state: string): string {
  switch (state) {
    case 'connecting':
      return 'Connecting...';
    case 'idle':
      return 'Listening...';
    case 'speaking':
      return 'Speaking...';
    case 'thinking':
      return 'Thinking...';
    case 'listening':
      return 'Listening...';
    default:
      return state || 'Connecting...';
  }
}

function getAgentStateColor(state: string): string {
  switch (state) {
    case 'connecting':
      return 'bg-yellow-400';
    case 'idle':
    case 'listening':
      return 'bg-green-400';
    case 'speaking':
      return 'bg-violet-500';
    case 'thinking':
      return 'bg-blue-400';
    default:
      return 'bg-gray-400';
  }
}
