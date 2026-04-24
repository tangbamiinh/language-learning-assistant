import { NextRequest, NextResponse } from 'next/server';
import { AccessToken } from 'livekit-server-sdk';
import { RoomConfiguration } from '@livekit/protocol';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    const apiKey = process.env.LIVEKIT_API_KEY;
    const apiSecret = process.env.LIVEKIT_API_SECRET;
    const serverUrl = process.env.LIVEKIT_URL;

    if (!apiKey || !apiSecret || !serverUrl) {
      return NextResponse.json(
        { error: 'Server configuration error. Check .env for LIVEKIT_* variables.' },
        { status: 500 }
      );
    }

    const roomName = body.room_name || `chat-${Date.now()}`;
    const participantIdentity = body.participant_identity || `user-${Date.now()}`;
    const participantName = body.participant_name || 'Student';
    const roomConfig = body.room_config;

    const at = new AccessToken(apiKey, apiSecret, {
      identity: participantIdentity,
      name: participantName,
      metadata: body.participant_metadata || '',
      attributes: body.participant_attributes || {},
      ttl: '30m',
    });

    at.addGrant({
      roomJoin: true,
      room: roomName,
      canPublish: true,
      canSubscribe: true,
      canPublishData: true,
    });

    if (roomConfig) {
      at.roomConfig = new RoomConfiguration(roomConfig);
    }

    const participantToken = await at.toJwt();

    return NextResponse.json(
      {
        server_url: serverUrl,
        participant_token: participantToken,
      },
      { status: 201 }
    );
  } catch (error) {
    console.error('LiveKit token generation error:', error);
    return NextResponse.json(
      { error: 'Failed to generate LiveKit token' },
      { status: 500 }
    );
  }
}
