import { NextRequest, NextResponse } from 'next/server'
import { handlePredict } from '@/app/api/predict/route'

export async function POST(req: NextRequest) {
  return handlePredict(req, true)
}

export async function OPTIONS() {
  return new NextResponse(null, {
    status: 200,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    },
  })
}
