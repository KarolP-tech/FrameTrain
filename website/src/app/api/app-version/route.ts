import { NextResponse } from 'next/server';

// Force dynamic rendering
export const dynamic = 'force-dynamic';

// GitHub Repository Info
const GITHUB_OWNER = process.env.GITHUB_OWNER || 'KarolP-tech';
const GITHUB_REPO = process.env.GITHUB_REPO || 'FrameTrain';
const GITHUB_TOKEN = process.env.GITHUB_TOKEN;

interface GitHubRelease {
  tag_name: string;
  name: string;
  published_at: string;
}

/**
 * Get the latest release version from GitHub
 */
async function getLatestVersion(): Promise<{ version: string; name: string; date: string } | null> {
  try {
    const headers: HeadersInit = {
      'Accept': 'application/vnd.github+json',
      'X-GitHub-Api-Version': '2022-11-28',
    };

    if (GITHUB_TOKEN) {
      headers['Authorization'] = `Bearer ${GITHUB_TOKEN}`;
    }

    const response = await fetch(
      `https://api.github.com/repos/${GITHUB_OWNER}/${GITHUB_REPO}/releases/latest`,
      { 
        headers,
        next: { revalidate: 300 } // Cache for 5 minutes
      }
    );

    if (!response.ok) {
      console.error('GitHub API error:', response.status, response.statusText);
      return null;
    }

    const release: GitHubRelease = await response.json();
    
    return {
      version: release.tag_name,
      name: release.name,
      date: release.published_at
    };
  } catch (error) {
    console.error('Failed to fetch version:', error);
    return null;
  }
}

/**
 * GET /api/app-version
 * Returns the latest app version from GitHub releases
 */
export async function GET() {
  try {
    const versionInfo = await getLatestVersion();
    
    if (!versionInfo) {
      return NextResponse.json(
        { 
          error: 'Version nicht gefunden',
          version: '1.0.0', // Fallback
        },
        { status: 200 } // Still return 200 with fallback
      );
    }

    return NextResponse.json({
      success: true,
      ...versionInfo
    });

  } catch (error) {
    console.error('Version fetch error:', error);
    return NextResponse.json(
      { 
        error: 'Fehler beim Abrufen der Version',
        version: '1.0.0', // Fallback
      },
      { status: 200 }
    );
  }
}
