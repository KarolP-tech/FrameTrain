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
  prerelease: boolean;
  draft: boolean;
}

/**
 * Get the latest release version from GitHub
 * Uses /releases endpoint and sorts by date to get the ACTUAL latest release
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

    // CRITICAL FIX: Get ALL releases and sort by date to find the ACTUAL latest
    // Don't rely on GitHub's "latest" flag which might be outdated
    const response = await fetch(
      `https://api.github.com/repos/${GITHUB_OWNER}/${GITHUB_REPO}/releases`,
      { 
        headers,
        next: { revalidate: 60 } // Cache for 1 minute only
      }
    );

    if (!response.ok) {
      console.error('GitHub API error:', response.status, response.statusText);
      return null;
    }

    const releases: GitHubRelease[] = await response.json();
    
    if (!releases || releases.length === 0) {
      console.error('No releases found');
      return null;
    }

    // Filter out pre-releases and drafts, then sort by date
    const validReleases = releases
      .filter(r => !r.prerelease && !r.draft)
      .sort((a, b) => new Date(b.published_at).getTime() - new Date(a.published_at).getTime());
    
    if (validReleases.length === 0) {
      console.error('No valid releases found');
      return null;
    }

    const latestRelease = validReleases[0];
    
    console.log('[App Version] Latest release:', latestRelease.tag_name, 'published:', latestRelease.published_at);
    console.log('[App Version] All releases:', releases.map(r => `${r.tag_name} (${r.published_at})`));
    
    return {
      version: latestRelease.tag_name,
      name: latestRelease.name,
      date: latestRelease.published_at
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
