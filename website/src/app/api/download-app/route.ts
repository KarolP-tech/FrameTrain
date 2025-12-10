import { NextRequest, NextResponse } from 'next/server';
import { prisma } from '@/lib/prisma';
import { verifyJWT } from '@/lib/jwt';

// Force dynamic rendering for this route
export const dynamic = 'force-dynamic';

// GitHub Repository Info
const GITHUB_OWNER = process.env.GITHUB_OWNER || 'YourUsername';
const GITHUB_REPO = process.env.GITHUB_REPO || 'FrameTrain';
const GITHUB_TOKEN = process.env.GITHUB_TOKEN; // Optional: für private repos

interface ReleaseAsset {
  name: string;
  browser_download_url: string;
  size: number;
}

interface GitHubRelease {
  tag_name: string;
  assets: ReleaseAsset[];
  prerelease: boolean;
  draft: boolean;
  published_at: string;
}

/**
 * Get the latest release from GitHub
 * CRITICAL FIX: Get ALL releases and sort by date to find the ACTUAL latest
 */
async function getLatestRelease(): Promise<GitHubRelease | null> {
  try {
    const headers: HeadersInit = {
      'Accept': 'application/vnd.github+json',
      'X-GitHub-Api-Version': '2022-11-28',
    };

    if (GITHUB_TOKEN) {
      headers['Authorization'] = `Bearer ${GITHUB_TOKEN}`;
    }

    // Get ALL releases, not just "latest" tag
    const response = await fetch(
      `https://api.github.com/repos/${GITHUB_OWNER}/${GITHUB_REPO}/releases`,
      { headers }
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
    
    console.log('[Download] Latest release:', latestRelease.tag_name, 'published:', latestRelease.published_at);
    console.log('[Download] Assets:', latestRelease.assets.map(a => a.name));
    
    return latestRelease;
  } catch (error) {
    console.error('Failed to fetch release:', error);
    return null;
  }
}

/**
 * Find the correct asset for the platform
 */
function findAssetForPlatform(assets: ReleaseAsset[], platform: string): ReleaseAsset | null {
  // Exclude source code archives
  const installerAssets = assets.filter(asset => 
    !asset.name.includes('Source code') &&
    !asset.name.endsWith('.zip') && 
    !asset.name.endsWith('.tar.gz')
  );
  
  const patterns: Record<string, RegExp> = {
    windows: /\.(msi|exe)$/i,
    mac: /\.dmg$/i,
    linux: /\.AppImage$/i,
  };

  const pattern = patterns[platform];
  if (!pattern) return null;

  const found = installerAssets.find(asset => pattern.test(asset.name));
  
  // Debug logging
  console.log('Platform:', platform);
  console.log('Available installer assets:', installerAssets.map(a => a.name));
  console.log('Found asset:', found?.name);
  
  return found || null;
}

/**
 * Verify API key and check if user has access
 */
async function verifyAccess(apiKey?: string, token?: string): Promise<boolean> {
  try {
    // Check JWT token (logged in user)
    if (token) {
      const decoded = verifyJWT(token);
      if (decoded?.userId) {
        const user = await prisma.user.findUnique({
          where: { id: decoded.userId },
          select: { hasPaid: true },
        });
        return user?.hasPaid || false;
      }
    }

    // Check API key (CLI users)
    if (apiKey) {
      const key = await prisma.apiKey.findUnique({
        where: { key: apiKey },
        select: { isActive: true },
      });
      return key?.isActive || false;
    }

    return false;
  } catch (error) {
    console.error('Access verification error:', error);
    return false;
  }
}

/**
 * Log download for analytics
 */
async function logDownload(platform: string, userId?: string, apiKey?: string) {
  try {
    // You can implement download tracking here
    console.log(`Download: platform=${platform}, userId=${userId}, apiKey=${apiKey?.substring(0, 8)}...`);
  } catch (error) {
    console.error('Failed to log download:', error);
  }
}

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const platform = searchParams.get('platform');
    const apiKey = searchParams.get('key');
    const authHeader = request.headers.get('authorization');
    const token = authHeader?.replace('Bearer ', '');

    // Validate platform
    if (!platform || !['windows', 'mac', 'linux'].includes(platform)) {
      return NextResponse.json(
        { error: 'Ungültige Plattform. Bitte wähle: windows, mac oder linux' },
        { status: 400 }
      );
    }

    // Verify access (API key or JWT token)
    const hasAccess = await verifyAccess(apiKey || undefined, token || undefined);
    
    if (!hasAccess) {
      return NextResponse.json(
        { 
          error: 'Zugriff verweigert',
          message: 'Ungültiger API-Key oder fehlende Berechtigung. Bitte kaufe zuerst eine Lizenz auf https://frametrain.ai'
        },
        { status: 403 }
      );
    }

    // Get latest release from GitHub
    const release = await getLatestRelease();
    
    if (!release) {
      return NextResponse.json(
        { 
          error: 'Release nicht gefunden',
          message: 'Es ist noch keine Version verfügbar. Bitte versuche es später erneut.'
        },
        { status: 404 }
      );
    }

    // Find the correct asset for the platform
    const asset = findAssetForPlatform(release.assets, platform);
    
    if (!asset) {
      return NextResponse.json(
        { 
          error: 'Download nicht verfügbar',
          message: `Für ${platform} ist noch kein Installer verfügbar.`
        },
        { status: 404 }
      );
    }

    // Log the download
    await logDownload(platform, token || undefined, apiKey || undefined);

    // Return download URL (redirect or JSON)
    const returnUrl = searchParams.get('redirect') === 'true';
    
    if (returnUrl) {
      // Direct redirect to GitHub download
      return NextResponse.redirect(asset.browser_download_url);
    } else {
      // Return JSON with download info
      return NextResponse.json({
        success: true,
        version: release.tag_name,
        platform,
        download_url: asset.browser_download_url,
        filename: asset.name,
        size: asset.size,
        size_mb: (asset.size / 1024 / 1024).toFixed(2),
      });
    }

  } catch (error) {
    console.error('Download-Fehler:', error);
    return NextResponse.json(
      { 
        error: 'Download fehlgeschlagen',
        message: error instanceof Error ? error.message : 'Unbekannter Fehler'
      },
      { status: 500 }
    );
  }
}
