import { Heart } from 'lucide-react'

export function Footer() {
  return (
    <footer className="relative mt-32 border-t border-white/5">
      {/* Gradient overlay */}
      <div className="absolute inset-0 bg-gradient-to-t from-purple-900/10 to-transparent pointer-events-none" />
      
      <div className="relative max-w-6xl mx-auto px-4 py-16">
        <div className="grid md:grid-cols-4 gap-12 mb-12">
          {/* Brand */}
          <div className="md:col-span-2">
            <h3 className="text-2xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent mb-4">
              FrameTrain
            </h3>
            <p className="text-gray-400 text-sm leading-relaxed max-w-md">
              Professionelle Desktop-Anwendung für lokales Machine Learning Training. 
              Train smarter, not harder.
            </p>
            
            {/* Social badges */}
            <div className="flex gap-3 mt-6">
              <div className="glass px-4 py-2 rounded-lg text-sm text-gray-400 hover:text-purple-400 transition cursor-pointer">
                GitHub
              </div>
              <div className="glass px-4 py-2 rounded-lg text-sm text-gray-400 hover:text-blue-400 transition cursor-pointer">
                Twitter
              </div>
              <div className="glass px-4 py-2 rounded-lg text-sm text-gray-400 hover:text-pink-400 transition cursor-pointer">
                Discord
              </div>
            </div>
          </div>
          
          {/* Product */}
          <div>
            <h4 className="font-semibold text-white mb-4">Produkt</h4>
            <ul className="space-y-3 text-sm text-gray-400">
              <li>
                <a href="/#features" className="hover:text-purple-400 transition">
                  Features
                </a>
              </li>
              <li>
                <a href="/#pricing" className="hover:text-purple-400 transition">
                  Pricing
                </a>
              </li>
              <li>
                <a href="/docs" className="hover:text-purple-400 transition">
                  Dokumentation
                </a>
              </li>
              <li>
                <a href="/changelog" className="hover:text-purple-400 transition">
                  Changelog
                </a>
              </li>
            </ul>
          </div>
          
          {/* Legal */}
          <div>
            <h4 className="font-semibold text-white mb-4">Rechtliches</h4>
            <ul className="space-y-3 text-sm text-gray-400">
              <li>
                <a href="/privacy" className="hover:text-purple-400 transition">
                  Datenschutz
                </a>
              </li>
              <li>
                <a href="/terms" className="hover:text-purple-400 transition">
                  AGB
                </a>
              </li>
              <li>
                <a href="/imprint" className="hover:text-purple-400 transition">
                  Impressum
                </a>
              </li>
              <li>
                <a href="/cookies" className="hover:text-purple-400 transition">
                  Cookies
                </a>
              </li>
            </ul>
          </div>
        </div>
        
        {/* Bottom bar */}
        <div className="pt-8 border-t border-white/5 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="text-sm text-gray-500">
            © {new Date().getFullYear()} FrameTrain. All rights reserved.
          </div>
          
          <div className="flex items-center gap-2 text-xs text-gray-600">
            <span>Made with</span>
            <Heart className="w-4 h-4 text-pink-500 fill-pink-500 animate-pulse" />
            <span>for ML Engineers</span>
          </div>
        </div>
      </div>
    </footer>
  )
}
