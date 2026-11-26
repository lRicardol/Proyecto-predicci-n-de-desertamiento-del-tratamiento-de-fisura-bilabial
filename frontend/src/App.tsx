import React, { useState } from 'react';
import { Lock, Menu, ShieldCheck, X } from 'lucide-react';
import { AuthProvider, useAuth } from './context/AuthContext';
import { ErrorBoundary } from './components/ErrorBoundary';
import { OverviewPage } from './pages/OverviewPage';
import { RiskFactorsPage } from './pages/RiskFactorsPage';
import { SimulatorPage } from './pages/SimulatorPage';
import { FollowUpPage } from './pages/FollowUpPage';
import { PrivacyGovernancePage } from './pages/PrivacyGovernancePage';
import { PageId } from './types/clinical';

const NAV_ITEMS: { id: PageId; label: string }[] = [
  { id: 'resumen', label: 'Resumen' },
  { id: 'factores', label: 'Factores' },
  { id: 'simulador', label: 'Simulador' },
  { id: 'seguimiento', label: 'Seguimiento' },
  { id: 'privacidad', label: 'Privacidad' },
];

function AppShell() {
  const [activePage, setActivePage] = useState<PageId>('resumen');
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const { user, followUps, signInWithGoogle } = useAuth();

  const handleNavigate = (page: PageId) => {
    setActivePage(page);
    setMobileMenuOpen(false);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 text-slate-900">
      {/* Strict 3-Zone Top Bar Contract */}
      <header className="sticky top-0 z-40 bg-white border-b border-slate-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between gap-8">
          {/* Zone 1: Single-element Brand Wordmark */}
          <button
            type="button"
            onClick={() => handleNavigate('resumen')}
            className="text-lg font-bold tracking-tight text-slate-900 whitespace-nowrap shrink-0 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-600 rounded"
          >
            FisuraCare
          </button>

          {/* Zone 2: 5 concise single-line text nav links (Desktop & Tablet Landscape) */}
          <nav
            aria-label="Navegación principal"
            className="hidden md:flex items-center gap-6 text-sm font-medium text-slate-600"
          >
            {NAV_ITEMS.map((item) => {
              const isActive = activePage === item.id;
              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => handleNavigate(item.id)}
                  className={`py-2 whitespace-nowrap shrink-0 border-b-2 transition-colors ${
                    isActive
                      ? 'text-teal-800 border-teal-700 font-semibold'
                      : 'border-transparent hover:text-slate-900 hover:border-slate-300'
                  }`}
                >
                  {item.label}
                </button>
              );
            })}
          </nav>

          {/* Zone 3: 1 Primary Action + Tablet/Mobile drawer trigger */}
          <div className="flex items-center gap-2.5 shrink-0">
            <button
              type="button"
              onClick={() => (user ? handleNavigate('seguimiento') : signInWithGoogle())}
              className="inline-flex items-center gap-2 px-4 py-2 min-h-[40px] text-xs font-semibold text-white bg-teal-700 hover:bg-teal-800 rounded-lg transition-colors whitespace-nowrap shrink-0"
            >
              <Lock className="w-3.5 h-3.5" />
              {user ? 'Panel Médico' : 'Acceso Especialista'}
            </button>

            <button
              type="button"
              aria-label={mobileMenuOpen ? 'Cerrar menú' : 'Abrir menú'}
              onClick={() => setMobileMenuOpen((v) => !v)}
              className="md:hidden p-2 min-h-[44px] min-w-[44px] inline-flex items-center justify-center text-slate-700 hover:bg-slate-100 rounded-lg"
            >
              {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
          </div>
        </div>

        {/* Tablet Portrait / Mobile Navigation Drawer */}
        {mobileMenuOpen && (
          <nav
            aria-label="Navegación móvil"
            className="md:hidden border-t border-slate-200 bg-white px-4 py-3 space-y-1"
          >
            {NAV_ITEMS.map((item) => {
              const isActive = activePage === item.id;
              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => handleNavigate(item.id)}
                  className={`w-full text-left px-3 py-2.5 min-h-[44px] rounded-lg text-sm font-medium transition-colors ${
                    isActive
                      ? 'bg-teal-50 text-teal-800 font-semibold'
                      : 'text-slate-700 hover:bg-slate-50'
                  }`}
                >
                  {item.label}
                </button>
              );
            })}
          </nav>
        )}
      </header>

      {/* Main Viewport */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 py-6 sm:py-8">
        {activePage === 'resumen' && (
          <OverviewPage
            onNavigate={handleNavigate}
            specialistSignedIn={Boolean(user)}
            assignedCasesCount={followUps.length}
          />
        )}
        {activePage === 'factores' && <RiskFactorsPage />}
        {activePage === 'simulador' && <SimulatorPage onNavigate={handleNavigate} />}
        {activePage === 'seguimiento' && <FollowUpPage />}
        {activePage === 'privacidad' && <PrivacyGovernancePage />}
      </main>

      {/* Clean Quiet Footer */}
      <footer className="bg-white border-t border-slate-200 mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-5 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 text-xs text-slate-500">
          <div className="flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-teal-700 shrink-0" />
            <span>
              Cohorte 100% sintética · Cumplimiento Ley 1581 de 2012, RGPD e HIPAA Safe Harbor
            </span>
          </div>
          <div className="flex items-center gap-4">
            <button
              type="button"
              onClick={() => handleNavigate('privacidad')}
              className="hover:text-slate-800 underline underline-offset-2"
            >
              Política de datos sintéticos
            </button>
            <span aria-hidden="true">·</span>
            <span>FisuraCare</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default function App() {
  return (
    <ErrorBoundary>
      <AuthProvider>
        <AppShell />
      </AuthProvider>
    </ErrorBoundary>
  );
}
