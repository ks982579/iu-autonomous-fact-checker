import React, { createContext, useContext, useState, useEffect } from 'react';
import type { ReactNode } from 'react';

interface HealthStatus {
  status: 'healthy' | 'unhealthy' | 'checking' | 'unknown';
  lastChecked: Date | null;
  baseUrl: string;
  error?: string;
}

interface HealthContextType {
  health: HealthStatus;
  checkHealth: () => Promise<void>;
}

interface Config {
  api: {
    host: string;
    port: number;
    base_url: string;
  };
  extension: {
    health_check_poll_interval_ms: number;
    request_timeout_ms: number;
  };
}

const HealthContext = createContext<HealthContextType | undefined>(undefined);

export const useHealth = (): HealthContextType => {
  const context = useContext(HealthContext);
  if (!context) {
    throw new Error('useHealth must be used within a HealthProvider');
  }
  return context;
};

interface HealthProviderProps {
  children: ReactNode;
}

export const HealthProvider: React.FC<HealthProviderProps> = ({ children }) => {
  const [config, setConfig] = useState<Config | null>(null);
  const [health, setHealth] = useState<HealthStatus>({
    status: 'unknown',
    lastChecked: null,
    baseUrl: 'http://localhost:8000',
  });

  // Load config on mount
  useEffect(() => {
    const loadConfig = async () => {
      try {
        const response = await fetch(chrome.runtime.getURL('config.json'));
        const configData: Config = await response.json();
        setConfig(configData);
        
        // Update baseUrl from config
        setHealth(prev => ({
          ...prev,
          baseUrl: configData.api.base_url
        }));
      } catch (error) {
        console.error('Failed to load config:', error);
        // Use default config
        const defaultConfig: Config = {
          api: { host: 'localhost', port: 8000, base_url: 'http://localhost:8000' },
          extension: { health_check_poll_interval_ms: 3000, request_timeout_ms: 180000 }
        };
        setConfig(defaultConfig);
      }
    };

    loadConfig();
  }, []);

  const checkHealth = async (): Promise<void> => {
    if (!config) return;

    setHealth(prev => ({ ...prev, status: 'checking' }));

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), config.extension.request_timeout_ms);

      const response = await fetch(`${health.baseUrl}/health`, {
        method: 'GET',
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (response.ok) {
        const data = await response.json();
        setHealth(prev => ({
          ...prev,
          status: data.status === 'healthy' ? 'healthy' : 'unhealthy',
          lastChecked: new Date(),
          error: undefined
        }));
      } else {
        setHealth(prev => ({
          ...prev,
          status: 'unhealthy',
          lastChecked: new Date(),
          error: `HTTP ${response.status}: ${response.statusText}`
        }));
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      setHealth(prev => ({
        ...prev,
        status: 'unhealthy',
        lastChecked: new Date(),
        error: errorMessage.includes('aborted') ? 'Request timeout' : errorMessage
      }));
    }
  };

  // Set up polling interval
  useEffect(() => {
    if (!config) return;

    // Initial health check
    checkHealth();

    // Set up interval for regular checks
    const interval = setInterval(checkHealth, config.extension.health_check_poll_interval_ms);

    // Cleanup on unmount
    return () => {
      clearInterval(interval);
    };
  }, [config]); // Re-run when config changes

  const contextValue: HealthContextType = {
    health,
    checkHealth,
  };

  return (
    <HealthContext.Provider value={contextValue}>
      {children}
    </HealthContext.Provider>
  );
};