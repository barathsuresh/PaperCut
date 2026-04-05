import { motion, AnimatePresence } from 'framer-motion'
import { useEffect, useState } from 'react'

const WORD = 'PaperCut'

const containerVariants = {
  hidden: {},
  show: { transition: { staggerChildren: 0.07, delayChildren: 0.15 } },
  exit:  { transition: { staggerChildren: 0.04, staggerDirection: -1 } },
}

const letterVariants = {
  hidden: { y: 28, opacity: 0, filter: 'blur(8px)' },
  show:   { y: 0,  opacity: 1, filter: 'blur(0px)', transition: { duration: 0.5, ease: [0.22, 1, 0.36, 1] } },
  exit:   { y: -20, opacity: 0, filter: 'blur(4px)', transition: { duration: 0.3, ease: 'easeIn' } },
}

const glowVariants = {
  hidden:  { opacity: 0, scale: 0.6 },
  show:    { opacity: 1, scale: 1,   transition: { delay: 0.5, duration: 0.8, ease: 'easeOut' } },
  exit:    { opacity: 0, scale: 1.3, transition: { duration: 0.5 } },
}

const subVariants = {
  hidden: { opacity: 0, y: 8 },
  show:   { opacity: 1, y: 0, transition: { delay: 0.85, duration: 0.5 } },
  exit:   { opacity: 0,       transition: { duration: 0.3 } },
}

export default function SplashScreen({ onComplete }) {
  const [phase, setPhase] = useState('show') // show | exit

  useEffect(() => {
    const t1 = setTimeout(() => setPhase('exit'), 2200)
    const t2 = setTimeout(onComplete, 2850)
    return () => { clearTimeout(t1); clearTimeout(t2) }
  }, [onComplete])

  return (
    <AnimatePresence>
      {phase !== 'gone' && (
        <motion.div
          className="splash"
          initial={{ opacity: 1 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.55 }}
        >
          {/* Glow */}
          <motion.div
            className="splash-glow"
            variants={glowVariants}
            initial="hidden"
            animate={phase === 'show' ? 'show' : 'exit'}
          />

          <div className="splash-inner">
            {/* ∂ symbol above the wordmark */}
            <motion.div
              initial={{ opacity: 0, scale: 0.6 }}
              animate={phase === 'show' ? { opacity: 1, scale: 1 } : { opacity: 0, scale: 1.2 }}
              transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
              style={{
                fontSize: 52,
                fontWeight: 700,
                color: 'var(--accent2)',
                lineHeight: 1,
                marginBottom: 12,
                textShadow: '0 0 40px rgba(139,92,246,0.9), 0 0 80px rgba(139,92,246,0.4)',
              }}
            >
              ∂
            </motion.div>

            {/* Staggered letters */}
            <motion.div
              style={{ display: 'flex', alignItems: 'baseline', gap: 0 }}
              variants={containerVariants}
              initial="hidden"
              animate={phase === 'show' ? 'show' : 'exit'}
            >
              {WORD.split('').map((ch, i) => (
                <motion.span
                  key={i}
                  variants={letterVariants}
                  style={{
                    fontSize: 88,
                    fontWeight: 800,
                    fontFamily: "'Inter', system-ui, sans-serif",
                    lineHeight: 1,
                    letterSpacing: '-3px',
                    color: i < 5 ? '#ffffff' : 'var(--accent2)',
                    display: 'inline-block',
                  }}
                >
                  {ch}
                </motion.span>
              ))}
            </motion.div>

            {/* Subtitle */}
            <motion.p
              variants={subVariants}
              initial="hidden"
              animate={phase === 'show' ? 'show' : 'exit'}
              style={{
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: 11,
                letterSpacing: '4px',
                textTransform: 'uppercase',
                color: 'var(--text3)',
                marginTop: 18,
                textAlign: 'center',
              }}
            >
              Research · Accelerated
            </motion.p>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
