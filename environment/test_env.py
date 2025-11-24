"""
Script de test pour vérifier l'environnement sans entraînement
Usage: python test_env.py --steps 100
"""

import argparse
import numpy as np
from environment.mh_env import MonsterHunterEnv


def test_environment(n_steps=100, grayscale=False):
    """
    Teste l'environnement avec actions aléatoires

    Vérifie :
    - Pas de NaN/Inf
    - Rewards raisonnables
    - Observations valides
    """
    print("\n" + "=" * 70)
    print("🧪 TEST ENVIRONNEMENT - Actions Aléatoires")
    print("=" * 70)

    # Créer env
    env = MonsterHunterEnv(
        use_vision=True,
        use_memory=True,
        grayscale=grayscale,
        frame_stack=4,
        use_controller=True,
        use_advanced_rewards=True,
        auto_reload_save_state=False  # Désactiver pour test
    )

    print(f"\n✅ Environnement créé")
    print(f"   Actions: {env.action_space.n}")
    print(f"   Observation: {env.observation_space}")

    # Reset
    print(f"\n🔄 Reset...")
    obs, info = env.reset()

    # Vérifier obs
    if isinstance(obs, dict):
        for key, val in obs.items():
            if np.any(np.isnan(val)):
                print(f"❌ NaN détecté dans obs[{key}]")
                return False
            if np.any(np.isinf(val)):
                print(f"❌ Inf détecté dans obs[{key}]")
                return False

    print(f"✅ Reset OK")

    # Test steps
    print(f"\n🎮 Test {n_steps} steps avec actions aléatoires...\n")

    episode_rewards = []
    current_episode_reward = 0.0

    nan_count = 0
    inf_count = 0
    extreme_reward_count = 0

    for step in range(n_steps):
        # Action aléatoire
        action = env.action_space.sample()

        # Step
        obs, reward, terminated, truncated, info = env.step(action)
        current_episode_reward += reward

        # Vérifications
        if np.isnan(reward):
            nan_count += 1
            print(f"❌ Step {step}: NaN reward!")

        if np.isinf(reward):
            inf_count += 1
            print(f"❌ Step {step}: Inf reward!")

        if abs(reward) > 100:
            extreme_reward_count += 1
            print(f"⚠️ Step {step}: Reward extrême = {reward:.2f}")

        # Vérifier obs
        if isinstance(obs, dict):
            for key, val in obs.items():
                if np.any(np.isnan(val)):
                    print(f"❌ Step {step}: NaN dans obs[{key}]")
                if np.any(np.isinf(val)):
                    print(f"❌ Step {step}: Inf dans obs[{key}]")

        # Affichage périodique
        if (step + 1) % 10 == 0:
            hp = info.get('hp', 'N/A')
            stamina = info.get('stamina', 'N/A')
            zone = info.get('current_zone', 'N/A')

            print(f"Step {step + 1}/{n_steps}: "
                  f"Reward={reward:+.2f}, "
                  f"HP={hp}, "
                  f"Stamina={stamina}, "
                  f"Zone={zone}")

        # Reset si épisode terminé
        if terminated or truncated:
            episode_rewards.append(current_episode_reward)

            print(f"\n📊 Épisode terminé:")
            print(f"   Reward totale: {current_episode_reward:.2f}")
            print(f"   Longueur: {info.get('episode_steps', 'N/A')} steps")
            print(f"   Morts: {info.get('death_count', 'N/A')}\n")

            obs, info = env.reset()
            current_episode_reward = 0.0

    # Résumé
    print("\n" + "=" * 70)
    print("📊 RÉSUMÉ DU TEST")
    print("=" * 70)

    print(f"\n✅ Test terminé: {n_steps} steps")
    print(f"\nErreurs détectées:")
    print(f"   NaN rewards: {nan_count}")
    print(f"   Inf rewards: {inf_count}")
    print(f"   Rewards extrêmes (|r| > 100): {extreme_reward_count}")

    if episode_rewards:
        print(f"\nÉpisodes complétés: {len(episode_rewards)}")
        print(f"   Reward moyenne: {np.mean(episode_rewards):.2f}")
        print(f"   Reward min: {np.min(episode_rewards):.2f}")
        print(f"   Reward max: {np.max(episode_rewards):.2f}")

    env.close()

    # Verdict
    success = (nan_count == 0 and inf_count == 0)

    if success:
        print("\n✅ TEST RÉUSSI - Environnement stable")
    else:
        print("\n❌ TEST ÉCHOUÉ - Problèmes détectés")

    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test environnement MH')
    parser.add_argument('--steps', type=int, default=100,
                        help='Nombre de steps à tester')
    parser.add_argument('--grayscale', action='store_true',
                        help='Utiliser grayscale')

    args = parser.parse_args()

    success = test_environment(
        n_steps=args.steps,
        grayscale=args.grayscale
    )

    exit(0 if success else 1)