from __future__ import annotations
#from evaluate_externalized_backend import evaluate as evaluate_externalized_backend
#from evaluate_failure_campaign import evaluate as evaluate_failure_campaign
#from evaluate_matching import evaluate as evaluate_matching
#from evaluate_matching_baselines import evaluate as evaluate_matching_baselines
#rom evaluate_overhead import evaluate as evaluate_overhead
#from evaluate_portability import evaluate as evaluate_portability
from evaluation import evaluate_trust_baselines
from evaluation import evaluate_fog_fusion
from evaluation import evaluate_trust_overhead
from evaluation import evaluate_random_scenarios
from evaluation import evaluate_ablation
from evaluation import evaluate_scalability


#def main() -> None:
 #   evaluate_overhead()
  #  evaluate_portability()
   # evaluate_matching()
    #evaluate_matching_baselines()
    #evaluate_failure_campaign()
    #evaluate_externalized_backend()
    #print("All evaluation artefacts refreshed.")


#if __name__ == "__main__":
 #   main()




def main() -> None:
    print("Running trust baseline evaluation...")
    evaluate_trust_baselines.main()

    print("Running fog fusion evaluation...")
    evaluate_fog_fusion.main()

    print("Running random scenario evaluation...")
    evaluate_random_scenarios.main()

    print("Running ablation evaluation...")
    evaluate_ablation.main()

    print("Running scalability evaluation...")
    evaluate_scalability.main()

    print("Running trust overhead evaluation...")
    evaluate_trust_overhead.main()

    print("All trust-aware evaluations completed.")


if __name__ == "__main__":
    main()

# def main() -> None:
#     print("Running trust baseline evaluation...")
#     evaluate_trust_baselines.main()

#     print("Running fog fusion evaluation...")
#     evaluate_fog_fusion.main()

#     print("Running trust overhead evaluation...")
#     evaluate_trust_overhead.main()

#     print("All trust-aware evaluations completed.")


# if __name__ == "__main__":
#     main()
