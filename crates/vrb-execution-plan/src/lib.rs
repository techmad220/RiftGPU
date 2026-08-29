#![forbid(unsafe_code)]

//! Whole-workload routing that accounts for backend transition costs.
//!
//! `vrb-core` intentionally keeps single-operation routing small. This higher
//! layer plans an ordered operation chain so a locally faster kernel does not
//! win when switching backends makes the full workload slower.

use std::collections::BTreeMap;

use thiserror::Error;
use vrb_core::{BackendId, BackendKind, BackendProbe, PerformanceTable, RouteRequest};

#[derive(Debug, Clone, Default)]
pub struct TransitionCostTable {
    costs_us: BTreeMap<(BackendId, BackendId), f64>,
}

impl TransitionCostTable {
    pub fn record(
        &mut self,
        from: BackendId,
        to: BackendId,
        microseconds: f64,
    ) -> Result<(), ExecutionPlanError> {
        validate_cost("transition", microseconds)?;
        self.costs_us.insert((from, to), microseconds);
        Ok(())
    }

    #[must_use]
    pub fn get(&self, from: &BackendId, to: &BackendId) -> Option<f64> {
        self.costs_us.get(&(from.clone(), to.clone())).copied()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlannerConfig {
    pub unmeasured_operator_us: f64,
    pub default_transition_us: f64,
    pub bootstrap_rank_penalty_us: f64,
}

impl Default for PlannerConfig {
    fn default() -> Self {
        Self {
            unmeasured_operator_us: 1_000.0,
            default_transition_us: 50.0,
            bootstrap_rank_penalty_us: 10.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionStep {
    pub request: RouteRequest,
    pub backend: BackendId,
    pub operator_us: f64,
    pub transition_us: f64,
    pub cumulative_us: f64,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ExecutionPlan {
    pub steps: Vec<ExecutionStep>,
    pub estimated_total_us: f64,
}

pub struct CostAwarePlanner {
    config: PlannerConfig,
    transitions: TransitionCostTable,
}

impl CostAwarePlanner {
    pub fn new(
        config: PlannerConfig,
        transitions: TransitionCostTable,
    ) -> Result<Self, ExecutionPlanError> {
        validate_cost("unmeasured operator", config.unmeasured_operator_us)?;
        validate_cost("default transition", config.default_transition_us)?;
        validate_cost("bootstrap rank penalty", config.bootstrap_rank_penalty_us)?;
        Ok(Self {
            config,
            transitions,
        })
    }

    pub fn plan(
        &self,
        requests: &[RouteRequest],
        probes: &[BackendProbe],
        performance: &PerformanceTable,
    ) -> Result<ExecutionPlan, ExecutionPlanError> {
        if requests.is_empty() {
            return Ok(ExecutionPlan::default());
        }

        let mut layers: Vec<BTreeMap<BackendId, State>> = Vec::with_capacity(requests.len());

        for (index, request) in requests.iter().copied().enumerate() {
            let candidates: Vec<&BackendProbe> = probes
                .iter()
                .filter(|probe| probe.available && probe.capabilities.supports(&request))
                .collect();
            if candidates.is_empty() {
                return Err(ExecutionPlanError::NoCompatibleBackend { step: index });
            }

            let mut layer = BTreeMap::new();
            for candidate in candidates {
                let operator_us = self.operator_cost(candidate, request, performance);
                if index == 0 {
                    layer.insert(
                        candidate.id.clone(),
                        State {
                            total_us: operator_us,
                            operator_us,
                            transition_us: 0.0,
                            previous: None,
                        },
                    );
                    continue;
                }

                let previous_layer = layers
                    .last()
                    .expect("a previous planning layer exists for non-zero step");
                let best = previous_layer
                    .iter()
                    .map(|(previous_id, previous_state)| {
                        let transition_us = self.transition_cost(previous_id, &candidate.id);
                        (
                            previous_state.total_us + transition_us + operator_us,
                            transition_us,
                            previous_id,
                        )
                    })
                    .min_by(|left, right| left.0.total_cmp(&right.0))
                    .ok_or(ExecutionPlanError::NoCompatibleBackend { step: index })?;

                layer.insert(
                    candidate.id.clone(),
                    State {
                        total_us: best.0,
                        operator_us,
                        transition_us: best.1,
                        previous: Some(best.2.clone()),
                    },
                );
            }
            layers.push(layer);
        }

        let (final_backend, final_state) = layers
            .last()
            .expect("non-empty requests produce at least one layer")
            .iter()
            .min_by(|left, right| left.1.total_us.total_cmp(&right.1.total_us))
            .ok_or(ExecutionPlanError::NoCompatibleBackend {
                step: requests.len() - 1,
            })?;
        let estimated_total_us = final_state.total_us;
        let mut current = final_backend.clone();
        let mut reversed = Vec::with_capacity(requests.len());

        for index in (0..requests.len()).rev() {
            let state = layers[index]
                .get(&current)
                .expect("reconstruction backend must exist in its planning layer");
            reversed.push(ExecutionStep {
                request: requests[index],
                backend: current.clone(),
                operator_us: state.operator_us,
                transition_us: state.transition_us,
                cumulative_us: state.total_us,
            });
            let Some(previous) = &state.previous else {
                break;
            };
            current = previous.clone();
        }

        reversed.reverse();
        Ok(ExecutionPlan {
            steps: reversed,
            estimated_total_us,
        })
    }

    fn operator_cost(
        &self,
        probe: &BackendProbe,
        request: RouteRequest,
        performance: &PerformanceTable,
    ) -> f64 {
        performance
            .median_us(&probe.id, request.operation, request.data_type)
            .filter(|value| value.is_finite() && *value >= 0.0)
            .unwrap_or_else(|| {
                self.config.unmeasured_operator_us
                    + self.config.bootstrap_rank_penalty_us * f64::from(bootstrap_rank(probe.kind))
            })
    }

    fn transition_cost(&self, from: &BackendId, to: &BackendId) -> f64 {
        if from == to {
            0.0
        } else {
            self.transitions
                .get(from, to)
                .unwrap_or(self.config.default_transition_us)
        }
    }
}

#[derive(Debug, Clone)]
struct State {
    total_us: f64,
    operator_us: f64,
    transition_us: f64,
    previous: Option<BackendId>,
}

fn bootstrap_rank(kind: BackendKind) -> u8 {
    match kind {
        BackendKind::Hip => 0,
        BackendKind::Vulkan => 1,
        BackendKind::Hybrid => 2,
        BackendKind::Plugin => 3,
        BackendKind::Cpu => 4,
    }
}

fn validate_cost(name: &'static str, value: f64) -> Result<(), ExecutionPlanError> {
    if !value.is_finite() || value < 0.0 {
        return Err(ExecutionPlanError::InvalidCost { name, value });
    }
    Ok(())
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum ExecutionPlanError {
    #[error("no compatible backend exists for execution-plan step {step}")]
    NoCompatibleBackend { step: usize },
    #[error("invalid {name} cost {value}")]
    InvalidCost { name: &'static str, value: f64 },
}

#[cfg(test)]
mod tests {
    use super::*;
    use vrb_core::{BackendKind, CapabilitySet, DataType, OperationKind, PerformanceRecord};

    fn probe(id: &str, kind: BackendKind, operations: Vec<OperationKind>) -> BackendProbe {
        BackendProbe {
            id: BackendId::new(id).unwrap(),
            kind,
            name: id.to_owned(),
            vendor: "test".to_owned(),
            available: true,
            device_count: 1,
            detail: "ok".to_owned(),
            capabilities: CapabilitySet {
                operations,
                data_types: vec![DataType::F32],
                external_memory: true,
                external_semaphore: true,
                zero_copy: true,
            },
        }
    }

    #[test]
    fn whole_plan_can_reject_locally_fastest_first_kernel() {
        let hip = probe(
            "hip",
            BackendKind::Hip,
            vec![OperationKind::Gemm, OperationKind::Attention],
        );
        let vulkan = probe(
            "vulkan",
            BackendKind::Vulkan,
            vec![OperationKind::Gemm, OperationKind::Attention],
        );
        let mut performance = PerformanceTable::default();
        for (backend, operation, cost) in [
            ("hip", OperationKind::Gemm, 10.0),
            ("vulkan", OperationKind::Gemm, 20.0),
            ("hip", OperationKind::Attention, 200.0),
            ("vulkan", OperationKind::Attention, 20.0),
        ] {
            performance.record(PerformanceRecord {
                backend: BackendId::new(backend).unwrap(),
                operation,
                data_type: DataType::F32,
                median_microseconds: cost,
                samples: 20,
            });
        }
        let mut transitions = TransitionCostTable::default();
        transitions
            .record(
                BackendId::new("hip").unwrap(),
                BackendId::new("vulkan").unwrap(),
                100.0,
            )
            .unwrap();
        transitions
            .record(
                BackendId::new("vulkan").unwrap(),
                BackendId::new("hip").unwrap(),
                100.0,
            )
            .unwrap();

        let planner = CostAwarePlanner::new(PlannerConfig::default(), transitions).unwrap();
        let plan = planner
            .plan(
                &[
                    RouteRequest::new(OperationKind::Gemm, DataType::F32),
                    RouteRequest::new(OperationKind::Attention, DataType::F32),
                ],
                &[hip, vulkan],
                &performance,
            )
            .unwrap();

        assert_eq!(plan.steps[0].backend.as_str(), "vulkan");
        assert_eq!(plan.steps[1].backend.as_str(), "vulkan");
        assert!((plan.estimated_total_us - 40.0).abs() < f64::EPSILON);
    }

    #[test]
    fn zero_copy_requirement_filters_candidates() {
        let mut cpu = probe("cpu", BackendKind::Cpu, vec![OperationKind::Gemm]);
        cpu.capabilities.zero_copy = false;
        cpu.capabilities.external_memory = false;
        cpu.capabilities.external_semaphore = false;
        let hip = probe("hip", BackendKind::Hip, vec![OperationKind::Gemm]);
        let planner =
            CostAwarePlanner::new(PlannerConfig::default(), TransitionCostTable::default())
                .unwrap();

        let plan = planner
            .plan(
                &[RouteRequest::new(OperationKind::Gemm, DataType::F32).zero_copy()],
                &[cpu, hip],
                &PerformanceTable::default(),
            )
            .unwrap();
        assert_eq!(plan.steps[0].backend.as_str(), "hip");
    }

    #[test]
    fn empty_request_list_produces_empty_plan() {
        let planner =
            CostAwarePlanner::new(PlannerConfig::default(), TransitionCostTable::default())
                .unwrap();
        assert_eq!(
            planner
                .plan(&[], &[], &PerformanceTable::default())
                .unwrap(),
            ExecutionPlan::default()
        );
    }
}
