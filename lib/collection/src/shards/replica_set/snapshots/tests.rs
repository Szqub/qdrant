use std::collections::HashSet;
use std::num::NonZeroU32;
use std::sync::Arc;

use common::ambient::{AmbientContext, AmbientFutureExt};
use common::budget::ResourceBudget;
use common::save_on_disk::SaveOnDisk;
use common::types::DeferredBehavior;
use segment::types::Distance;
use tempfile::{Builder, TempDir};
use tokio::runtime::Handle;
use tokio::sync::{RwLock, oneshot};

use super::*;
use crate::collection::payload_index_schema::PayloadIndexSchema;
use crate::common::adaptive_handle::AdaptiveSearchHandle;
use crate::config::{CollectionConfigInternal, CollectionParams, WalConfig};
use crate::operations::point_ops::{
    PointInsertOperationsInternal, PointOperations, PointStructPersisted, VectorStructPersisted,
};
use crate::operations::shared_storage_config::SharedStorageConfig;
use crate::operations::types::{CountRequestInternal, VectorsConfig};
use crate::operations::vector_params_builder::VectorParamsBuilder;
use crate::operations::{CollectionUpdateOperations, OperationWithClockTag};
use crate::optimizers_builder::OptimizersConfig;
use crate::shards::channel_service::ChannelService;
use crate::shards::replica_set::replica_set_state::ReplicaState;
use crate::shards::replica_set::{AbortShardTransfer, ChangePeerFromState};
use crate::shards::shard::ShardId;
use crate::shards::shard_trait::WaitUntil;

const TEST_COLLECTION_ID: &str = "test_collection";
const TEST_TARGET_SHARD_ID: ShardId = 1;
const TEST_SOURCE_SHARD_ID: ShardId = 2;
const TEST_PEER_ID: PeerId = 1;

/// How long the abandoned recovery stalls before restoring, once it is clear the
/// retry is not going to finish because it is queued behind it.
const STALLED_RECOVERY_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(3);

#[tokio::test(flavor = "multi_thread")]
async fn test_updates_do_not_wait_for_snapshot_clear() {
    let dir = Builder::new()
        .prefix("snapshot-clear-updates")
        .tempdir()
        .unwrap();
    let replica = Arc::new(new_shard_replica_set(&dir, TEST_TARGET_SHARD_ID).await);
    replica
        .set_replica_state(TEST_PEER_ID, ReplicaState::PartialSnapshot)
        .await
        .unwrap();
    let flag = shard_initializing_flag_path(dir.path(), TEST_TARGET_SHARD_ID);
    let (reached_tx, reached_rx) = oneshot::channel();
    let (release_tx, release_rx) = oneshot::channel();
    CLEAR_LOCAL_REPLICA_HOOK
        .lock()
        .unwrap()
        .push((flag.clone(), reached_tx, release_rx));
    let clearing = {
        let replica = Arc::clone(&replica);
        let path = dir.path().to_owned();
        tokio::spawn(async move { replica.clear_local_for_snapshot_recovery(&path).await })
    };
    reached_rx.await.unwrap();
    assert!(flag.exists());
    let update = OperationWithClockTag::from(CollectionUpdateOperations::PointOperation(
        PointOperations::UpsertPoints(PointInsertOperationsInternal::PointsList(vec![
            PointStructPersisted {
                id: 42.into(),
                vector: VectorStructPersisted::Single(vec![0.1, 0.2, 0.3, 0.4]),
                payload: None,
            },
        ])),
    ));
    let result = tokio::time::timeout(std::time::Duration::from_secs(2), async {
        let rejected = replica
            .update_local(update.clone(), WaitUntil::Visible, None, false)
            .measured(AmbientContext::new())
            .await
            .unwrap();
        assert!(rejected.is_none());
        let forced = replica
            .update_local(update, WaitUntil::Visible, None, true)
            .measured(AmbientContext::new())
            .await;
        assert!(forced.is_err());
    })
    .await;
    let _ = release_tx.send(());
    clearing.await.unwrap().unwrap();
    replica.stop_gracefully().await;
    assert!(result.is_ok(), "updates waited for shard cleanup");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_cancel_snapshot_recovery_before_initializing_flag_does_not_mark_dirty() {
    let target_collection_dir = Builder::new()
        .prefix("snapshot-recovery-cancel-target")
        .tempdir()
        .unwrap();
    let source_collection_dir = Builder::new()
        .prefix("snapshot-recovery-cancel-source")
        .tempdir()
        .unwrap();

    let target_replica_set =
        new_shard_replica_set(&target_collection_dir, TEST_TARGET_SHARD_ID).await;
    let source_replica_set =
        new_shard_replica_set(&source_collection_dir, TEST_SOURCE_SHARD_ID).await;

    let source_shard_path = source_collection_dir
        .path()
        .join(TEST_SOURCE_SHARD_ID.to_string());
    assert!(
        LocalShard::check_data(&source_shard_path),
        "test fixture must create a valid source local shard"
    );

    let shard_flag =
        shard_initializing_flag_path(target_collection_dir.path(), TEST_TARGET_SHARD_ID);
    let (reached_tx, reached_rx) = oneshot::channel();
    let (release_tx, release_rx) = oneshot::channel();
    install_restore_local_replica_before_flag_hook(shard_flag.clone(), reached_tx, release_rx);

    let cancel = cancel::CancellationToken::new();
    let restore = target_replica_set.restore_local_replica_from(
        &source_shard_path,
        RecoveryType::Full,
        target_collection_dir.path(),
        cancel.clone(),
    );
    tokio::pin!(restore);

    tokio::select! {
        _ = reached_rx => {}
        result = &mut restore => {
            panic!("snapshot recovery completed before the cancellation window: {result:?}");
        }
    }

    cancel.cancel();
    let _ = release_tx.send(());

    let result = restore.await;
    assert!(
        matches!(result, Err(CollectionError::Cancelled { .. })),
        "recovery should observe cancellation before creating the initializing flag, got {result:?}"
    );
    assert!(
        !shard_flag.exists(),
        "cancellation before destructive restore starts must not leave a false dirty marker"
    );
    assert!(
        !target_replica_set.is_dummy().await,
        "cancellation before destructive restore starts must keep the old local shard installed"
    );

    let local = target_replica_set.local.read().await;
    assert!(
        matches!(local.as_ref(), Some(Shard::Local(_))),
        "the old local shard should remain installed"
    );
    drop(local);

    target_replica_set.stop_gracefully().await;
    source_replica_set.stop_gracefully().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_snapshot_clear_serializes_restore_and_empty_initialization() {
    let dir = Builder::new()
        .prefix("snapshot-clear-lifecycle")
        .tempdir()
        .unwrap();
    let replica = Arc::new(new_shard_replica_set(&dir, TEST_TARGET_SHARD_ID).await);
    replica
        .set_replica_state(TEST_PEER_ID, ReplicaState::PartialSnapshot)
        .await
        .unwrap();
    let flag = shard_initializing_flag_path(dir.path(), TEST_TARGET_SHARD_ID);
    let (_snapshot_dir, snapshot) = new_shard_snapshot().await;
    let (reached_tx, reached_rx) = oneshot::channel();
    let (release_tx, release_rx) = oneshot::channel();
    CLEAR_LOCAL_REPLICA_HOOK
        .lock()
        .unwrap()
        .push((flag.clone(), reached_tx, release_rx));
    let clearing = {
        let replica = Arc::clone(&replica);
        let path = dir.path().to_owned();
        tokio::spawn(async move { replica.clear_local_for_snapshot_recovery(&path).await })
    };
    reached_rx.await.unwrap();
    let mut init = Box::pin(replica.init_empty_local_shard(dir.path()));
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(100), &mut init)
            .await
            .is_err()
    );
    drop(init);
    let mut restore = Box::pin(replica.restore_local_replica_from(
        &snapshot,
        RecoveryType::Full,
        dir.path(),
        cancel::CancellationToken::new(),
    ));
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(100), &mut restore)
            .await
            .is_err()
    );
    let _ = release_tx.send(());
    clearing.await.unwrap().unwrap();
    assert!(restore.await.unwrap());
    assert!(!flag.exists());
    replica.stop_gracefully().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_snapshot_clear_rejects_source_and_preserves_local_on_flag_error() {
    let dir = Builder::new()
        .prefix("snapshot-clear-flag-error")
        .tempdir()
        .unwrap();
    let replica = new_shard_replica_set(&dir, TEST_TARGET_SHARD_ID).await;
    upsert_point(&replica, 7).await;
    let flag = shard_initializing_flag_path(dir.path(), TEST_TARGET_SHARD_ID);
    assert!(
        replica
            .clear_local_for_snapshot_recovery(dir.path())
            .await
            .is_err()
    );
    assert!(!flag.exists());
    assert_eq!(count_points(&replica).await, 1);
    replica
        .set_replica_state(TEST_PEER_ID, ReplicaState::PartialSnapshot)
        .await
        .unwrap();
    fs_err::create_dir(&flag).unwrap();
    assert!(
        replica
            .clear_local_for_snapshot_recovery(dir.path())
            .await
            .is_err()
    );
    assert!(matches!(
        replica.local.read().await.as_ref(),
        Some(Shard::Local(_))
    ));
    replica.stop_gracefully().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_cancel_collection_clear_keeps_holder_and_data_locks_until_cleanup() {
    use crate::collection::Collection;
    use crate::shards::collection_shard_distribution::CollectionShardDistribution;
    use crate::shards::transfer::{ShardTransfer, ShardTransferMethod};

    let config_dir = Builder::new().prefix("clear-config").tempdir().unwrap();
    let fixture = new_shard_replica_set(&config_dir, TEST_SOURCE_SHARD_ID).await;
    let config = fixture.collection_config.read().await.clone();
    fixture.stop_gracefully().await;
    let dir = Builder::new()
        .prefix("clear-cancel-collection")
        .tempdir()
        .unwrap();
    let snapshots = Builder::new()
        .prefix("clear-cancel-snapshots")
        .tempdir()
        .unwrap();
    let collection = Arc::new(
        Collection::new(
            TEST_COLLECTION_ID.to_string(),
            TEST_PEER_ID,
            dir.path(),
            snapshots.path(),
            &config,
            Arc::new(SharedStorageConfig::default()),
            CollectionShardDistribution::all_local(Some(1), TEST_PEER_ID),
            None,
            ChannelService::default(),
            dummy_on_replica_failure(),
            Arc::new(|_| {}),
            dummy_abort_shard_transfer(),
            Some(AdaptiveSearchHandle::current_for_tests()),
            Some(Handle::current()),
            ResourceBudget::default(),
            None,
        )
        .await
        .unwrap(),
    );
    // Reject an unregistered sender before touching data.
    assert!(
        collection
            .clear_local_shard_for_snapshot_recovery(0, Some(2))
            .await
            .is_err()
    );
    let holder = collection.shards_holder();
    let replica = {
        let shards = holder.read().await;
        shards
            .register_start_shard_transfer(ShardTransfer {
                shard_id: 0,
                to_shard_id: None,
                from: 2,
                to: TEST_PEER_ID,
                sync: true,
                method: Some(ShardTransferMethod::Snapshot),
                filter: None,
            })
            .unwrap();
        Arc::clone(shards.get_shard(0).unwrap())
    };
    replica
        .set_replica_state(TEST_PEER_ID, ReplicaState::PartialSnapshot)
        .await
        .unwrap();
    let flag = shard_initializing_flag_path(dir.path(), 0);
    let (reached_tx, reached_rx) = oneshot::channel();
    let (release_tx, release_rx) = oneshot::channel();
    CLEAR_LOCAL_REPLICA_HOOK
        .lock()
        .unwrap()
        .push((flag.clone(), reached_tx, release_rx));
    let caller = {
        let collection = Arc::clone(&collection);
        tokio::spawn(async move {
            collection
                .clear_local_shard_for_snapshot_recovery(0, Some(2))
                .await
        })
    };
    reached_rx.await.unwrap();
    caller.abort();
    assert!(caller.await.unwrap_err().is_cancelled());
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(100), holder.write())
            .await
            .is_err()
    );
    let mut init = Box::pin(replica.init_empty_local_shard(dir.path()));
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(100), &mut init)
            .await
            .is_err()
    );
    let _ = release_tx.send(());
    init.await.unwrap();
    drop(holder.write().await);
    assert!(!flag.exists());
    assert!(LocalShard::check_data(&replica.shard_path));
    replica.stop_gracefully().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_snapshot_clear_sync_error_keeps_old_shard_and_delete_error_keeps_dummy() {
    let dir = Builder::new()
        .prefix("clear-failure-states")
        .tempdir()
        .unwrap();
    let replica = new_shard_replica_set(&dir, TEST_TARGET_SHARD_ID).await;
    upsert_point(&replica, 7).await;
    replica
        .set_replica_state(TEST_PEER_ID, ReplicaState::PartialSnapshot)
        .await
        .unwrap();
    let flag = shard_initializing_flag_path(dir.path(), TEST_TARGET_SHARD_ID);
    CLEAR_FLAG_SYNC_ERRORS.lock().unwrap().push(flag.clone());
    let error = replica
        .clear_local_for_snapshot_recovery(dir.path())
        .await
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("injected initializing flag sync failure")
    );
    assert!(matches!(
        replica.local.read().await.as_ref(),
        Some(Shard::Local(_))
    ));
    assert!(flag.exists());

    replica.stop_gracefully().await;
    // A directory in place of a data file produces a real filesystem remove_file
    // error on both Unix and Windows, without permissions or platform-specific handles.
    let files = shard::files::get_shard_data_files(&replica.shard_path);
    if files.newest_clocks_path.exists() {
        fs_err::remove_file(&files.newest_clocks_path).unwrap();
    }
    fs_err::create_dir(&files.newest_clocks_path).unwrap();
    assert!(
        replica
            .clear_local_for_snapshot_recovery(dir.path())
            .await
            .is_err()
    );
    assert!(matches!(
        replica.local.read().await.as_ref(),
        Some(Shard::Dummy(_))
    ));
    assert!(flag.exists());
    fs_err::remove_dir(&files.newest_clocks_path).unwrap();
    replica.init_empty_local_shard(dir.path()).await.unwrap();
    assert!(!flag.exists());
    assert!(LocalShard::check_data(&replica.shard_path));
    replica.stop_gracefully().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_snapshot_clear_restore_roundtrip_preserves_records() {
    use segment::types::{WithPayloadInterface, WithVector};
    let source_dir = Builder::new()
        .prefix("clear-roundtrip-source")
        .tempdir()
        .unwrap();
    let source = new_shard_replica_set(&source_dir, TEST_SOURCE_SHARD_ID).await;
    let points = (1..=3)
        .map(|id| PointStructPersisted {
            id: id.into(),
            vector: VectorStructPersisted::Single(vec![id as f32, 0.2, 0.3, 0.4]),
            payload: Some(
                serde_json::from_value(
                    serde_json::json!({"label": format!("point-{id}"), "number": id}),
                )
                .unwrap(),
            ),
        })
        .collect();
    source
        .update_local(
            OperationWithClockTag::from(CollectionUpdateOperations::PointOperation(
                PointOperations::UpsertPoints(PointInsertOperationsInternal::PointsList(points)),
            )),
            WaitUntil::Visible,
            None,
            false,
        )
        .measured(AmbientContext::new())
        .await
        .unwrap()
        .unwrap();
    let before = source
        .local_scroll_by_id(
            None,
            10,
            &WithPayloadInterface::Bool(true),
            &WithVector::Bool(true),
            None,
            None,
            None,
            DeferredBehavior::VisibleOnly,
        )
        .measured(AmbientContext::new())
        .await
        .unwrap();
    assert_eq!(before.len(), 3);
    source.force_flush_local_for_test().await;
    source.stop_gracefully().await;
    let dir = Builder::new()
        .prefix("clear-roundtrip-target")
        .tempdir()
        .unwrap();
    let target = new_shard_replica_set(&dir, TEST_TARGET_SHARD_ID).await;
    upsert_point(&target, 99).await;
    target
        .set_replica_state(TEST_PEER_ID, ReplicaState::PartialSnapshot)
        .await
        .unwrap();
    target
        .clear_local_for_snapshot_recovery(dir.path())
        .await
        .unwrap();
    assert!(
        target
            .restore_local_replica_from(
                &source.shard_path,
                RecoveryType::Full,
                dir.path(),
                cancel::CancellationToken::new()
            )
            .await
            .unwrap()
    );
    target
        .set_replica_state(TEST_PEER_ID, ReplicaState::Active)
        .await
        .unwrap();
    let after = target
        .local_scroll_by_id(
            None,
            10,
            &WithPayloadInterface::Bool(true),
            &WithVector::Bool(true),
            None,
            None,
            None,
            DeferredBehavior::VisibleOnly,
        )
        .measured(AmbientContext::new())
        .await
        .unwrap();
    assert_eq!(after, before);
    target.stop_gracefully().await;
}

/// An abandoned snapshot recovery must not roll back the one that replaced it.
///
/// Unserialized, two recoveries of one shard resolve as *last to finish wins*, and the
/// abandoned one wins: it is abandoned because it stalled, so it finishes after the
/// retry it was replaced by, and restores on top of it.
///
/// ```text
/// stalled: clear -> download ..........................-> restore -> rolls back retry
/// retry:      clear -> download -> restore -> Ok to caller -> caller writes
/// ```
#[tokio::test(flavor = "multi_thread")]
async fn test_abandoned_snapshot_recovery_does_not_roll_back_the_retry() {
    let target_collection_dir = Builder::new()
        .prefix("concurrent-recovery-target")
        .tempdir()
        .unwrap();

    let target_replica_set =
        Arc::new(new_shard_replica_set(&target_collection_dir, TEST_TARGET_SHARD_ID).await);

    // `clear_local_for_snapshot_recovery` refuses to run on a source-of-truth replica.
    // The receiving replica of a snapshot transfer sits in `PartialSnapshot`.
    target_replica_set
        .set_replica_state(TEST_PEER_ID, ReplicaState::PartialSnapshot)
        .await
        .unwrap();

    // One downloaded copy of the shard snapshot per recovery: `restore_local_replica_from`
    // moves the data out of the directory it is given, so they cannot share one.
    let (_stalled_snapshot_dir, stalled_snapshot) = new_shard_snapshot().await;
    let (_retry_snapshot_dir, retry_snapshot) = new_shard_snapshot().await;

    let (stalled_cleared_tx, stalled_cleared_rx) = oneshot::channel();
    let (retry_done_tx, retry_done_rx) = oneshot::channel();

    // The recovery that will be abandoned. It clears the shard, then stalls in its
    // download for as long as the retry needs.
    let stalled_recovery = {
        let replica_set = Arc::clone(&target_replica_set);
        let collection_path = target_collection_dir.path().to_path_buf();

        tokio::spawn(async move {
            let _recovery_lock = replica_set.take_snapshot_recovery_lock().await;

            replica_set
                .clear_local_for_snapshot_recovery(&collection_path)
                .await
                .unwrap();

            let _ = stalled_cleared_tx.send(());

            // Stall until the retry is done - or, once recoveries are serialized, until
            // it is clear the retry cannot proceed because it is waiting on this lock.
            let _ = tokio::time::timeout(STALLED_RECOVERY_TIMEOUT, retry_done_rx).await;

            replica_set
                .restore_local_replica_from(
                    &stalled_snapshot,
                    RecoveryType::Full,
                    &collection_path,
                    cancel::CancellationToken::new(),
                )
                .await
                .unwrap()
        })
    };

    // The retry lands while the stalled recovery is in flight.
    stalled_cleared_rx.await.unwrap();

    let retry_recovery = {
        let replica_set = Arc::clone(&target_replica_set);
        let collection_path = target_collection_dir.path().to_path_buf();

        tokio::spawn(async move {
            let _recovery_lock = replica_set.take_snapshot_recovery_lock().await;

            replica_set
                .clear_local_for_snapshot_recovery(&collection_path)
                .await
                .unwrap();

            let restored = replica_set
                .restore_local_replica_from(
                    &retry_snapshot,
                    RecoveryType::Full,
                    &collection_path,
                    cancel::CancellationToken::new(),
                )
                .await
                .unwrap();

            // The retry reported success, so its caller acts on it: the sender switches
            // the transfer to `Partial` and flushes its queue proxy into the shard.
            replica_set
                .set_replica_state(TEST_PEER_ID, ReplicaState::Partial)
                .await
                .unwrap();
            upsert_point(&replica_set, 42).await;

            let _ = retry_done_tx.send(());

            restored
        })
    };

    assert!(
        retry_recovery.await.unwrap(),
        "the retry should have restored the shard"
    );
    assert!(
        stalled_recovery.await.unwrap(),
        "the stalled recovery should have restored the shard"
    );

    assert_eq!(
        count_points(&target_replica_set).await,
        1,
        "the abandoned recovery rolled back the retry that replaced it: the write \
         the caller made after the retry reported success was discarded",
    );

    target_replica_set.stop_gracefully().await;
}

/// Build a valid unpacked shard snapshot to recover from.
///
/// Returns the temp dir - which the caller must keep alive - and the replica path
/// inside it, in the shape `restore_local_replica_from` expects.
async fn new_shard_snapshot() -> (TempDir, std::path::PathBuf) {
    let collection_dir = Builder::new()
        .prefix("concurrent-recovery-snapshot")
        .tempdir()
        .unwrap();

    let replica_set = new_shard_replica_set(&collection_dir, TEST_SOURCE_SHARD_ID).await;
    replica_set.force_flush_local_for_test().await;
    replica_set.stop_gracefully().await;

    let replica_path = collection_dir.path().join(TEST_SOURCE_SHARD_ID.to_string());
    assert!(
        LocalShard::check_data(&replica_path),
        "test fixture must produce a valid shard snapshot"
    );

    (collection_dir, replica_path)
}

async fn upsert_point(replica_set: &ShardReplicaSet, id: u64) {
    let operation = OperationWithClockTag::from(CollectionUpdateOperations::PointOperation(
        PointOperations::UpsertPoints(PointInsertOperationsInternal::PointsList(vec![
            PointStructPersisted {
                id: id.into(),
                vector: VectorStructPersisted::Single(vec![0.1, 0.2, 0.3, 0.4]),
                payload: None,
            },
        ])),
    ));

    replica_set
        .update_local(operation, WaitUntil::Visible, None, false)
        .measured(AmbientContext::new())
        .await
        .expect("failed to upsert point")
        .expect("local shard must be present");
}

async fn count_points(replica_set: &ShardReplicaSet) -> usize {
    replica_set
        .count_local(
            Arc::new(CountRequestInternal {
                filter: None,
                exact: true,
            }),
            None,
            DeferredBehavior::VisibleOnly,
        )
        .measured(AmbientContext::new())
        .await
        .expect("failed to count points")
        .expect("local shard must be present")
        .count
}

fn install_restore_local_replica_before_flag_hook(
    shard_flag: std::path::PathBuf,
    reached: oneshot::Sender<()>,
    release: oneshot::Receiver<()>,
) {
    let mut hook = RESTORE_LOCAL_REPLICA_BEFORE_FLAG_HOOK.lock().unwrap();
    assert!(
        hook.is_none(),
        "restore-local-replica test hook is already installed"
    );
    *hook = Some((shard_flag, reached, release));
}

async fn new_shard_replica_set(collection_dir: &TempDir, shard_id: ShardId) -> ShardReplicaSet {
    let update_runtime = Handle::current();
    let search_runtime = AdaptiveSearchHandle::current_for_tests();

    let wal_config = WalConfig {
        wal_capacity_mb: 1,
        wal_segments_ahead: 0,
        wal_retain_closed: 1,
    };

    let collection_params = CollectionParams {
        vectors: VectorsConfig::Single(VectorParamsBuilder::new(4, Distance::Dot).build()),
        shard_number: NonZeroU32::new(1).unwrap(),
        replication_factor: NonZeroU32::new(1).unwrap(),
        write_consistency_factor: NonZeroU32::new(1).unwrap(),
        ..CollectionParams::empty()
    };

    let optimizers_config = OptimizersConfig::fixture();
    let config = CollectionConfigInternal {
        params: collection_params,
        optimizer_config: optimizers_config.clone(),
        wal_config,
        hnsw_config: Default::default(),
        quantization_config: None,
        strict_mode_config: None,
        uuid: None,
        metadata: None,
        created_at: None,
    };

    let payload_index_schema_file = collection_dir.path().join("payload-schema.json");
    let payload_index_schema: Arc<SaveOnDisk<PayloadIndexSchema>> =
        Arc::new(SaveOnDisk::load_or_init_default(payload_index_schema_file).unwrap());
    let shared_config = Arc::new(RwLock::new(config));

    ShardReplicaSet::build(
        shard_id,
        None,
        TEST_COLLECTION_ID.to_string(),
        TEST_PEER_ID,
        true,
        HashSet::new(),
        dummy_on_replica_failure(),
        dummy_abort_shard_transfer(),
        collection_dir.path(),
        shared_config,
        optimizers_config.clone(),
        Arc::new(SharedStorageConfig::default()),
        payload_index_schema,
        ChannelService::default(),
        update_runtime,
        search_runtime,
        ResourceBudget::default(),
        Some(ReplicaState::Active),
    )
    .await
    .unwrap()
}

fn dummy_on_replica_failure() -> ChangePeerFromState {
    Arc::new(move |_peer_id, _shard_id, _from_state| {})
}

fn dummy_abort_shard_transfer() -> AbortShardTransfer {
    Arc::new(|_shard_transfer, _reason| {})
}
