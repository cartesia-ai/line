class SnapshotMap<T> {
  private snapshots: Array<Map<string, T>>;

  constructor() {
    this.snapshots = [new Map<string, T>()];
  }

  snapshot(): number {
    const newSnapshot = new Map<string, any>();
    this.snapshots.push(newSnapshot);
    return this.snapshots.length - 1;
  }

  get(key: string, snapshotId: null | number): any {
    if (snapshotId !== null && (snapshotId < 0 || snapshotId >= this.snapshots.length)) {
      throw new Error("Invalid snapshot ID");
    }
    snapshotId = snapshotId === null ? this.snapshots.length - 1 : snapshotId;

    const snapshot = this.snapshots
      .slice(0, snapshotId + 1).reverse()
      .find(snap => snap.has(key));
    return snapshot ? snapshot.get(key) : undefined;
  }
  
  set(key: string, value: any): void {
    this.snapshots[this.snapshots.length - 1].set(key, value);
  }

}
