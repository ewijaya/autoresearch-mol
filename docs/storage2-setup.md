# storage2 EBS Volume Setup

**Purpose:** Additional EBS volume to store the NLP control dataset (FineWeb-Edu / climbmix-400b-shuffle) for Phase 2 Track C experiments (see `docs/phase-prompts.md`).

**Why:** `storage1` (100GB, 69% used, 29GB free) cannot fit the climbmix-400b-shuffle dataset (~600GB). A dedicated volume avoids competing for space with molecular data and code.

**Minimum size:** 700GB (600GB dataset + ~100GB headroom for processed tokens and intermediate files).

---

## Current Instance

| Property | Value |
|----------|-------|
| Instance ID | `i-0620c2546bd7f9322` |
| Availability Zone | `ap-northeast-1c` |
| Region | `ap-northeast-1` |
| Existing storage1 | `/dev/nvme1n1` → `/home/ubuntu/storage1` (100GB, ext4) |

---

## Step 1: Create the EBS Volume

```bash
aws ec2 create-volume \
  --availability-zone ap-northeast-1c \
  --size 700 \
  --volume-type gp3 \
  --tag-specifications 'ResourceType=volume,Tags=[{Key=Name,Value=storage2-recursive-mol}]'
```

Note the `VolumeId` from the output (e.g., `vol-0abc123...`).

Wait for the volume to become available:

```bash
aws ec2 describe-volumes --volume-ids <VOLUME_ID> --query 'Volumes[0].State'
```

## Step 2: Attach the Volume

```bash
aws ec2 attach-volume \
  --volume-id <VOLUME_ID> \
  --instance-id i-0620c2546bd7f9322 \
  --device /dev/xvdf
```

Wait for attachment:

```bash
aws ec2 describe-volumes --volume-ids <VOLUME_ID> --query 'Volumes[0].Attachments[0].State'
```

## Step 3: Identify the Device

After attaching, the device name may differ from `/dev/xvdf` on NVMe instances. Find it:

```bash
lsblk
```

Look for the new ~700GB disk (e.g., `/dev/nvme3n1`). Use that device name in the steps below.

## Step 4: Format the Volume

```bash
sudo mkfs.ext4 /dev/nvme3n1
```

## Step 5: Mount the Volume

```bash
sudo mkdir -p /home/ubuntu/storage2
sudo mount /dev/nvme3n1 /home/ubuntu/storage2
sudo chown ubuntu:ubuntu /home/ubuntu/storage2
```

## Step 6: Persist the Mount (survives reboot)

Get the UUID:

```bash
sudo blkid /dev/nvme3n1
```

Add to `/etc/fstab`:

```bash
echo "UUID=<UUID_FROM_BLKID> /home/ubuntu/storage2 ext4 defaults,nofail 0 2" | sudo tee -a /etc/fstab
```

## Step 7: Verify

```bash
df -h /home/ubuntu/storage2
# Should show ~687GB available
```

---

## Cost Estimate

- gp3 700GB in ap-northeast-1: ~$56/month ($0.08/GB/month)
- **This volume is temporary** — delete after the project completes (target: May 2026 post-NeurIPS submission)
- Needed from Phase 2 (Mar 16) through Phase 5 (Apr 20) = ~5 weeks → **~$70 total**

## Cleanup: Delete the Volume

After the project is done, unmount and delete:

```bash
# Unmount
sudo umount /home/ubuntu/storage2

# Remove fstab entry
sudo sed -i '\|/home/ubuntu/storage2|d' /etc/fstab

# Detach from instance
aws ec2 detach-volume --volume-id <VOLUME_ID>

# Wait for detachment, then delete
aws ec2 delete-volume --volume-id <VOLUME_ID>
```
