export type Role = 'adjuster' | 'admin' | 'viewer';

export interface User {
  id: string;
  email: string;
  passwordHash: string;
  role: Role;
  createdAt: Date;
}

export type ClaimStatus = 'draft' | 'submitted' | 'in_review' | 'approved' | 'denied';

export interface Claim {
  id: string;
  policyNumber: string;
  claimantId: string;
  status: ClaimStatus;
  /** Estimated payout, in integer cents. See README conventions. */
  estimateCents: number;
  submittedAt: Date;
}
