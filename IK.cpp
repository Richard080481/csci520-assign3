#include "IK.h"
#include "FK.h"
#include "minivectorTemplate.h"
#include <Eigen/Dense>
#include <adolc/adolc.h>
#include <cassert>
#if defined(_WIN32) || defined(WIN32)
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#endif
#include <math.h>
using namespace std;

// CSCI 520 Computer Animation and Simulation
// Jernej Barbic and Yijing Li

namespace {
template <typename real> inline real deg2rad(real deg) {
  return deg * M_PI / 180.0;
}

template <typename real>
Mat3<real> Euler2Rotation(const real angle[3], RotateOrder order) {
  Mat3<real> RX = Mat3<real>::getElementRotationMatrix(0, deg2rad(angle[0]));
  Mat3<real> RY = Mat3<real>::getElementRotationMatrix(1, deg2rad(angle[1]));
  Mat3<real> RZ = Mat3<real>::getElementRotationMatrix(2, deg2rad(angle[2]));

  switch (order) {
  case RotateOrder::XYZ:
    return RZ * RY * RX;
  case RotateOrder::YZX:
    return RX * RZ * RY;
  case RotateOrder::ZXY:
    return RY * RX * RZ;
  case RotateOrder::XZY:
    return RY * RZ * RX;
  case RotateOrder::YXZ:
    return RZ * RX * RY;
  case RotateOrder::ZYX:
    return RX * RY * RZ;
  }
  assert(0);
}

template <typename real>
void forwardKinematicsFunction(int numIKJoints, const int *IKJointIDs,
                               const FK &fk,
                               const std::vector<real> &eulerAngles,
                               std::vector<real> &handlePositions) {
  int numJoints = fk.getNumJoints();

  std::vector<Mat3<real>> localRotations(numJoints);
  std::vector<Mat3<real>> globalRotations(numJoints);
  std::vector<Vec3<real>> globalTranslations(numJoints);

  for (int i = 0; i < numJoints; i++) {
    int jointID = fk.getJointUpdateOrder(i);

    real angles[3] = {eulerAngles[jointID * 3 + 0],
                      eulerAngles[jointID * 3 + 1],
                      eulerAngles[jointID * 3 + 2]};
    Mat3<real> R_user = Euler2Rotation(angles, fk.getJointRotateOrder(jointID));

    Vec3d orient = fk.getJointOrient(jointID);
    real oAngles[3] = {real(orient[0]), real(orient[1]), real(orient[2])};
    Mat3<real> R_orient = Euler2Rotation(oAngles, RotateOrder::XYZ);

    localRotations[jointID] = R_orient * R_user;

    Vec3d restT = fk.getJointRestTranslation(jointID);
    Vec3<real> localT = {real(restT[0]), real(restT[1]), real(restT[2])};

    int parentID = fk.getJointParent(jointID);
    if (parentID < 0) {
      globalRotations[jointID] = localRotations[jointID];
      globalTranslations[jointID] = localT;
    } else {
      multiplyAffineTransform4ds(
          globalRotations[parentID], globalTranslations[parentID],
          localRotations[jointID], localT, globalRotations[jointID],
          globalTranslations[jointID]);
    }
  }

  for (int i = 0; i < numIKJoints; i++) {
    int jointID = IKJointIDs[i];
    handlePositions[i * 3 + 0] = globalTranslations[jointID][0];
    handlePositions[i * 3 + 1] = globalTranslations[jointID][1];
    handlePositions[i * 3 + 2] = globalTranslations[jointID][2];
  }
}

} // end anonymous namespace

IK::IK(int numIKJoints, const int *IKJointIDs, FK *inputFK, int adolc_tagID) {
  this->numIKJoints = numIKJoints;
  this->IKJointIDs = IKJointIDs;
  this->fk = inputFK;
  this->adolc_tagID = adolc_tagID;

  FKInputDim = fk->getNumJoints() * 3;
  FKOutputDim = numIKJoints * 3;

  train_adolc();
}

void IK::train_adolc() {
  int n = FKInputDim;
  int m = FKOutputDim;

  std::vector<adouble> eulerAnglesAD(n);
  std::vector<adouble> handlePositionsAD(m);
  std::vector<double> eulerAnglesD(n, 0.0);

  trace_on(adolc_tagID);

  for (int i = 0; i < n; i++)
    eulerAnglesAD[i] <<= eulerAnglesD[i];

  forwardKinematicsFunction(numIKJoints, IKJointIDs, *fk, eulerAnglesAD,
                            handlePositionsAD);

  for (int i = 0; i < m; i++) {
    double ignored;
    handlePositionsAD[i] >>= ignored;
  }

  trace_off();
}

void IK::doIK(const Vec3d *targetHandlePositions, Vec3d *jointEulerAngles) {
  int numJoints = fk->getNumJoints();
  int n = FKInputDim;
  int m = FKOutputDim;

  std::vector<double> eulerAnglesD(n);
  for (int i = 0; i < numJoints; i++) {
    eulerAnglesD[i * 3 + 0] = jointEulerAngles[i][0];
    eulerAnglesD[i * 3 + 1] = jointEulerAngles[i][1];
    eulerAnglesD[i * 3 + 2] = jointEulerAngles[i][2];
  }

  std::vector<double> handlePos(m);
  ::function(adolc_tagID, m, n, eulerAnglesD.data(), handlePos.data());

  std::vector<double> J(m * n);
  std::vector<double *> Jrows(m);
  for (int i = 0; i < m; i++)
  {
    Jrows[i] = &J[i * n];
  }
  ::jacobian(adolc_tagID, m, n, eulerAnglesD.data(), Jrows.data());

    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMajorMatrixXd;
    Eigen::MatrixXd Jmat = Eigen::Map<RowMajorMatrixXd>(J.data(), m, n);

  Eigen::VectorXd deltaP(m);
  for (int i = 0; i < numIKJoints; i++) {
    deltaP[i * 3 + 0] = targetHandlePositions[i][0] - handlePos[i * 3 + 0];
    deltaP[i * 3 + 1] = targetHandlePositions[i][1] - handlePos[i * 3 + 1];
    deltaP[i * 3 + 2] = targetHandlePositions[i][2] - handlePos[i * 3 + 2];
  }

  // Tikhonov: delta_theta = (J^T J + alpha I)^-1 J^T delta_p
  const double alpha = 1e-3;
  Eigen::MatrixXd JtJ = Jmat.transpose() * Jmat;
  JtJ += alpha * Eigen::MatrixXd::Identity(n, n);
  Eigen::VectorXd deltaTheta = JtJ.ldlt().solve(Jmat.transpose() * deltaP);

  for (int i = 0; i < numJoints; i++) {
    jointEulerAngles[i][0] += deltaTheta[i * 3 + 0];
    jointEulerAngles[i][1] += deltaTheta[i * 3 + 1];
    jointEulerAngles[i][2] += deltaTheta[i * 3 + 2];
  }
}