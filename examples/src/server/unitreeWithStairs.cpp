// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.

#include "raisim/RaisimServer.hpp"
#include "raisim/World.hpp"

int main(int argc, char* argv[]) {
  auto binaryPath = raisim::Path::setFromArgv(argv[0]);

  /// create raisim world
  raisim::World world;
  world.setTimeStep(0.001);
  Eigen::Vector3d gravity;
  gravity << 0, 0, 0;
  world.setGravity(gravity);
  /// create objects
  auto ground = world.addGround();
  std::srand(std::time(NULL));

    float stepHeight_ = float(std::rand()%8)/100+0.12;
    float stepWidth_ = float(std::rand()%5)/100+0.25;
    uint8_t stepCount = 15;
    float mass = 1;
    float stairY = 0;
    float stairWidth = 1.5;
    float stairX = 0.25+stepWidth_/2;
    for (int stepNumber=0; stepNumber < stepCount; ++stepNumber)
    {
        auto box = world.addBox(stepWidth_, stairWidth, stepHeight_, mass);
        box->setPosition(raisim::Vec<3>{stairX+stepNumber*stepWidth_, stairY, stepHeight_*(stepNumber+0.5)});
        box->setBodyType(raisim::BodyType::STATIC);
    }

  ground->setAppearance("steel");
  auto aliengo = world.addArticulatedSystem(binaryPath.getDirectory() + "\\rsc\\aliengo\\aliengo.urdf");
  aliengo->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

    //x, y, z, gain, rxl, ryd, rzl,
  /// aliengo joint PD controller
  Eigen::VectorXd jointNominalConfigStart(aliengo->getGeneralizedCoordinateDim()), jointNominalConfigTarget(aliengo->getGeneralizedCoordinateDim()), jointVelocityTarget(aliengo->getDOF());
  jointNominalConfigStart <<3*stepWidth_/4, 0.0, 0.57, 1.0, 0.0, -1.5*atan(stepHeight_/stepWidth_)/3.14, 0.0, 0.0, 0.7, -0.6, 0.0, 0.4, -1.2, 0.0, 1.2, -0.8, 0.0, 1.2, -0.8; //одна лапа на второй ступени
//    jointNominalConfigStart <<0.0, 0.0, 0.345, 1.0, 0.0, -atan(stepHeight_/stepWidth_)/3.14/3, 0.0, 0.0, 0.7, -1.0, 0.0, 0.4, -1.9, 0.0, 0.9, -1.3, 0.0, 0.9, -1.3; //одна лапа на первой ступени
//    jointNominalConfigStart << stepWidth_/2, 0.0, 0.4, 1.0, 0.0, -atan(stepHeight_/stepWidth_)/3.14/2, 0.0, 0.0, 0.9, -1.9, 0.0, 0.9, -1.9, 0.0, 0.8, -0.9, 0.0, 0.8, -0.9; //нижняя стойка на первой ступени
//    jointNominalConfigStart  << stepWidth_, 0.0, 0.45, 1.0, 0.0, -atan(stepHeight_/stepWidth_)/3.14, 0.0, 0.0, 1.1, -1.3, 0.0, 1.1, -1.3, 0.0, 0.9, -0.7, 0.0, 0.9, -0.7; //высокая стойка на первой ступени
//    jointNominalConfigStart <<3*stepWidth_/4, 0.0, 0.48, 1.0, 0.0, -1.7*atan(stepHeight_/stepWidth_)/3.14, 0.0, 0.0, 0.8, -0.7, 0.0, 0.1, -1.0, 0.0, 1.2, -0.8, 0.0, 1.2, -0.8; //одна лапа на второй ступени

    jointNominalConfigTarget << 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  jointVelocityTarget << 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05;
//  jointVelocityTarget.setZero();jointNominalConfigStart

  Eigen::VectorXd jointPgain(aliengo->getDOF()), jointDgain(aliengo->getDOF());
  jointPgain.tail(12).setConstant(100.0);
  jointDgain.tail(12).setConstant(1.0);
  aliengo->setGeneralizedCoordinate(jointNominalConfigStart);
  aliengo->setGeneralizedForce(Eigen::VectorXd::Zero(aliengo->getDOF()));
  aliengo->setPdGains(jointPgain, jointDgain);
  aliengo->setPdTarget(jointNominalConfigStart, jointVelocityTarget);
  aliengo->setName("aliengo");

  /// launch raisim server
  raisim::RaisimServer server(&world);
  server.focusOn(aliengo);
  server.launchServer(8085);

    int footIdLF_ = 9;
    int footIdRF_ = 12;
    int footIdLB_ = 15;
    int footIdRB_ = 18;

//    auto aw = aliengo->getActuationLowerLimits();
//    Eigen::VectorXd limits(18);
//    limits << 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10.;
//    aliengo->setActuationLimits(limits, -limits);
//    auto bw = aliengo->getActuationLowerLimits();

  for (int i=0; i<2000000; i++) {
    RS_TIMED_LOOP(int(world.getTimeStep()*1e6))
    server.integrateWorldThreadSafe();
    aliengo->getState(jointNominalConfigStart, jointVelocityTarget);

    auto a = aliengo->getContacts();
//    std::cout << a.size() << std::endl;
//    std::cout << "gc" << jointNominalConfigTarget[5]/jointNominalConfigTarget[3] << std::endl;
//    std::cout << "ex" << -2*atan(stepHeight_/stepWidth_)/3.1415 << std::endl;
//    std::cout << jointNominalConfigTarget.head(3) << std::endl;

    auto b = aliengo->getFrames();
    raisim::Vec<3> footPositionLF;
    aliengo->getFramePosition(footIdLF_, footPositionLF);
    raisim::Vec<3> footPositionRF;
    aliengo->getFramePosition(footIdRF_, footPositionRF);
    raisim::Vec<3> footPositionLB;
    aliengo->getFramePosition(footIdLB_, footPositionLB);
    raisim::Vec<3> footPositionRB;
    aliengo->getFramePosition(footIdRB_, footPositionRB);
      std::cout << "jointVelocityTarget: " << jointVelocityTarget << std::endl;
  }

  std::cout<<"total mass "<<aliengo->getCompositeMass()[0]<<std::endl;

  server.killServer();
}
