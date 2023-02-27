//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include <stdlib.h>
#include <set>
#include "../../RaisimGymEnv.hpp"

namespace raisim {

    class ENVIRONMENT : public RaisimGymEnv {

    public:

        explicit ENVIRONMENT(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
                RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable), normDist_(0, 1) {

            /// create world
            world_ = std::make_unique<raisim::World>();

            /// add objects
            anymal_ = world_->addArticulatedSystem(resourceDir_+"/anymal/urdf/anymal.urdf");
            anymal_->setName("anymal");
            anymal_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
            world_->addGround();

            stepHeight_ = float(std::rand()%13)/100+0.12;
            stepWidth_ = float(std::rand()%5)/100+0.25;
            uint8_t stepCount = 20;
            float mass = 1;
            float stairY = 0;
            float stairWidth = 2;
            float stairX = 0.65;
            for (int stepNumber=0; stepNumber < stepCount; ++stepNumber)
            {
                auto box = world_->addBox(stepWidth_, stairWidth, stepHeight_, mass);
                box->setPosition(raisim::Vec<3>{stairX+stepNumber*stepWidth_, stairY, stepHeight_*(stepNumber+0.5)});
                box->setBodyType(raisim::BodyType::STATIC);
            }
            /// get robot data
            gcDim_ = anymal_->getGeneralizedCoordinateDim();
            gvDim_ = anymal_->getDOF();
            nJoints_ = gvDim_ - 6;

            /// initialize containers
            gc_.setZero(gcDim_); gc_init_.setZero(gcDim_);
            gv_.setZero(gvDim_); gv_init_.setZero(gvDim_);
            pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_); pTarget12_.setZero(nJoints_);

            /// this is nominal configuration of anymal
            gc_init_ << 0, 0, 0.50, 1.0, 0.0, 0.0, 0.0, 0.03, 0.4, -0.8, -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8;

            /// set pd gains
            Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
            jointPgain.setZero(); jointPgain.tail(nJoints_).setConstant(50.0);
            jointDgain.setZero(); jointDgain.tail(nJoints_).setConstant(0.2);
            anymal_->setPdGains(jointPgain, jointDgain);
            anymal_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

            /// MUST BE DONE FOR ALL ENVIRONMENTS
            obDim_ = 35;
            actionDim_ = nJoints_; actionMean_.setZero(actionDim_); actionStd_.setZero(actionDim_);
            obDouble_.setZero(obDim_);

            /// action scaling
            actionMean_ = gc_init_.tail(nJoints_);
            double action_std;
            READ_YAML(double, action_std, cfg_["action_std"]) /// example of reading params from the config
            actionStd_.setConstant(action_std);

            /// Reward coefficients
            rewards_.initializeFromConfigurationFile (cfg["reward"]);

            /// indices of links that should not make contact with ground
	    footIdLF_ = 9;
	    footIdRF_ = 13;
	    footIdLB_ = 17;
	    footIdRB_ = 21;
            /// visualize if it is the first environment
            if (visualizable_) {
                server_ = std::make_unique<raisim::RaisimServer>(world_.get());
                server_->launchServer();
                server_->focusOn(anymal_);
            }
        }

        void init() final { }

        void reset() final {
            anymal_->setState(gc_init_, gv_init_);
            updateObservation();
        }

        float step(const Eigen::Ref<EigenVec>& action) final {
            /// action scaling
            pTarget12_ = action.cast<double>();
            pTarget12_ = pTarget12_.cwiseProduct(actionStd_); // тут мы хотим чтобы действие совершалось не до конца
            pTarget12_ += actionMean_;
            pTarget_.tail(nJoints_) = pTarget12_;

            anymal_->setPdTarget(pTarget_, vTarget_);

            for(int i=0; i< int(control_dt_ / simulation_dt_ + 1e-10); i++){
                if(server_) server_->lockVisualizationServerMutex();
                world_->integrate();
                if(server_) server_->unlockVisualizationServerMutex();
            }

            updateObservation();

            raisim::Vec<3> footPositionLF;
            anymal_->getFramePosition(footIdLF_, footPositionLF);
            raisim::Vec<3> footPositionRF;
            anymal_->getFramePosition(footIdRF_, footPositionRF);
            raisim::Vec<3> footPositionLB;
            anymal_->getFramePosition(footIdLB_, footPositionLB);
            raisim::Vec<3> footPositionRB;
            anymal_->getFramePosition(footIdRB_, footPositionRB);

            int dx = 0.481;
            int dy = 0.245;

            float footDeviationY = (gc_[1]+dy-footPositionLF[1])*(gc_[1]+dy-footPositionLF[1])+(gc_[1]-dy-footPositionRF[1])*(gc_[1]-dy-footPositionRF[1])+(gc_[1]+dy-footPositionLB[1])*(gc_[1]+dy-footPositionLB[1])+(gc_[1]-dy-footPositionRB[1])*(gc_[1]-dy-footPositionRB[1]);
            float frontFeetDX = (gc_[0]+1.5*dx-footPositionLF[0])*(gc_[0]+1.5*dx-footPositionLF[0]) + (gc_[0]+1.5*dx-footPositionRF[0])*(gc_[0]+1.5*dx-footPositionRF[0]);
            float frontFeetDZ = (gc_[2]-footPositionLF[2])*(gc_[2]-footPositionLF[2]) + (gc_[2]-footPositionRF[2])*(gc_[2]-footPositionRF[2]);  
            float backFeetDX = (gc_[0]-dx-footPositionLB[0])*(gc_[0]-dx-footPositionLB[0]) + (gc_[0]-dx-footPositionRB[0])*(gc_[0]-dx-footPositionRB[0]);

            std::cout << "x: " << bodyLinearVel_[0] << std::endl;
            std::cout << "y: " << bodyLinearVel_[1] << std::endl;
            std::cout << "z: " << bodyLinearVel_[2] << std::endl;

            rewards_.record("footDeviationY", footDeviationY);
            rewards_.record("frontFeetDX", frontFeetDX);
            rewards_.record("frontFeetDZ", frontFeetDZ);
            rewards_.record("backFeetDX", backFeetDX);
            rewards_.record("torque", anymal_->getGeneralizedForce().squaredNorm());
            rewards_.record("xVelocity", std::min(0.25, bodyLinearVel_[0]));
	        rewards_.record("xAngular", bodyAngularVel_[0]*bodyAngularVel_[0]);
	        rewards_.record("yVelocity", bodyLinearVel_[1]*bodyLinearVel_[1]);
            rewards_.record("yAngularUp", -bodyAngularVel_[1]);
            rewards_.record("yAngularDown", std::max(0.1, bodyAngularVel_[1]));
            rewards_.record("zVelocity", bodyLinearVel_[2]);
            rewards_.record("zAngular", bodyAngularVel_[2]*bodyAngularVel_[2]);

            return rewards_.sum();
        }

        void updateObservation() {
            anymal_->getState(gc_, gv_);
//            raisim::Vec<4> quat;
//            raisim::Mat<3,3> rot;
//            quat[0] = gc_[3]; quat[1] = gc_[4]; quat[2] = gc_[5]; quat[3] = gc_[6];
//            raisim::quatToRotMat(quat, rot);
//            bodyLinearVel_ = rot.e().transpose() * gv_.segment(0, 3);
//            bodyAngularVel_ = rot.e().transpose() * gv_.segment(3, 3);
            obDouble_ <<
                    stepHeight_,
                    stepWidth_,
                    gc_, /// joint angles
                    gv_; /// joint velocity
//                    rot.e().row(2).transpose(), /// body orientation
//                    bodyLinearVel_, bodyAngularVel_, /// body linear&angular velocity

        }

        float getX()
        {
            return std::abs(gc_[0]);
        }
        float getY()
        {
            return std::abs(gc_[1]);
        }
        float getZ()
        {
            return std::abs(gc_[2]);
        }

        void observe(Eigen::Ref<EigenVec> ob) final {
            ob = obDouble_.cast<float>();
        }

        bool isTerminalState(float& terminalReward) final {
            terminalReward = float(terminalRewardCoeff_);

            if (anymal_->getContacts().empty())
                return true;

            terminalReward = 0.0;
            return false;
        }

        void curriculumUpdate() { };

    private:
        int gcDim_, gvDim_, nJoints_;
        bool visualizable_ = false;
        raisim::ArticulatedSystem* anymal_;
        Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_;
        double terminalRewardCoeff_ = -200;
        Eigen::VectorXd actionMean_, actionStd_, obDouble_;
        Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
        float stepHeight_;
        float stepWidth_;

        size_t footIdLF_;
        size_t footIdRF_;
        size_t footIdLB_;
        size_t footIdRB_;
        /// these variables are not in use. They are placed to show you how to create a random number sampler.
        std::normal_distribution<double> normDist_;
        thread_local static std::mt19937 gen_;
    };
    thread_local std::mt19937 raisim::ENVIRONMENT::gen_;

}
