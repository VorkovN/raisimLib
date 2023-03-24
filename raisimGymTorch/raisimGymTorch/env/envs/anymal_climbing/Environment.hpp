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

        explicit ENVIRONMENT(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable, int seed = 0) :
                RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable), normDist_(0, 1), seed_(seed) {

            /// create world
            world_ = std::make_unique<raisim::World>();

            /// add objects
            anymal_ = world_->addArticulatedSystem(resourceDir_+"/aliengo/aliengo.urdf");
            anymal_->setName("anymal");
            anymal_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
            world_->addGround();

            srand (seed_);
            stepHeight_ = float(std::rand()%8)/100+0.12;
            stepWidth_ = float(std::rand()%5)/100+0.25;
            uint8_t stepCount = 10;
            float mass = 1;
            float stairY = 0;
            float stairWidth = 1.5;
            float stairX = dx_+stepWidth_/2;
            for (int stepNumber=0; stepNumber < stepCount; ++stepNumber)
            {
                auto box = world_->addBox(stepWidth_, stairWidth, stepHeight_, mass);
                box->setPosition(raisim::Vec<3>{stairX+stepNumber*stepWidth_, stairY, stepHeight_*(stepNumber+0.5)});
                box->setBodyType(raisim::BodyType::STATIC);
            }
            targetAngularY_ = -2*atan(stepHeight_/stepWidth_)/3.14*1.2;

            /// get robot data
            gcDim_ = anymal_->getGeneralizedCoordinateDim();
            gvDim_ = anymal_->getDOF();
            nJoints_ = gvDim_ - 6;

            /// initialize containers
            gc_.setZero(gcDim_); gc_init_.setZero(gcDim_);
            gv_.setZero(gvDim_); gv_init_.setZero(gvDim_);
            pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_); pTarget12_.setZero(nJoints_);

            switch (seed_) {
                case 0:
                    gc_init_ << 0.0, 0.0, 0.38, 1.0, 0.0, 0.0, 0.0, 0.0, 0.9, -1.5, 0.0, 0.9, -1.5, 0.0, 0.9, -1.5, 0.0, 0.9, -1.5; //обычная стойка
                    break;
                case 1:
                    gc_init_ << 0.0, 0.0, 0.43, 1.0, 0.0, -atan(stepHeight_/stepWidth_)/3.14/3, 0.0, 0.0, 0.7, -1.0, 0.0, 0.4, -1.9, 0.0, 0.9, -1.3, 0.0, 0.9, -1.3; //одна лапа на первой ступени
                    break;
                case 2:
                    gc_init_ << stepWidth_/2, 0.0, 0.477, 1.0, 0.0, -atan(stepHeight_/stepWidth_)/3.14/4, 0.0, 0.0, 0.9, -1.9, 0.0, 0.9, -1.9, 0.0, 0.8, -0.9, 0.0, 0.8, -0.9; //нижняя стойка на первой ступени
                    break;
                case 3:
                    gc_init_  << stepWidth_, 0.0, 0.537, 1.0, 0.0, -atan(stepHeight_/stepWidth_)/3.14, 0.0, 0.0, 0.9, -1.1, 0.0, 0.9, -1.1, 0.0, 0.9, -0.9, 0.0, 0.9, -0.9; //высокая стойка на первой ступени
                    break;
                case 4:
                    gc_init_ << 3*stepWidth_/4, 0.0, 0.57, 1.0, 0.0, -1.5*atan(stepHeight_/stepWidth_)/3.14, 0.0, 0.0, 0.7, -0.6, 0.0, 0.4, -1.2, 0.0, 1.2, -0.8, 0.0, 1.2, -0.8; //одна лапа на второй ступени
                    break;
            }

            /// set pd gains
            Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
            jointPgain.setZero(); jointPgain.tail(nJoints_).setConstant(55.0);
            jointDgain.setZero(); jointDgain.tail(nJoints_).setConstant(2.0);
            anymal_->setPdGains(jointPgain, jointDgain);
            anymal_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

            /// MUST BE DONE FOR ALL ENVIRONMENTS
            obDim_ = 38;
//            obDim_ = 20;
            actionDim_ = nJoints_; actionMean_.setZero(actionDim_); actionStd_.setZero(actionDim_);
            obDouble_.setZero(obDim_);
            /// action scaling
            actionMean_ = gc_init_.tail(nJoints_);
            double action_std;
            READ_YAML(double, action_std, cfg_["action_std"]) /// example of reading params from the config
            actionStd_.setConstant(action_std);

//            Eigen::VectorXd limits(18);
//            limits << 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20.;
//            anymal_->setActuationLimits(limits, -limits);
//            auto bw = anymal_->getActuationLowerLimits();
            
            /// Reward coefficients
            rewards_.initializeFromConfigurationFile (cfg["reward"]);

            /// visualize if it is the first environment
            if (visualizable_) {
                server_ = std::make_unique<raisim::RaisimServer>(world_.get());
                server_->launchServer(8080+seed_);
                server_->focusOn(anymal_);
            }
        }

        void init() final { }

        float normalizeReward(float x, float coeff) {
            return (1./coeff-x*x);
        }

        void reset() final {
            anymal_->setState(gc_init_, gv_init_);
            updateObservation();
        }

        float step(const Eigen::Ref<EigenVec>& action) final {
/// action scaling
            pTarget12_ = action.cast<double>();
//            pTarget12_ = pTarget12_.cwiseProduct(actionStd_); // тут мы хотим чтобы действие совершалось не до конца
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

            float footDeviationY = normalizeReward((gc_[7]), rewards_.rewards_["footDeviationY"].coefficient)/4
                                 + normalizeReward((gc_[10]), rewards_.rewards_["footDeviationY"].coefficient)/4
                                 + normalizeReward((gc_[13]), rewards_.rewards_["footDeviationY"].coefficient)/4
                                 + normalizeReward((gc_[16]), rewards_.rewards_["footDeviationY"].coefficient)/4;
            rewards_.record("footDeviationY", footDeviationY);

            float frontFeetDX = normalizeReward(gc_[0]+1.5*dx_-footPositionLF[0], rewards_.rewards_["frontFeetDX"].coefficient)
                              + normalizeReward(gc_[0]+1.5*dx_-footPositionRF[0], rewards_.rewards_["frontFeetDX"].coefficient);
            rewards_.record("frontFeetDX", frontFeetDX);

            float frontFeetDZ = normalizeReward(gc_[2]-footPositionLF[2], rewards_.rewards_["frontFeetDZ"].coefficient)
                              + normalizeReward(gc_[2]-footPositionRF[2], rewards_.rewards_["frontFeetDZ"].coefficient);
            rewards_.record("frontFeetDZ", frontFeetDZ);

            float backFeetDX = normalizeReward(gc_[0]-dx_-footPositionLB[0], rewards_.rewards_["backFeetDX"].coefficient)
                             + normalizeReward(gc_[0]-dx_-footPositionRB[0], rewards_.rewards_["backFeetDX"].coefficient);
            rewards_.record("backFeetDX", backFeetDX);

            float backFeetDZ = normalizeReward(gc_[2]-footPositionLB[2], rewards_.rewards_["backFeetDZ"].coefficient)
                             + normalizeReward(gc_[2]-footPositionRB[2], rewards_.rewards_["backFeetDZ"].coefficient);
            rewards_.record("backFeetDZ", backFeetDZ);

            float torque = anymal_->getGeneralizedForce().squaredNorm();
            rewards_.record("torque", torque);

            float xVelocity = std::min(3.0, gv_[0]);
            rewards_.record("xVelocity", xVelocity);

            float xAngular = normalizeReward(gv_[3], rewards_.rewards_["xAngular"].coefficient);
            rewards_.record("xAngular", xAngular);

            float yVelocity = normalizeReward(gv_[1], rewards_.rewards_["yVelocity"].coefficient);
            rewards_.record("yVelocity", yVelocity);

            float yAngular = normalizeReward(gc_[5]/gc_[3]-targetAngularY_, rewards_.rewards_["yAngular"].coefficient);
            rewards_.record("yAngular", yAngular);

            float zVelocity = std::min(3.0, gv_[2]);
            rewards_.record("zVelocity", zVelocity);

            float zAngular = normalizeReward(gv_[5], rewards_.rewards_["zAngular"].coefficient);
            rewards_.record("zAngular", zAngular);

            float xCoord = gc_[0];
            rewards_.record("xCoord", xCoord, true);

            float zCoord = gc_[2];
            rewards_.record("zCoord", zCoord, true);

            static int counter = 0;
            if (counter++ % 98765 == 1)
            {
                std::cout << "CURRENT REWARDS:\n";
                std::cout << "footDeviationY: " << rewards_["footDeviationY"] << "\n";
                std::cout << "frontFeetDX: " << rewards_["frontFeetDX"] << "\n";
                std::cout << "frontFeetDZ: " << rewards_["frontFeetDZ"] << "\n";
                std::cout << "backFeetDX: " << rewards_["backFeetDX"] << "\n";
                std::cout << "backFeetDZ: " << rewards_["backFeetDZ"] << "\n";
                std::cout << "torque: " << rewards_["torque"] << "\n";
                std::cout << "xVelocity: " << rewards_["xVelocity"] << "\n";
                std::cout << "xAngular: " << rewards_["xAngular"] << "\n";
                std::cout << "yVelocity: " << rewards_["yVelocity"] << "\n";
                std::cout << "yAngular: " << rewards_["yAngular"] << "\n";
                std::cout << "zVelocity: " << rewards_["zVelocity"] << "\n";
                std::cout << "zAngular: " << rewards_["zAngular"] << "\n";
                std::cout << "xCoord: " << rewards_["xCoord"] << "\n";
                std::cout << "zCoord: " << rewards_["zCoord"] << "\n";

                std::cout << "ACTIONS:\n";
                std::cout << action[0] << " " << action[1] << " " << action[2] << " " << action[3] << " "
                          << action[4] << " " << action[5] << " " << action[6] << " " << action[7] << " "
                          << action[8] << " " << action[9] << " " << action[10] << " " << action[11] << "\n";
                std::cout.flush();
            }

            return rewards_.sum();
        }

        void updateObservation() {
            anymal_->getState(gc_, gv_);
            obDouble_ << stepHeight_, stepWidth_, gv_, gc_.head(3), gc_[4]/gc_[3], gc_[5]/gc_[3], gc_[6]/gc_[3], gc_.tail(12);
//            obDouble_ << stepHeight_, stepWidth_, gc_.head(3), gc_[4]/gc_[3], gc_[5]/gc_[3], gc_[6]/gc_[3], gc_.tail(12);
            actionMean_ = gc_;
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

            bool isFootContactLF = false;
            bool isFootContactRF = false;
            bool isFootContactLB = false;
            bool isFootContactRB = false;

            for(auto& contact: anymal_->getContacts())
            {

                if (contact.getlocalBodyIndex() == footIdLF_ - 6) // вычитание вызвано несовпадением индексов фреймов в методах getContacts() и getFrames()
                    isFootContactLF = true;
                if (contact.getlocalBodyIndex() == footIdRF_ - 6) // вычитание вызвано несовпадением индексов фреймов в методах getContacts() и getFrames()
                    isFootContactRF = true;
                if (contact.getlocalBodyIndex() == footIdLB_ - 6) // вычитание вызвано несовпадением индексов фреймов в методах getContacts() и getFrames()
                    isFootContactLB = true;
                if (contact.getlocalBodyIndex() == footIdRB_ - 6) // вычитание вызвано несовпадением индексов фреймов в методах getContacts() и getFrames()
                    isFootContactRB = true;
            }

            terminalReward = 0.0;
            if (isFootContactLF || isFootContactRF || isFootContactLB || isFootContactRB)
                return false;

            terminalReward = float(terminalRewardCoeff_);
            return true;
        }

        void curriculumUpdate() { };

    private:
        int gcDim_, gvDim_, nJoints_;
        bool visualizable_ = false;
        raisim::ArticulatedSystem* anymal_;
        Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_;
        double terminalRewardCoeff_ = -800;
        Eigen::VectorXd actionMean_, actionStd_, obDouble_;
        float stepHeight_;
        float stepWidth_;
        float targetAngularY_;
        float dx_ = 0.325;
        float dy_ = 0.14;
        size_t footIdLF_ = 9;
        size_t footIdRF_ = 12;
        size_t footIdLB_ = 15;
        size_t footIdRB_ = 18;
        int seed_;

        /// these variables are not in use. They are placed to show you how to create a random number sampler.
        std::normal_distribution<double> normDist_;
        thread_local static std::mt19937 gen_;
    };
    thread_local std::mt19937 raisim::ENVIRONMENT::gen_;

}
