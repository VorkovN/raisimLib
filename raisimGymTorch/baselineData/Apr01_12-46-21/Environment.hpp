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
	        world_->setDefaultMaterial(3.0, 0.0, 0.0);
            /// add objects
            anymal_ = world_->addArticulatedSystem(resourceDir_+"/aliengo/aliengo.urdf");
            anymal_->setName("anymal");
            anymal_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
            world_->addGround();

            srand (seed_);
//            stepHeight_ = float(std::rand()%3)/100+0.16;
//            stepWidth_ = float(std::rand()%4)/100+0.27;

            stepHeight_ = float(std::rand()%2)/100+0.16;
            stepWidth_ = float(std::rand()%2)/100+0.28;

            uint8_t stepCount = 6;
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
            targetAngularY_ = -1.5*atan(stepHeight_/stepWidth_)/3.14;

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
                    gc_init_ << 0.0, 0.0, 0.38, 1.0, 0.0, 0.0, 0.0, 0.0, 0.9, -1.5, 0.0, 0.9, -1.5, 0.0, 0.8, -1.6, 0.0, 0.8, -1.6; //обычная стойка
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
            jointPgain.setZero(); jointPgain.tail(nJoints_).setConstant(40.0);
            jointDgain.setZero(); jointDgain.tail(nJoints_).setConstant(1.0);
            anymal_->setPdGains(jointPgain, jointDgain);
            anymal_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

            /// MUST BE DONE FOR ALL ENVIRONMENTS
            obDim_ = 38;
//            obDim_ = 26;
            actionDim_ = nJoints_; actionMean_.setZero(actionDim_); actionStd_.setZero(actionDim_);
            obDouble_.setZero(obDim_);
            /// action scaling
            actionMean_ = gc_init_.tail(nJoints_);
            double action_std;
            READ_YAML(double, action_std, cfg_["action_std"]) /// example of reading params from the config
            actionStd_.setConstant(action_std);
            
            /// Reward coefficients
            rewards_.initializeFromConfigurationFile (cfg["reward"]);

            /// visualize if it is the first environment
            if (visualizable_) {
                server_ = std::make_unique<raisim::RaisimServer>(world_.get());
                server_->launchServer(8080+seed_);
                server_->focusOn(anymal_);
            }

            flySteps_ = 0;
        }

        float getGoalFootHeight(float x)
        {
            return std::max(float(0), static_cast<float>( stepHeight_ * static_cast<int>((x-dx_-0.4*stepWidth_) / stepWidth_ + 1) ) // целая ступень
                                    + static_cast<float>(pow(fmod((x-dx_-0.4*stepWidth_), stepWidth_), 4)) / static_cast<float>(pow(stepHeight_, 3))); // остаток в виде экспоненты
        }

        void init() final { }

        float normalizeReward(float x, float coeff) {
            return (1./coeff-x*x);
        }

        void reset() final {
            anymal_->setState(gc_init_, gv_init_);
            updateObservation();
            flySteps_ = 0;
        }

        float step(const Eigen::Ref<EigenVec>& action) final {

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

            raisim::Vec<3> footVelocityLF;
            anymal_->getFrameVelocity(footIdLF_, footVelocityLF);
            raisim::Vec<3> footVelocityRF;
            anymal_->getFrameVelocity(footIdRF_, footVelocityRF);
            raisim::Vec<3> footVelocityLB;
            anymal_->getFrameVelocity(footIdLB_, footVelocityLB);
            raisim::Vec<3> footVelocityRB;
            anymal_->getFrameVelocity(footIdRB_, footVelocityRB);

            bool isLeftFrontFeetContact = false;
            bool isRightFrontFeetContact = false;
            bool isLeftBackFeetContact = false;
            bool isRightBackFeetContact = false;
            auto& contacts = anymal_->getContacts();
            for (auto& contact: contacts)
            {
                if ((contact.getlocalBodyIndex() == footIdLF_ - 6) && (contact.getPairObjectBodyType() == raisim::BodyType::STATIC))
                    isLeftFrontFeetContact = true;

                if ((contact.getlocalBodyIndex() == footIdRF_ - 6) && (contact.getPairObjectBodyType() == raisim::BodyType::STATIC))
                    isRightFrontFeetContact = true;

                if ((contact.getlocalBodyIndex() == footIdLB_ - 6) && (contact.getPairObjectBodyType() == raisim::BodyType::STATIC))
                    isLeftBackFeetContact = true;

                if ((contact.getlocalBodyIndex() == footIdRB_ - 6) && (contact.getPairObjectBodyType() == raisim::BodyType::STATIC))
                    isRightBackFeetContact = true;
            }

            float footDeviationY = normalizeReward(gc_[7], rewards_.rewards_["footDeviationY"].coefficient)
                                 + normalizeReward(gc_[10], rewards_.rewards_["footDeviationY"].coefficient)
                                 + normalizeReward(gc_[13], rewards_.rewards_["footDeviationY"].coefficient)
                                 + normalizeReward(gc_[16], rewards_.rewards_["footDeviationY"].coefficient);
            rewards_.record("footDeviationY", footDeviationY);

            float leftFrontFeetDX = normalizeReward(gc_[0]+1.8*dx_-footPositionLF[0], rewards_.rewards_["leftFrontFeetDX"].coefficient);
            rewards_.record("leftFrontFeetDX", leftFrontFeetDX);

            float leftBackFeetDX = normalizeReward(gc_[0]-0.4*dx_-footPositionLB[0], rewards_.rewards_["leftBackFeetDX"].coefficient);
            rewards_.record("leftBackFeetDX", leftBackFeetDX);

            float rightBackFeetDX = normalizeReward(gc_[0]-footPositionRB[0], rewards_.rewards_["rightBackFeetDX"].coefficient);
            rewards_.record("rightBackFeetDX", rightBackFeetDX);

            float rightFrontFeetDX = normalizeReward(gc_[0]+1.5*dx_-footPositionRF[0], rewards_.rewards_["rightFrontFeetDX"].coefficient);
            rewards_.record("rightFrontFeetDX", rightFrontFeetDX);

            float leftFrontFeetDZ = normalizeReward(getGoalFootHeight(footPositionLF[0]) - footPositionLF[2], rewards_.rewards_["leftFrontFeetDZ"].coefficient);
            rewards_.record("leftFrontFeetDZ", leftFrontFeetDZ);

            float rightFrontFeetDZ = normalizeReward(getGoalFootHeight(footPositionRF[0]) - footPositionRF[2], rewards_.rewards_["rightFrontFeetDZ"].coefficient);
            rewards_.record("rightFrontFeetDZ", rightFrontFeetDZ);

            float leftBackFeetDZ = normalizeReward(getGoalFootHeight(footPositionLB[0]) - footPositionLB[2], rewards_.rewards_["leftBackFeetDZ"].coefficient);
            rewards_.record("leftBackFeetDZ", leftBackFeetDZ);

            float rightBackFeetDZ = normalizeReward(getGoalFootHeight(footPositionRB[0]) - footPositionRB[2], rewards_.rewards_["rightBackFeetDZ"].coefficient);
            rewards_.record("rightBackFeetDZ", rightBackFeetDZ);

            float leftFrontFeetFold = 0;
            if (!isLeftFrontFeetContact && footVelocityLF[2] > 0)
                leftFrontFeetFold = normalizeReward(2+gc_[9], rewards_.rewards_["leftFrontFeetFold"].coefficient);
            rewards_.record("leftFrontFeetFold", leftFrontFeetFold);

            float rightFrontFeettFold = 0;
            if (!isRightFrontFeetContact && footVelocityRF[2] > 0)
                rightFrontFeettFold = normalizeReward(2+gc_[12], rewards_.rewards_["rightFrontFeettFold"].coefficient);
            rewards_.record("rightFrontFeettFold", rightFrontFeettFold);

            float leftBackFeetFold = 0;
            if (!isLeftBackFeetContact && footVelocityLB[2] > 0)
                leftBackFeetFold = normalizeReward(2+gc_[15], rewards_.rewards_["leftBackFeetFold"].coefficient);
            rewards_.record("leftBackFeetFold", leftBackFeetFold);

            float rightBackFeetFold = 0;
            if (!isRightBackFeetContact && footVelocityRB[2] > 0)
                rightBackFeetFold = normalizeReward(2+gc_[18], rewards_.rewards_["rightBackFeetFold"].coefficient);
            rewards_.record("rightBackFeetFold", rightBackFeetFold);


            float torque = anymal_->getGeneralizedForce().squaredNorm();
            rewards_.record("torque", torque);

            float xVelocity = std::min(1.0, gv_[0]);
            rewards_.record("xVelocity", xVelocity);

            float xAngular = normalizeReward(gc_[4]/gc_[3], rewards_.rewards_["xAngular"].coefficient);
            rewards_.record("xAngular", xAngular);

            float yVelocity = normalizeReward(gv_[1], rewards_.rewards_["yVelocity"].coefficient);
            rewards_.record("yVelocity", yVelocity);

            float yAngular = normalizeReward(gc_[5]/gc_[3]-targetAngularY_, rewards_.rewards_["yAngular"].coefficient);
            rewards_.record("yAngular", yAngular);

            float zAngular = normalizeReward(gc_[6]/gc_[3], rewards_.rewards_["zAngular"].coefficient);
            rewards_.record("zAngular", zAngular);

            float xCoord = gc_[0];
            rewards_.record("xCoord", xCoord, true);

            float zCoord = gc_[2]-gc_init_[2];
            rewards_.record("zCoord", zCoord, true);

            static int counter = 0;
            if (counter++ % 98765 == 1)
            {
                std::cout << "CURRENT REWARDS:\n";
                std::cout << "footDeviationY: " << rewards_["footDeviationY"] << "\n";
                std::cout << "leftFrontFeetDX: " << rewards_["leftFrontFeetDX"] << "\n";
                std::cout << "rightFrontFeetDX: " << rewards_["rightFrontFeetDX"] << "\n";
                std::cout << "leftBackFeetDX: " << rewards_["leftBackFeetDX"] << "\n";
                std::cout << "rightBackFeetDX: " << rewards_["rightBackFeetDX"] << "\n";

                std::cout << "leftFrontFeetDZ: " << rewards_["leftFrontFeetDZ"] << "\n";
                std::cout << "rightFrontFeetDZ: " << rewards_["rightFrontFeetDZ"] << "\n";
                std::cout << "leftBackFeetDZ: " << rewards_["leftBackFeetDZ"] << "\n";
                std::cout << "rightBackFeetDZ: " << rewards_["rightBackFeetDZ"] << "\n";

                std::cout << "leftFrontFeetFold: " << rewards_["leftFrontFeetFold"] << "\n";
                std::cout << "rightFrontFeettFold: " << rewards_["rightFrontFeettFold"] << "\n";
                std::cout << "leftBackFeetFold: " << rewards_["leftBackFeetFold"] << "\n";
                std::cout << "rightBackFeetFold: " << rewards_["rightBackFeetFold"] << "\n";

                std::cout << "torque: " << rewards_["torque"] << "\n";
                std::cout << "xVelocity: " << rewards_["xVelocity"] << "\n";
                std::cout << "xAngular: " << rewards_["xAngular"] << "\n";
                std::cout << "yVelocity: " << rewards_["yVelocity"] << "\n";
                std::cout << "yAngular: " << rewards_["yAngular"] << "\n";
                std::cout << "zAngular: " << rewards_["zAngular"] << "\n";
                std::cout << "xCoord: " << rewards_["xCoord"] << "\n";
                std::cout << "zCoord: " << rewards_["zCoord"] << "\n";

                std::cout << "ANGULAR:\n";
                std::cout << targetAngularY_ << " " << gc_[3] << " " << gc_[4] << " " << gc_[5] << " " << gc_[6] << "\n";

                std::cout << "ACTIONS:\n";
                std::cout << action[0] << " " << action[1] << " " << action[2] << " " << action[3] << " "
                          << action[4] << " " << action[5] << " " << action[6] << " " << action[7] << " "
                          << action[8] << " " << action[9] << " " << action[10] << " " << action[11] << "\n";
                          
                std::cout << "FOOT POSITION:\n";
                std::cout << footPositionLF[0] << " " << footPositionRF[0] << " " << footPositionLB[0] << " " << footPositionRB[0] << "\n";
                std::cout << footPositionLF[1] << " " << footPositionRF[1] << " " << footPositionLB[1] << " " << footPositionRB[1] << "\n";
                std::cout << footPositionLF[2] << " " << footPositionRF[2] << " " << footPositionLB[2] << " " << footPositionRB[2] << "\n";
                
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

                if ((contact.getlocalBodyIndex() == footIdLF_ - 6) && (contact.getPairObjectBodyType() == raisim::BodyType::STATIC)) // вычитание вызвано несовпадением индексов фреймов в методах getContacts() и getFrames()
                    isFootContactLF = true;
                if ((contact.getlocalBodyIndex() == footIdRF_ - 6) && (contact.getPairObjectBodyType() == raisim::BodyType::STATIC)) // вычитание вызвано несовпадением индексов фреймов в методах getContacts() и getFrames()
                    isFootContactRF = true;
                if ((contact.getlocalBodyIndex() == footIdLB_ - 6) && (contact.getPairObjectBodyType() == raisim::BodyType::STATIC)) // вычитание вызвано несовпадением индексов фреймов в методах getContacts() и getFrames()
                    isFootContactLB = true;
                if ((contact.getlocalBodyIndex() == footIdRB_ - 6) && (contact.getPairObjectBodyType() == raisim::BodyType::STATIC)) // вычитание вызвано несовпадением индексов фреймов в методах getContacts() и getFrames()
                    isFootContactRB = true;
            }

            terminalReward = 0.0;
            if (isFootContactLF || isFootContactRF || isFootContactLB || isFootContactRB)
                flySteps_ = 0;

            ++flySteps_;

            if (flySteps_ >= 4)
            {
                terminalReward = float(terminalRewardCoeff_);
                return true;
            }

            return false;
        }

        void curriculumUpdate() { };

    private:
        int gcDim_, gvDim_, nJoints_;
        bool visualizable_ = false;
        raisim::ArticulatedSystem* anymal_;
        Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_;
        double terminalRewardCoeff_ = -1000;
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
        int flySteps_;

        /// these variables are not in use. They are placed to show you how to create a random number sampler.
        std::normal_distribution<double> normDist_;
        thread_local static std::mt19937 gen_;
    };
    thread_local std::mt19937 raisim::ENVIRONMENT::gen_;

}
