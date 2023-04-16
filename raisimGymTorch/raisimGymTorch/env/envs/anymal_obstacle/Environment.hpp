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
	    world_->setDefaultMaterial(2.0, 0.0, 0.0);
            /// add objects
            anymal_ = world_->addArticulatedSystem(resourceDir_+"/aliengo/aliengo.urdf");
            anymal_->setName("anymal");
            anymal_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
            ///world_->addGround();
    	raisim::TerrainProperties terrainProperties;  // Randomized terrain
    	terrainProperties.frequency = 2.5;
    	terrainProperties.zScale = 0.25;
    	terrainProperties.xSize = 15.0;
    	terrainProperties.ySize = 15.0;
    	terrainProperties.xSamples = 200;
    	terrainProperties.ySamples = 200;
    	terrainProperties.fractalOctaves = 8;
    	terrainProperties.fractalLacunarity = 4.0;
    	terrainProperties.fractalGain = 0.9;
    	auto hm = world_->addHeightMap(0.0, 0.0, terrainProperties);

            
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
                    gc_init_ << 0.0, 0.0, 0.429+0.15, 1.0, 0.0, -0.02, 0.0, -0.05, 0.5, -1.2, 0.05, 0.5, -1.2, -0.05, 0.6, -1.3, 0.05, 0.6, -1.3;
                    break;
                case 1:
                    gc_init_ << 0.0, 0.0, 0.428+0.15, 1.0, 0.0, -0.005, 0.0, -0.0, 0.5, -1.2, 0.0, 0.5, -1.2, -0.0, 0.6, -1.3, 0.0, 0.6, -1.3;
                    break;
                case 2:
                    gc_init_ << 0.0, 0.0, 0.427+0.15, 1.0, 0.0, 0.0, 0.0, -0.0, 0.5, -1.2, 0.0, 0.5, -1.2, -0.0, 0.4, -1.2, 0.0, 0.4, -1.2;
                    break;
                case 3:
                    gc_init_ << 0.0, 0.0, 0.37+0.15, 1.0, 0.0, 0.0, 0.0, -0.0, 0.8, -1.6, 0.0, 0.8, -1.6, -0.0, 0.8, -1.6, 0.0, 0.8, -1.6;
                    break;
                case 4:
                    gc_init_ << 0.0, 0.0, 0.49+0.15, 1.0, 0.0, 0.0, 0.0, -0.0, 0.3, -0.7, 0.0, 0.3, -0.7, -0.0, 0.3, -0.7, 0.0, 0.3, -0.7;
                    break;
                case 5:
                    gc_init_ << 0.0, 0.0, 0.428+0.15, 1.0, 0.0, -0.005, 0.0, -0.0, 0.5, -1.2, 0.0, 0.5, -1.2, -0.0, 0.6, -1.3, 0.0, 0.6, -1.3;
                    break;
            }

            /// set pd gains
            Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
            jointPgain.setZero(); jointPgain.tail(nJoints_).setConstant(50.0);
            jointDgain.setZero(); jointDgain.tail(nJoints_).setConstant(0.5);
            anymal_->setPdGains(jointPgain, jointDgain);
            anymal_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

            /// MUST BE DONE FOR ALL ENVIRONMENTS
            obDim_ = 36;
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
                server_->launchServer(8090+seed_);
                server_->focusOn(anymal_);
            }

            flySteps_ = 0;
        }

        void init() final { }

        double normalizeReward(double x, double coeff) {
            return (5.0/coeff-x*x);
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

            raisim::Vec<3> footPositionLF; anymal_->getFramePosition(footIdLF_, footPositionLF);
            raisim::Vec<3> footPositionRF; anymal_->getFramePosition(footIdRF_, footPositionRF);
            raisim::Vec<3> footPositionLB; anymal_->getFramePosition(footIdLB_, footPositionLB);
            raisim::Vec<3> footPositionRB; anymal_->getFramePosition(footIdRB_, footPositionRB);

            raisim::Vec<3> footVelocityLF; anymal_->getFrameVelocity(footIdLF_, footVelocityLF);
            raisim::Vec<3> footVelocityRF; anymal_->getFrameVelocity(footIdRF_, footVelocityRF);
            raisim::Vec<3> footVelocityLB; anymal_->getFrameVelocity(footIdLB_, footVelocityLB);
            raisim::Vec<3> footVelocityRB; anymal_->getFrameVelocity(footIdRB_, footVelocityRB);

            double oneStepFront = 0;
            if (footVelocityLF[0] + abs(footVelocityLF[2]) > 0.5 && (footVelocityLF[0] - footVelocityRF[0] < 0.1))
                oneStepFront = normalizeReward(footVelocityRF[0] + abs(footVelocityRF[2]), rewards_.rewards_["oneStepFront"].coefficient);
            if (footVelocityRF[0] + abs(footVelocityRF[2]) > 0.5 && (footVelocityRF[0] - footVelocityLF[0] < 0.1))
                oneStepFront = normalizeReward(footVelocityLF[0] + abs(footVelocityLF[2]), rewards_.rewards_["oneStepFront"].coefficient);
            rewards_.record("oneStepFront", oneStepFront);

            double oneStepBack = 0;
            if (footVelocityRB[0] + abs(footVelocityRB[2]) > 0.5 && (footVelocityLB[0] - footVelocityRB[0] < 0.1))
                oneStepBack = normalizeReward(footVelocityLB[0] + abs(footVelocityLB[2]), rewards_.rewards_["oneStepBack"].coefficient);
            if (footVelocityLB[0] + abs(footVelocityLB[2]) > 0.5 && (footVelocityRB[0] - footVelocityLB[0] < 0.1))
                oneStepBack = normalizeReward(footVelocityRB[0] + abs(footVelocityRB[2]), rewards_.rewards_["oneStepBack"].coefficient);
            rewards_.record("oneStepBack", oneStepBack);


            if (footVelocityLF[0] > 0.05)
                rewards_.record("footVelocityFront", footVelocityLF[2]);
            if (footVelocityRF[0] > 0.05)
                rewards_.record("footVelocityFront", footVelocityRF[2]);
            if (footVelocityLB[0] > 0.05)
                rewards_.record("footVelocityBack", footVelocityLB[2]);
            if (footVelocityRB[0] > 0.05)
                rewards_.record("footVelocityBack", footVelocityRB[2]);


            double footDeviationY = normalizeReward(abs(gc_init_[7] - gc_[7]), rewards_.rewards_["footDeviationY"].coefficient)
                                  + normalizeReward(abs(gc_init_[10] - gc_[10]), rewards_.rewards_["footDeviationY"].coefficient)
                                  + normalizeReward(abs(gc_init_[13] - gc_[13]), rewards_.rewards_["footDeviationY"].coefficient)
                                  + normalizeReward(abs(gc_init_[16] - gc_[16]), rewards_.rewards_["footDeviationY"].coefficient);
            rewards_.record("footDeviationY", footDeviationY);


            double leftBackFeetDX = normalizeReward(gc_[0]-0.2*dx_-footPositionLB[0], rewards_.rewards_["leftBackFeetDX"].coefficient);
            rewards_.record("leftBackFeetDX", leftBackFeetDX);

            double rightBackFeetDX = normalizeReward(gc_[0]-0.2*dx_-footPositionRB[0], rewards_.rewards_["rightBackFeetDX"].coefficient);
            rewards_.record("rightBackFeetDX", rightBackFeetDX);

            double xVelocity = std::min(2.0, gv_[0]);
            rewards_.record("xVelocity", xVelocity);

            double xAngular = normalizeReward(gv_[3], rewards_.rewards_["xAngular"].coefficient);
            rewards_.record("xAngular", xAngular);

            double yVelocity = normalizeReward(gv_[1], rewards_.rewards_["yVelocity"].coefficient);
            rewards_.record("yVelocity", yVelocity);

            double yAngular = normalizeReward(gv_[4], rewards_.rewards_["yAngular"].coefficient);
            rewards_.record("yAngular", yAngular);

            double zAngular = normalizeReward(gv_[5], rewards_.rewards_["zAngular"].coefficient);
            rewards_.record("zAngular", zAngular);

            double xCoord = gc_[0];
            rewards_.record("xCoord", xCoord, true);

            double zCoord = normalizeReward(gc_[2]-gc_init_[2], rewards_.rewards_["zCoord"].coefficient);
            rewards_.record("zCoord", zCoord);

            static int counter = 0;
            if (counter++ % 98765 == 1)
            {
                std::cout << "CURRENT REWARDS:\n";

                std::cout << "oneStepFront: " << rewards_["oneStepFront"] << "\n";
                std::cout << "oneStepBack: " << rewards_["oneStepBack"] << "\n";

                std::cout << "footDeviationY: " << rewards_["footDeviationY"] << "\n";

                std::cout << "leftFrontFeetDX: " << rewards_["leftFrontFeetDX"] << "\n";
                std::cout << "rightFrontFeetDX: " << rewards_["rightFrontFeetDX"] << "\n";
                std::cout << "leftBackFeetDX: " << rewards_["leftBackFeetDX"] << "\n";
                std::cout << "rightBackFeetDX: " << rewards_["rightBackFeetDX"] << "\n";

                std::cout << "xVelocity: " << rewards_["xVelocity"] << "\n";
                std::cout << "xAngular: " << rewards_["xAngular"] << "\n";
                std::cout << "yVelocity: " << rewards_["yVelocity"] << "\n";
                std::cout << "yAngular: " << rewards_["yAngular"] << "\n";
                std::cout << "zAngular: " << rewards_["zAngular"] << "\n";
                std::cout << "xCoord: " << rewards_["xCoord"] << "\n";
                std::cout << "zCoord: " << rewards_["zCoord"] << "\n";

//                std::cout << "IS CONTACT:\n";
//                std::cout << isLeftFrontFeetContact << " " << isRightFrontFeetContact << " " << isLeftBackFeetContact << " " << isRightBackFeetContact << "\n";

                std::cout << "ACTIONS:\n";
                std::cout << action[0] << " " << action[1] << " " << action[2] << " " << action[3] << " "
                          << action[4] << " " << action[5] << " " << action[6] << " " << action[7] << " "
                          << action[8] << " " << action[9] << " " << action[10] << " " << action[11] << "\n";

                std::cout << "FOOT JOINT VELOS:\n";
                std::cout << gv_[6] << " " << gv_[7] << " " << gv_[8] << "\n";
                std::cout << gv_[9] << " " << gv_[10] << " " << gv_[11] << "\n";
                std::cout << gv_[12] << " " << gv_[13] << " " << gv_[14] << "\n";
                std::cout << gv_[15] << " " << gv_[16] << " " << gv_[17] << "\n";

                std::cout << "FOOT POSITION:\n";
                std::cout << footPositionLF[0] << " " << footPositionRF[0] << " " << footPositionLB[0] << " " << footPositionRB[0] << "\n";
                std::cout << footPositionLF[1] << " " << footPositionRF[1] << " " << footPositionLB[1] << " " << footPositionRB[1] << "\n";
                std::cout << footPositionLF[2] << " " << footPositionRF[2] << " " << footPositionLB[2] << " " << footPositionRB[2] << "\n";

                std::cout << "FOOT VELO:\n";
                std::cout << footVelocityLF[0] << " " << footVelocityRF[0] << " " << footVelocityLB[0] << " " << footVelocityRB[0] << "\n";
                std::cout << footVelocityLF[1] << " " << footVelocityRF[1] << " " << footVelocityLB[1] << " " << footVelocityRB[1] << "\n";
                std::cout << footVelocityLF[2] << " " << footVelocityRF[2] << " " << footVelocityLB[2] << " " << footVelocityRB[2] << "\n";

                std::cout.flush();
            }

            return rewards_.sum();
        }

        void updateObservation() {
            anymal_->getState(gc_, gv_);
            obDouble_ << gv_, gc_.head(3), gc_[4]/gc_[3], gc_[5]/gc_[3], gc_[6]/gc_[3], gc_.tail(12);
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
            if (!(gc_[2] > 0.3 && abs(gc_[6]/gc_[3]) < 0.4 && abs(gc_[4]/gc_[3]) < 0.4 && gc_[0] > -0.05))
            {
                terminalReward = float(terminalRewardCoeff_);
                return true;
            }

            return false;
            
	        bool isFootContactLF = false;
            bool isFootContactRF = false;
            bool isFootContactLB = false;
            bool isFootContactRB = false;

            for(auto& contact: anymal_->getContacts())
            {

                if ((contact.getlocalBodyIndex() == footIdLF_ - 6) && (contact.getPairObjectBodyType() == raisim::BodyType::STATIC) && (contact.getNormal()[2] != 0)) // вычитание вызвано несовпадением индексов фреймов в методах getContacts() и getFrames()
                    isFootContactLF = true;
                else if ((contact.getlocalBodyIndex() == footIdRF_ - 6) && (contact.getPairObjectBodyType() == raisim::BodyType::STATIC) && (contact.getNormal()[2] != 0)) // вычитание вызвано несовпадением индексов фреймов в методах getContacts() и getFrames()
                    isFootContactRF = true;
                else if ((contact.getlocalBodyIndex() == footIdLB_ - 6) && (contact.getPairObjectBodyType() == raisim::BodyType::STATIC) && (contact.getNormal()[2] != 0)) // вычитание вызвано несовпадением индексов фреймов в методах getContacts() и getFrames()
                    isFootContactLB = true;
                else if ((contact.getlocalBodyIndex() == footIdRB_ - 6) && (contact.getPairObjectBodyType() == raisim::BodyType::STATIC) && (contact.getNormal()[2] != 0)) // вычитание вызвано несовпадением индексов фреймов в методах getContacts() и getFrames()
                    isFootContactRB = true;
                else if ((contact.getPairObjectBodyType() == raisim::BodyType::STATIC) && (contact.getNormal()[2] != 0))
                {
                    terminalReward = float(terminalRewardCoeff_);
                    return true;
                }
            }

            terminalReward = 0.0;
            if (isFootContactLF || isFootContactRF || isFootContactLB || isFootContactRB)
                flySteps_ = 0;

            ++flySteps_;

            if (flySteps_ > 2)
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
        double terminalRewardCoeff_ = -100;
        Eigen::VectorXd actionMean_, actionStd_, obDouble_;
        double stepHeight_;
        double stepWidth_;
        double targetAngularY_;
        double dx_ = 0.2;
        double dy_ = 0.14;
        size_t footIdRF_ = 9;
        size_t footIdLF_ = 12;
        size_t footIdRB_ = 15;
        size_t footIdLB_ = 18;
        int seed_;
        int flySteps_;

        /// these variables are not in use. They are placed to show you how to create a random number sampler.
        std::normal_distribution<double> normDist_;
        thread_local static std::mt19937 gen_;
    };
    thread_local std::mt19937 raisim::ENVIRONMENT::gen_;

}
