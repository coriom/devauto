// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract Hello {
    function greet() external pure returns (string memory) {
        return "Hello, AutoDev!";
    }

    function autre() external pure returns (uint256) {
        return 42;
    }
}
