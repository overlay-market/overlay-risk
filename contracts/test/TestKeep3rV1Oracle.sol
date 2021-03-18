// SPDX-License-Identifier: MIT
pragma solidity ^0.6.12;

/*
 * This Oracle serves as a mock for testing returning
 * hard coded sample data.
 */
contract TestKeep3rV1Oracle {
  uint[] public prices;

  function setPrices(uint[] calldata _prices) external {
    for (uint256 i = 0; i < _prices.length; i++) {
      prices[i] = _prices[i];
    }
  }

  function sample(address tokenIn, uint amountIn, address tokenOut, uint points, uint window) external view returns (uint[] memory) {
    return prices;
  }
}
