package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.text.DateFormat;
import java.util.Date;
import java.util.Vector;

public class EWrapperMsgGenerator_updatePortfolio_10_0_Test {

    @Test
    public void testUpdatePortfolio() {
        // Arrange
        Contract contract = new Contract();
        int position = 10;
        double marketPrice = 100.0;
        double marketValue = 500.0;
        double averageCost = 50.0;
        double unrealizedPNL = 20.0;
        double realizedPNL = 30.0;
        String accountName = "Test Account";
        // Act
        String result = EWrapperMsgGenerator.updatePortfolio(contract, position, marketPrice, marketValue, averageCost, unrealizedPNL, realizedPNL, accountName);
        // Assert
        assertEquals("updatePortfolio: ContractMsg(Contract)", result);
    }

    @Test
    public void testUpdatePortfolio_InvalidPosition() {
        // Arrange
        Contract contract = new Contract();
        int position = -10;
        double marketPrice = 100.0;
        double marketValue = 500.0;
        double averageCost = 50.0;
        double unrealizedPNL = 20.0;
        double realizedPNL = 30.0;
        String accountName = "Test Account";
        // Act and Assert
        String result = EWrapperMsgGenerator.updatePortfolio(contract, position, marketPrice, marketValue, averageCost, unrealizedPNL, realizedPNL, accountName);
        assertEquals("updatePortfolio: ContractMsg(Contract)", result);
    }

    @Test
    public void testUpdatePortfolio_InvalidMarketPrice() {
        // Arrange
        Contract contract = new Contract();
        int position = 10;
        double marketPrice = -100.0;
        double marketValue = 500.0;
        double averageCost = 50.0;
        double unrealizedPNL = 20.0;
        double realizedPNL = 30.0;
        String accountName = "Test Account";
        // Act and Assert
        String result = EWrapperMsgGenerator.updatePortfolio(contract, position, marketPrice, marketValue, averageCost, unrealizedPNL, realizedPNL, accountName);
        assertEquals("updatePortfolio: ContractMsg(Contract)", result);
    }

    @Test
    public void testUpdatePortfolio_InvalidMarketValue() {
        // Arrange
        Contract contract = new Contract();
        int position = 10;
        double marketPrice = 100.0;
        double marketValue = -500.0;
        double averageCost = 50.0;
        double unrealizedPNL = 20.0;
        double realizedPNL = 30.0;
        String accountName = "Test Account";
        // Act and Assert
        String result = EWrapperMsgGenerator.updatePortfolio(contract, position, marketPrice, marketValue, averageCost, unrealizedPNL, realizedPNL, accountName);
        assertEquals("updatePortfolio: ContractMsg(Contract)", result);
    }

    @Test
    public void testUpdatePortfolio_InvalidAverageCost() {
        // Arrange
        Contract contract = new Contract();
        int position = 10;
        double marketPrice = 100.0;
        double marketValue = 500.0;
        double unrealizedPNL = 20.0;
        double realizedPNL = 30.0;
        String accountName = "Test Account";
        // Act and Assert
        String result = EWrapperMsgGenerator.updatePortfolio(contract, position, marketPrice, marketValue, -50.0, unrealizedPNL, realizedPNL, accountName);
        assertEquals("updatePortfolio: ContractMsg(Contract)", result);
    }

    @Test
    public void testUpdatePortfolio_InvalidRealizedPNL() {
        // Arrange
        Contract contract = new Contract();
        int position = 10;
        double marketPrice = 100.0;
        double marketValue = 500.0;
        double averageCost = 50.0;
        double unrealizedPNL = -20.0;
        double realizedPNL = 30.0;
        String accountName = "Test Account";
        // Act and Assert
        String result = EWrapperMsgGenerator.updatePortfolio(contract, position, marketPrice, marketValue, averageCost, unrealizedPNL, realizedPNL, accountName);
        assertEquals("updatePortfolio: ContractMsg(Contract)", result);
    }
}
