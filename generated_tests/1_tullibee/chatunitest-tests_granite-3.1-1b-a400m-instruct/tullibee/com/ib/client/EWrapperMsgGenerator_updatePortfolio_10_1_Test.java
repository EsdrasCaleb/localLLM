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

class EWrapperMsgGenerator_updatePortfolio_10_1_Test {

    @Test
    void testUpdatePortfolio() {
        // Arrange
        Contract contract = new Contract();
        int position = 100;
        double marketPrice = 1000.0;
        double marketValue = 100000.0;
        double averageCost = 5000.0;
        double unrealizedPNL = 5000.0;
        double realizedPNL = 5000.0;
        String accountName = "Account 1";
        EWrapperMsgGenerator eWrapperMsgGenerator = new EWrapperMsgGenerator();
        // Act
        String msg = eWrapperMsgGenerator.updatePortfolio(contract, position, marketPrice, marketValue, averageCost, unrealizedPNL, realizedPNL, accountName);
        // Assert
        assertEquals("updatePortfolio: Contract: Contract, Position: 100, Market Price: 1000.0, Market Value: 100000.0, Average Cost: 5000.0, Unrealized P&L: 5000.0, Realized P&L: 5000.0, Account Name: Account 1", msg);
    }
}
