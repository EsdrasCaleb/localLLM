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
        // Assuming a default constructor exists
        Contract contract = new Contract();
        int position = 10;
        double marketPrice = 100.50;
        double marketValue = 1005.00;
        double averageCost = 95.00;
        double unrealizedPNL = 55.00;
        double realizedPNL = 20.00;
        String accountName = "TestAccount";
        // Act
        String result = EWrapperMsgGenerator.updatePortfolio(contract, position, marketPrice, marketValue, averageCost, unrealizedPNL, realizedPNL, accountName);
        // Assert
        String expected = "updatePortfolio: " + EWrapperMsgGenerator.contractMsg(contract) + " " + position + " " + marketPrice + " " + marketValue + " " + averageCost + " " + unrealizedPNL + " " + realizedPNL + " " + accountName;
        assertEquals(expected, result);
    }
}
