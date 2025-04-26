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

class EWrapperMsgGenerator_updatePortfolio_10_0_Test {

    @Test
    void testUpdatePortfolio() {
        // Arrange
        Contract contract = Mockito.mock(Contract.class);
        int position = 1;
        double marketPrice = 100.0;
        double marketValue = 200.0;
        double averageCost = 50.0;
        double unrealizedPNL = 10.0;
        double realizedPNL = 5.0;
        String accountName = "TestAccount";
        String expectedMsg = "updatePortfolio: Contract Mock, 1 100.0 200.0 50.0 10.0 5.0 TestAccount";
        // Act
        String actualMsg = EWrapperMsgGenerator.updatePortfolio(contract, position, marketPrice, marketValue, averageCost, unrealizedPNL, realizedPNL, accountName);
        // Assert
        assertEquals(expectedMsg, actualMsg);
    }
}
