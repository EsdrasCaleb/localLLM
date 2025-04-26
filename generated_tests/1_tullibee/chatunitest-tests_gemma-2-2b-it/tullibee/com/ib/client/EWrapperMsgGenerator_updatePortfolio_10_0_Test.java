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
    void updatePortfolio() {
        // Arrange
        Contract contract = mock(Contract.class);
        int position = 10;
        double marketPrice = 100.0;
        double marketValue = 1000.0;
        double averageCost = 50.0;
        double unrealizedPNL = 50.0;
        double realizedPNL = 0.0;
        String accountName = "testAccount";
        // Act
        String actualMsg = EWrapperMsgGenerator.updatePortfolio(contract, position, marketPrice, marketValue, averageCost, unrealizedPNL, realizedPNL, accountName);
        // Assert
        assertEquals("updatePortfolio: SCANNER PARAMETERS: 10 100.0 1000.0 50.0 50.0 0.0 0.0 testAccount", actualMsg);
    }
}
