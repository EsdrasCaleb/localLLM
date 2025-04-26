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

    @Mock
    private Contract mockContract;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    void testUpdatePortfolio() {
        // Arrange
        int position = 10;
        double marketPrice = 100.0;
        double marketValue = 10000.0;
        double averageCost = 98.0;
        double unrealizedPNL = 200.0;
        double realizedPNL = 150.0;
        String accountName = "TestAccount";
        when(mockContract.toString()).thenReturn("Contract[conId=12345, symbol=IBM, secType=STK, exchange=SMART, currency=USD]");
        // Expected output
        String expectedMsg = "updatePortfolio: Contract[conId=12345, symbol=IBM, secType=STK, exchange=SMART, currency=USD]10 100.0 10000.0 98.0 200.0 150.0 TestAccount";
        // Act
        String result = EWrapperMsgGenerator.updatePortfolio(mockContract, position, marketPrice, marketValue, averageCost, unrealizedPNL, realizedPNL, accountName);
        // Assert
        assertEquals(expectedMsg, result);
    }
}
