package com.ib.client;

import com.ib.client.Contract;
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

    private Contract mockContract;

    @BeforeEach
    void setUp() {
        mockContract = mock(Contract.class);
        when(mockContract.toString()).thenReturn("MockContract");
    }

    @Test
    void testUpdatePortfolio() {
        int position = 100;
        double marketPrice = 150.5;
        double marketValue = 15050.0;
        double averageCost = 140.0;
        double unrealizedPNL = 1050.0;
        double realizedPNL = 500.0;
        String accountName = "TestAccount";
        String expectedMsg = "updatePortfolio: MockContract100 150.5 15050.0 140.0 1050.0 500.0 TestAccount";
        String actualMsg = EWrapperMsgGenerator.updatePortfolio(mockContract, position, marketPrice, marketValue, averageCost, unrealizedPNL, realizedPNL, accountName);
        assertEquals(expectedMsg, actualMsg);
    }
}
