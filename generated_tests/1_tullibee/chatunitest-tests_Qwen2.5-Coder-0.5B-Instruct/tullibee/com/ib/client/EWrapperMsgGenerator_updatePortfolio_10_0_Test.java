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
        Contract contract = mock(Contract.class);
        int position = 100;
        double marketPrice = 123.456;
        double marketValue = 789.0;
        double averageCost = 987.654;
        double unrealizedPNL = 567.890;
        double realizedPNL = 456.789;
        String accountName = "John Doe";
        // Act
        String result = EWrapperMsgGenerator.updatePortfolio(contract, position, marketPrice, marketValue, averageCost, unrealizedPNL, realizedPNL, accountName);
        // Assert
        assertEquals("updatePortfolio: " + "100 " + "123.456" + "789.0" + "987.654" + "567.890" + "456.789" + "John Doe", result);
    }
}
